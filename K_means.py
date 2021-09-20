import numpy as np
import torch
from torch.optim import Adam
import gym
import time

from sklearn.cluster import KMeans
import spinup.algos.pytorch.acdf_cuda_mine.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch_cuda import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools_cuda import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

from spinup.algos.pytorch.acdf_cuda_mine.demo_env import DemoGymEnv

def Variable(var):
    return var.to(device)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class ACDFBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.std_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, std=0.):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.std_buf[self.ptr] = std
        self.ptr += 1

    def delete(self, num):
        """
        Delete one timestep of agent-environment interaction with index num in the buffer.
        """
        for i in range(num,self.ptr-1):
            self.obs_buf[i] = self.obs_buf[i+1]
            self.act_buf[i] = self.act_buf[i+1]
            self.rew_buf[i] = self.rew_buf[i+1]
            self.val_buf[i] = self.val_buf[i+1]
            self.logp_buf[i] = self.logp_buf[i+1]
            self.std_buf[i] = self.std_buf[i+1]
        self.ptr -= 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, std=self.std_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k,v in data.items()}



def acdf(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, pi_epochs=20, vf_epochs=20, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10, demo_file=""):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # demo environment
    demo_env = DemoGymEnv(demo_file=demo_file, seed=seed)
    demo_env.check_env(env)
    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)

    # Sync params across processes
    sync_params(ac)
    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v, ac.v_pi])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d and v_pi: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = ACDFBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = Variable(data['obs']), Variable(data['act']), Variable(data['adv']), Variable(data['logp'])

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32, device=device).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = Variable(data['obs']), Variable(data['ret'])
        return ((ac.v(obs) - ret)**2).mean()
    def compute_loss_v_pi(data):
        obs, ret = Variable(data['obs']), Variable(data['ret'])
        return ((ac.v_pi(obs) - ret)**2).mean()
    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    vf_pi_optimizer = Adam(ac.v_pi.parameters(), lr=vf_lr)
    # Set up model savingF
    logger.setup_pytorch_saver(ac)
    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    def demo_update():
        data = buf.get()
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v_pi(data).item()
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                # logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()
        # logger.store(StopIter=i)
        for i in range(train_v_iters):
            vf_pi_optimizer.zero_grad()
            loss_v = compute_loss_v_pi(data)
            loss_v.backward()
            mpi_avg_grads(ac.v_pi)
            vf_pi_optimizer.step()
        print("Pi loss:     {}".format(pi_l_old))
        # kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        # logger.store(LossPi=pi_l_old, LossV=v_l_old,
        #              KL=kl, Entropy=ent, ClipFrac=cf,
        #              DeltaLossPi=(loss_pi.item() - pi_l_old),
        #              DeltaLossV=(loss_v.item() - v_l_old))

    def update_vf():
        data = buf.get()
        v_l_old = compute_loss_v(data).item()
        print("Loss for Value function: {}".format(v_l_old))
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)
            vf_optimizer.step()

    # pretraining epochs
    # pi_epochs, vf_epochs = 100, 50

    # demonstration training: main loop, for policy network
    o, ep_ret, ep_len = demo_env.reset(), 0, 0
    start_time = time.time()
    print('local_steps_per_epoch',local_steps_per_epoch)
    for epoch in range(pi_epochs):
        print("pi_Epoch",epoch)
        for t in range(local_steps_per_epoch):
            a, v, logp_a, m, std = ac.pretrain_step(torch.as_tensor(o, dtype=torch.float32, device=device))
            next_o, r, d, _ = demo_env.step(a, std)
            ep_ret += r
            ep_len += 1

            buf.store(o, a, r, v, logp_a, std=std)
            # logger.store(VVals=v)
            o = next_o
            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1
            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _, _, _ = ac.pretrain_step(torch.as_tensor(o, dtype=torch.float32, device=device))
                else:
                    v = 0
                # if terminal:
                #     # only save EpRet / EpLen if trajectory finished
                #     # logger.store(EpRet=ep_ret, EpLen=ep_len)
                buf.finish_path(v)
                o, ep_ret, ep_len = demo_env.reset(), 0, 0
        demo_update()

    # for the value function pre-training
    o, ep_ret, ep_len = demo_env.reset(), 0, 0
    start_time = time.time()
    for epoch in range(vf_epochs):
        print("vf_Epoch",epoch)
        for t in range(local_steps_per_epoch):
            next_o, r, d, _, a = demo_env.free_step()
            v = ac.v(torch.as_tensor(o, dtype=torch.float32, device=device)).cpu().detach().numpy()
            ep_ret += r
            ep_len += 1
            buf.store(o, a, r, v, 1)
            # logger.store(VVals=v)
            o = next_o
            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1
            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    v = ac.v(torch.as_tensor(o, dtype=torch.float32, device=device)).cpu().detach().numpy()
                else:
                    v = 0
                buf.finish_path(v)
                o, ep_ret, ep_len = demo_env.reset(), 0, 0
        print("Pretraining for value function at Epoch: {}".format(epoch))
        update_vf()


    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = demo_env.reset(), 0, 0
    buf = ACDFBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    diedown = 0.005
    last_k = 0
    # Main loop: collect experience in env and update/log each epoch
    Obs_data = np.zeros([int(local_steps_per_epoch*diedown),27], dtype = float)
    Act_data = np.zeros([int(local_steps_per_epoch*diedown),act_dim[0]], dtype = float)
    Rew_data = np.zeros([int(local_steps_per_epoch*diedown),1], dtype = float)

    for epoch in range(epochs):
        # diedown = diedown * 0.9
        # if(diedown<0.01): diedown= 0.01
        #
        # diedown=1
        # print('diedown: ',diedown)

        demon_buffer = int(local_steps_per_epoch*diedown)
        #buf = ACDFBuffer(obs_dim, act_dim, local_steps_per_epoch+int(local_steps_per_epoch*diedown), gamma, lam)
        #buf = ACDFBuffer(obs_dim, act_dim, int(local_steps_per_epoch*diedown), gamma, lam)
        for j in range(demon_buffer):
            next_o, r, logp_a, d, _, a = demo_env.Buffer_step()

            #_, v, _, m, std = ac.Buffer_Demon_step(torch.as_tensor(o, dtype=torch.float32, device=device),torch.as_tensor(a, dtype=torch.float32, device=device))
            v = ac.v(torch.as_tensor(o, dtype=torch.float32, device=device)).cpu().detach().numpy()
            #print((r, v, logp_a))
            ep_ret += r
            ep_len += 1

            #buf.store(o, a, r, v, logp_a)
            #print('o:::',o[0:27])
            Obs_data[j] = o[0:27]
            Act_data[j] = a
            Rew_data[j] = r
            # logger.store(VVals=v)
            o = next_o
            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = j == demon_buffer - 1
            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    v = ac.v(torch.as_tensor(o, dtype=torch.float32, device=device)).cpu().detach().numpy()
                else:
                    v = 0
                buf.finish_path(v)
                print("ep_ret:",ep_ret)
                o, ep_ret, ep_len = demo_env.reset(), 0, 0

        #print('Obs_data',Obs_data)
        #print('Act_data',Act_data)
        #print('Rew_data',Rew_data)

        Demon_Rew_Min = np.min(Rew_data)
        Demon_Rew_Max = np.max(Rew_data)
        Rew_data_normalised = (Rew_data - Demon_Rew_Min) / (Demon_Rew_Max - Demon_Rew_Min)

        Demon_Act_Min = np.min(Act_data)
        Demon_Act_Max = np.max(Act_data)
        Act_data_normalised = (Act_data - Demon_Act_Min) / (Demon_Act_Max - Demon_Act_Min)

        Demon_Obs_Min = np.min(Obs_data)
        Demon_Obs_Max = np.max(Obs_data)
        Obs_data_normalised = (Obs_data - Demon_Obs_Min) / (Demon_Obs_Max - Demon_Obs_Min)

        Demonstration_data = np.concatenate((Obs_data_normalised, Act_data_normalised), axis=1)
        #print('SHAPE_act:::',Act_data_normalised.shape)
        #print('SHAPE_obs:::',Obs_data_normalised.shape)
        #print('Demonstration_data:::',Demonstration_data)

        #构造聚类器
        num_of_clusters = int(Demonstration_data.shape[1]/5)
        estimator = KMeans(n_clusters=num_of_clusters)#构造聚类器
        estimator.fit(Demonstration_data)#聚类
        label_pred = estimator.labels_ #获取聚类标签
        centroids = estimator.cluster_centers_ #获取聚类中心
        inertia = estimator.inertia_ # 获取聚类准则的总和
        #print('label_pred',label_pred)
        #print('centroids',centroids.shape)
        #print('inertia',inertia)
        reward_estimation = np.zeros((num_of_clusters,1), dtype=float)
        for j in range(0,num_of_clusters):
            reward_estimation[j] = np.sum(Rew_data_normalised[np.where(label_pred==j)])

        Rew_Est_Min = np.min(reward_estimation)
        Rew_Est_Max = np.max(reward_estimation)
        Reward_Estimation_normalised = (reward_estimation - Rew_Est_Min) / (Rew_Est_Max - Rew_Est_Min)
        print('Reward_Estimation_normalised',Reward_Estimation_normalised)
        #print('PREDICTION:::', estimator.predict([Demonstration_data[0],]))

        # Exploration and Exploitation
        o, ep_ret, ep_len = env.reset(), 0, 0
        #for t in range(local_steps_per_epoch):
        last_k = 0
        while(buf.ptr < buf.max_size):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32, device=device))

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # test whether the collected data is good enough
            #o_norm = (o[0:27] - Demon_Obs_Min) / (Demon_Obs_Max - Demon_Obs_Min)
            #a_norm = (a - Demon_Act_Min) / (Demon_Act_Max - Demon_Act_Min)
            #print('o_norm',o_norm)
            #print('a_norm',a_norm)
            #print(estimator.predict([np.concatenate((o_norm, a_norm), axis=0),]))
            #if(Reward_Estimation_normalised[estimator.predict([np.concatenate((o_norm, a_norm), axis=0),])] > 0.3):
                # save and log
            buf.store(o, a, r, v, logp)
            logger.store(VVals=v)
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = buf.ptr == buf.max_size

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32, device=device))
                else:
                    v = 0
                buf.finish_path(v)
                k = last_k
                while True:
                    if buf.max_size - buf.ptr < 15:
                        break
                    o_norm = (buf.obs_buf[k][0:27] - Demon_Obs_Min) / (Demon_Obs_Max - Demon_Obs_Min)
                    a_norm = (buf.act_buf[k] - Demon_Act_Min) / (Demon_Act_Max - Demon_Act_Min)
                    if(Reward_Estimation_normalised[estimator.predict([np.concatenate((o_norm, a_norm), axis=0),])] < 0.2):
                        buf.delete(k)
                        k -= 1
                    k += 1
                    if k == buf.ptr:
                        break
                last_k = buf.ptr
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)
        print('PTR',buf.ptr)
        #print('MAXSIZE',buf.max_size)
        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--env', type=str, default='Ant-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=40000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--exp_name', type=str, default='EXP4')
    parser.add_argument('--demo-file', type=str, default='Ant_Bill_100.pickle')
    parser.add_argument('--pi-epochs', type=int, default=0)
    parser.add_argument('--vf-epochs', type=int, default=0)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    from spinup.utils.run_utils import setup_logger_kwargs
    for args.seed in range(0,10):
        logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
        acdf(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        pi_epochs=args.pi_epochs, vf_epochs=args.vf_epochs,
        logger_kwargs=logger_kwargs, demo_file=args.demo_file)

