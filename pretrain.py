import inspect
from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time


import spinup.algos.pytorch.acdf_cuda.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch_cuda import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools_cuda import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup.algos.pytorch.acdf.demo_env import DemoGymEnv
from spinup.algos.pytorch.acdf_cuda.acdf import Variable, ACDFBuffer, acdf, device
from spinup.algos.pytorch.acdf_cuda.custom_saver import save_pi, save_vf

SAVE_FREQ = [0, 1, 5, 10, 15, 20, 30, 40, 50, 100, 200, 500]

def pretrain(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
                steps_per_epoch=4000, epochs=50, pi_epochs=100, vf_epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
                vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
                target_kl=0.01, logger_kwargs=dict(), save_freq=10, demo_file=""):

    setup_pytorch_for_mpi()
    logger = EpochLogger(**logger_kwargs)
    # locals() return all local variable
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # demo environment
    demo_env = DemoGymEnv(demo_file=demo_file, seed=seed)
    demo_env.check_env(env)
    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)

    sync_params(ac)
    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = ACDFBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    vf_pi_optimizer = Adam(ac.v_pi.parameters(), lr=vf_lr)

    logger.setup_pytorch_saver(ac)
    def compute_loss_v(data):
        obs, ret = Variable(data['obs']), Variable(data['ret'])
        return ((ac.v(obs) - ret)**2).mean()
    def compute_loss_v_pi(data):
        obs, ret = Variable(data['obs']), Variable(data['ret'])
        return ((ac.v_pi(obs) - ret)**2).mean()
    def demo_update():
        data = buf.get()
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()
        pi_info, loss_pi, loss_v = {}, 0, 0
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
        logger.store(StopIter=i)
        for i in range(train_v_iters):
            vf_pi_optimizer.zero_grad()
            loss_v = compute_loss_v_pi(data)
            loss_v.backward()
            mpi_avg_grads(ac.v_pi)
            vf_pi_optimizer.step()
        print("Pi loss:     {}".format(pi_l_old))
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

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

    buf = ACDFBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    # pretraining epochs
    # demonstration training: main loop, for policy network
    o, ep_ret, ep_len = demo_env.reset(), 0, 0
    start_time = time.time()
    for epoch in range(pi_epochs):

        pi_old_data = [deepcopy(p.data) for p in ac.pi.parameters()]
        vf_old_data = [deepcopy(p.data) for p in ac.v.parameters()]
        vf_pi_old_data = [deepcopy(p.data) for p in ac.v_pi.parameters()]
        for t in range(local_steps_per_epoch):
            a, v, logp_a, m, std = ac.pretrain_step(torch.as_tensor(o, dtype=torch.float32, device=device))
            next_o, r, d, _ = demo_env.step(a, std)
            ep_ret += r
            ep_len += 1

            buf.store(o, a, r, v, logp_a, std=std)
            logger.store(VVals=v)
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
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                buf.finish_path(v)
                o, ep_ret, ep_len = demo_env.reset(), 0, 0


        # Save model
        # if (epoch % save_freq == 0) or (epoch == pi_epochs-1):
        if (epoch in SAVE_FREQ) or (epoch == pi_epochs - 1):
            save_pi(logger_kwargs.get('output_dir', "model"), itr=epoch, paramenters=ac.pi)
            logger.save_state({'env': env}, None)
        demo_update()
        delta_v, delta_v_pi, delta_pi = 0, 0, 0
        for i, param in enumerate(ac.v_pi.parameters()):
            delta_v_pi += torch.norm(param.data - vf_pi_old_data[i])
        for i, param in enumerate(ac.v.parameters()):
            delta_v += torch.norm(param.data - vf_old_data[i])
        for i, param in enumerate(ac.pi.parameters()):
            delta_pi += torch.norm(param.data - pi_old_data[i])
        print("delta v_pi: {}; delta vf: {}; delta pi: {}".format(delta_v_pi, delta_v, delta_pi))
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()
    logger.save_state({'env': env}, pi_epochs)
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
    # for the value function pre-training
    o, ep_ret, ep_len = demo_env.reset(), 0, 0
    start_time = time.time()
    for epoch in range(vf_epochs):
        pi_old_data = [deepcopy(p.data) for p in ac.pi.parameters()]
        vf_old_data = [deepcopy(p.data) for p in ac.v.parameters()]
        vf_pi_old_data = [deepcopy(p.data) for p in ac.v_pi.parameters()]
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
        delta_v, delta_v_pi, delta_pi = 0, 0, 0
        for i, param in enumerate(ac.v_pi.parameters()):
            delta_v_pi += torch.norm(param.data - vf_pi_old_data[i])
        for i, param in enumerate(ac.v.parameters()):
            delta_v += torch.norm(param.data - vf_old_data[i])
        for i, param in enumerate(ac.pi.parameters()):
            delta_pi += torch.norm(param.data - pi_old_data[i])
        print("delta v_pi: {}; delta vf: {}; delta pi: {}".format(delta_v_pi, delta_v, delta_pi))
        if (epoch in SAVE_FREQ) or (epoch == vf_epochs - 1):
            save_vf(logger_kwargs.get('output_dir', "model"), itr=epoch, paramenters=ac.v)
            logger.save_state({'env': env}, None)

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def argprinter(**kwargs):
    for k in kwargs:
        print(k)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--env', type=str, default='Ant-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=40000)
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--demo-file', type=str, default='data/Ant50epoch.pickle')
    parser.add_argument('--pi-epochs', type=int, default=10)
    parser.add_argument('--vf-epochs', type=int, default=10)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    pretrain(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
             ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
             seed=args.seed, steps_per_epoch=args.steps,
             pi_epochs=args.pi_epochs, vf_epochs=args.vf_epochs,
             logger_kwargs=logger_kwargs, demo_file=args.demo_file)
