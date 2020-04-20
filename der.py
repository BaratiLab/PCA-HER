import os

import click
import numpy as np
import json
from mpi4py import MPI

from baselines import logger
from baselines.common import set_global_seeds, tf_util
from baselines.common.mpi_moments import mpi_moments
import baselines.der.experiment.config as config
from baselines.der.rollout import RolloutWorker

def mpi_average(value, dtype=np.float32):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    #return mpi_moments(np.array(value))[0]
    if hasattr(value[0], "dtype"):
        dtype = value[0].dtype
    return mpi_moments(np.array(value, dtype=dtype))[0]


def train(*, policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_path, demo_file, hv_file, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    if save_path:
        latest_policy_path = os.path.join(save_path, 'policy_latest.pkl')
        best_policy_path = os.path.join(save_path, 'policy_best.pkl')
        periodic_policy_path = os.path.join(save_path, 'policy_{}.pkl')
    
    experiment_tag = os.path.basename(save_path) if save_path else None

    logger.info("Training...")
    best_success_rate = -1

    if policy.bc_loss == 1: policy.init_demo_buffer(demo_file) #initialize demo buffer if training with demonstrations
    if policy.hv_buffer_enabled: policy.init_HV_buffer(hv_file) # Initialize HV Buffer
    
    # num_timesteps = n_epochs * n_cycles * rollout_length * number of rollout workers
    for epoch in range(n_epochs):
        
        # train
        rollout_worker.clear_history()
        for _ in range(n_cycles):
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            for _ in range(n_batches):
                policy.train()
            policy.update_target_net()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()
        
        # State of hvb functionality - ADDED BY JUAN 10/17/2019
        if policy.hv_buffer_enabled:
            if policy.hv_buffer_shutoff > 0:
                if policy.hv_buffer_shutoff >= policy.total_episode_count:
                     hv_buffer_online = True
                else:
                     hv_buffer_online = False
            else:
                hv_buffer_online = True
        else:
             hv_buffer_online = False
        # record logs
        logger.record_tabular('epoch', epoch)
        logger.record_tabular('epoch_completion', '%.2f%%'%(100*(epoch+1)/n_epochs))
        logger.record_tabular('hvb_online',hv_buffer_online) # ADDED BY JUAN 10/17/2019
        logger.record_tabular('hvb_updates',policy.hv_update_enabled) # ADDED BY JUAN 10/17/2019
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            if experiment_tag: print('-'*36,'\nReport for:',experiment_tag)
            logger.dump_tabular()
            
        

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate > best_success_rate and save_path:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            policy.save(best_policy_path) #evaluator.save_policy(best_policy_path)
            policy.save(latest_policy_path) #evaluator.save_policy(latest_policy_path)
            #policy.save_buffer(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_path:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            policy.save(policy_path) #evaluator.save_policy(policy_path)
            policy.save_buffer(latest_policy_path, er = False, hv = True)
        
        if policy.hv_update_enabled: #params['hv_update_enabled']:
            if epoch % policy.hv_nep == 0 and epoch != 0: policy.update_HV_buffer()
            
        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]
    
    return policy


def learn(*, network, env, total_timesteps,
    seed=None,
    eval_env=None,
    replay_strategy='future',
    policy_save_interval=5,
    clip_return=True,
    demo_file=None,
    hv_file=None,
    override_params=None,
    load_path=None,
    save_path=None,
    **kwargs
):

    override_params = override_params or {}
    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
        num_cpu = MPI.COMM_WORLD.Get_size()
    
    seed_path = None if save_path == None else os.path.join(save_path,'seeds')
    # Seed everything.
    print('[learn@der] -> Creating rank_seed...')
    rank_seed = seed + 1000000 * rank if seed is not None else None
    set_global_seeds(rank_seed, seed_path)
    print('[learn@der] -> Done creating rank_seed...')

    # Prepare params.
    params = config.DEFAULT_PARAMS
    env_name = env.specs[0].id
    params['seed'] = rank_seed
    params['env_name'] = env_name
    params['replay_strategy'] = replay_strategy
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
         json.dump(params, f)
    print('[learn@der] -> Preparing params...')
    params = config.prepare_params(params)
    params['rollout_batch_size'] = env.num_envs
    
    params.update(kwargs)

    if demo_file is not None:
        params['bc_loss'] = 1
        
    if hv_file is None and params['hv_buffer_enabled']:
        assert hv_file is not None, 'HV Buffer has been enabled but no hv_file has been supplied.'
    

    print('[learn@der] -> Configuring log_params...')
    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    dims = config.configure_dims(params)
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)
    if load_path is not None:
        tf_util.load_variables(load_path)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    eval_env = eval_env or env

    print('[learn@der] -> Creating RolloutWorker...')
    rollout_worker = RolloutWorker(env, policy, dims, logger, monitor=True, **rollout_params)
    print('[learn@der] -> Creating Evaluator...')
    evaluator = RolloutWorker(eval_env, policy, dims, logger, **eval_params)

    n_cycles = params['n_cycles']
    n_epochs = total_timesteps // n_cycles // rollout_worker.T // rollout_worker.rollout_batch_size
    
    try:
        # To show seeds are right just before training starts
        st0 = np.random.get_state()
        print('Numpy seed:',st0[1][0])
        print('TF Graph seed:',policy.sess.graph.seed)
        print('TF Op seed:',policy.sess.graph._last_id)
    except:
        pass

    return train(
        save_path=save_path, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, demo_file=demo_file, hv_file = hv_file)


@click.command()
@click.option('--env', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--total_timesteps', type=int, default=int(5e5), help='the number of timesteps to run')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--demo_file', type=str, default = 'PATH/TO/DEMO/DATA/FILE.npz', help='demo data file path')
@click.option('--successful_only', type=str, default = 'no', help='hv file path')
@click.option('--hv_file', type=str, default = 'PATH/TO/HVFILE/DATA/FILE.pkl', help='hv file path')
@click.option('--evar', type=float, default =0.9, help='expected variance')
@click.option('--shape', type=str, default = 'circle', help='circle or regular')
@click.option('--stdevs', type=float, default =3.0, help='stdev to remove outliers')
def main(**kwargs):
    learn(**kwargs)


if __name__ == '__main__':
    main()
