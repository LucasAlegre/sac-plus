import gym
import random
import numpy as np
import torch as th
from sac.sac_continuos_action import SAC
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)


def run():

    env = gym.make('Hopper-v2')
    env.seed(SEED)

    agent = SAC(env,
                gradient_updates=20,
                num_q_nets=2,
                m_sample=None,
                buffer_size=int(4e5),
                mbpo=False,
                experiment_name=f'sac-hopper-{SEED}',
                log=True,
                wandb=True)

    agent.learn(total_timesteps=175000)
    agent.save()

if __name__ == '__main__':
    run()
