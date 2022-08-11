import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# # print ', '.join(["'{}'".format(name.split('.')[0]) for name in atari_py.list_games()])
for game in ['pong']:
    for obs_type in ['image', 'ram']:
        # space_invaders should yield SpaceInvaders-v0 and SpaceInvaders-ram-v0
        name = ''.join([g.capitalize() for g in game.split('_')])
        if obs_type == 'ram':
            name = '{}-ram'.format(name)

        nondeterministic = False
        if game == 'space_invaders':
            frameskip = 3
        else:
            frameskip = 4

        # Use a deterministic frame skip.
        register(
            id='{}NoFrameskip-v5'.format(name),
            entry_point='ENV.env.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1}, # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic,
        )

#classic

# mujoco

register(
    id='HalfCheetah-cost',
    entry_point='ENV.env.mujoco:HalfCheetahEnv_cost',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)


register(
    id='Pointcircle-cost',
    entry_point='ENV.env.mujoco:PointEnv',
    max_episode_steps=65,
)

register(
    id='Ant-cost',
    entry_point='ENV.env.mujoco:AntEnv_cost',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
register(
    id='Humanoid-cost',
    entry_point='ENV.env.mujoco:HumanoidEnv_cost',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)


register(
    id='Spacereach-cost',
    entry_point='ENV.env.mujoco:SpaceReachEnv_cost',
    max_episode_steps=512,
)


register(
    id='Spacerandom-cost',
    entry_point='ENV.env.mujoco:SpaceRandomEnv_cost',
    max_episode_steps=512,
)

register(
    id='Spacedualarm-cost',
    entry_point='ENV.env.mujoco:SpaceRobotDualArm_cost',
    max_episode_steps=512,
)