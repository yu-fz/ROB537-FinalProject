from gym.envs.registration import register

register(
    id='Explore2D-Easy-v0',
    entry_point='gym_Explore2D.envs:Explore2D_Env_Easy',
)


register(
    id='Explore2D-Medium-v0',
    entry_point='gym_Explore2D.envs:Explore2D_Env_Medium',
)


register(
    id='Explore2D-Hard-v0',
    entry_point='gym_Explore2D.envs:Explore2D_Env_Hard',
)
