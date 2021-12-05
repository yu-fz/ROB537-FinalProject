from gym.envs.registration import register

register(
    id = 'Explore2DTrial-v0',
    entry_point = 'gym_Explore2DTrial.envs:Explore2DTrial_Env',
)