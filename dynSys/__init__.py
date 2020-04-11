from gym.envs.registration import register

register(
    id='dynSys-v0',
    entry_point='dynSys.envs:DynamicalSystem'
)
