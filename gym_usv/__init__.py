from gym.envs.registration import register

register(
    id='usv-asmc-v0',
    entry_point='gym_usv.envs:UsvAsmcEnv'
)

register(
    id='usv-pid-v0',
    entry_point='gym_usv.envs:UsvPidEnv'
)

register(
    id='usv-asmc-ye-int-v0',
    entry_point='gym_usv.envs:UsvAsmcYeIntEnv'
)
