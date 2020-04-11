from dynSys.systems import LorenzSystem
import gym

system = LorenzSystem()
env = gym.make('dynSys:dynSys-v0', system=system, dt=0.01)

env.reset()
for t in range(1000):
    state, reward, done, _ = env.step([0.0] * 3)
    print(state, reward, done)
    env.render()
