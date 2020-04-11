import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gym
from gym import spaces


class DynamicalSystem(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, system, dt=0.01, desiredTrajectory=None):
        super(DynamicalSystem, self).__init__()
        sysDim = system.numStateVars
        # Setup spaces
        self.action_space = spaces.Box(low=np.array([-1.0] * sysDim), high=np.array([1.0] * sysDim), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-np.inf] * sysDim), high=np.array([np.inf] * sysDim),
                                            dtype=np.float32)

        self.system = system
        self.state = system.initialize()
        self.dt = dt
        # Buffers
        self.desiredTrajectory = desiredTrajectory
        self.trajectory = np.expand_dims(self.state, axis=0)
        self.fig = plt.figure()

    # noinspection PyTypeChecker
    def step(self, action):
        deltaState = self.system.dynamics(self.state)
        self.state = self.state + deltaState * self.dt + action
        self.trajectory = np.concatenate((self.trajectory, np.expand_dims(self.state, axis=0)), axis=0)
        dist = np.sum((self.desiredTrajectory - self.state) ** 2, axis=1)
        reward = -np.min(dist)
        if reward < -1e6:
            done = True
        else:
            done = False
        return self.state, reward, done, {'episode': None}

    def reset(self):
        self.state = self.system.initialize()
        for t in range(300):
            deltaState = self.system.dynamics(self.state)
            self.state = self.state + deltaState * self.dt
        if self.desiredTrajectory is None:
            self.desiredTrajectory = np.expand_dims(self.state, axis=0)
            for t in range(300):
                deltaState = self.system.dynamics(self.state)
                self.state = self.state + deltaState * self.dt
                self.desiredTrajectory = np.concatenate((self.desiredTrajectory,
                                                         np.expand_dims(self.state, axis=0)), axis=0)
                self.trajectory = np.expand_dims(self.state, axis=0)
        else:
            self.trajectory = np.concatenate((self.trajectory, np.expand_dims(self.state, axis=0)), axis=0)
        return self.state

    def render(self, mode='human', close=False):
        ax = self.fig.gca(projection='3d')
        ax.plot(self.trajectory[:, 0], self.trajectory[:, 1], self.trajectory[:, 2], lw=0.5)
        ax.plot(self.desiredTrajectory[:, 0], self.desiredTrajectory[:, 1], self.desiredTrajectory[:, 2], lw=0.5)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title(self.system.name + " System")
        plt.show(block=False)
        plt.pause(.01)
        plt.clf()

