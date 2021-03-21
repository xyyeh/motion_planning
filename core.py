from util import *

import time
import torch

torch.no_grad()

class Trajectory(object):
    """
    Trajectory class
    """
        
    def __init__(self, timesteps=100, dof=7):
        """
        Initialize fixed endpoint trajectory.
        """
        self.timesteps = config.cfg.timesteps
        self.dof = dof
        self.goal_set = []
        self.goal_quality = []

        # organize joint space trajectories as q_init^T, [q1^T; q2^T; ...] , q_goal^T
        self.start = np.array([0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785])
        self.data = np.zeros([self.timesteps, dof])
        self.end = np.array([-0.99, -1.74, -0.61, -3.04, 0.88, 1.21, -1.12])

        self.interpolate_waypoints(mode=config.cfg.traj_interpolate)