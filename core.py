from util import *
import config

import numpy as np
import eigen as e
import sva as s
import rbdyn as rbd

from rbdyn.parsers import *

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

        self.interpolate_waypoints(mode=config.cfg.trajectory_interpolate)

    def update(self, grad):
        """
        Update trajectory based on functional gradient.
        """
        if config.cfg.consider_finger:  # TODO: set to false
            self.data += grad
        else:
            self.data[:, :-2] += grad[:, :-2]
        self.data[:, -2:] = np.minimum(np.maximum(self.data[:, -2:], 0), 0.04)

    def set(self, new_trajectory):
        """
        Set trajectory by given data.
        """
        self.data = new_trajectory

    def interpolate_waypoints(self, waypoints=None, mode="cubic"):
        """
        Interpolate the waypoints using interpolation.
        """
        timesteps = config.cfg.timesteps
        if config.cfg.dynamic_timestep:
            timesteps = min(
                max(
                    int(np.linalg.norm(self.start - self.end) / config.cfg.trajectory_delta),
                    config.cfg.trajectory_min_step,
                ),
                config.cfg.trajectory_max_step,
            )
            config.cfg.timesteps = timesteps
            self.data = np.zeros([timesteps, self.dof])  # fixed start and end
            config.cfg.get_global_param(timesteps)
            self.timesteps = timesteps
        self.data = interpolate_waypoints(
            np.stack([self.start, self.end]), timesteps, self.start.shape[0], mode=mode
        )

class Robot(object):
    """
    Robot class
    """
    def __init__(self, data_path):
        self.robot_kinematics = robot_kinematics("panda_arm", data_path=data_path)
        self.extents = np.loadtxt(config.cfg.robot_model_path + "/extents.txt")
        self.sphere_size = (
            np.linalg.norm(self.extents, axis=-1).reshape([self.extents.shape[0], 1])
            / 2.0
        )
        self.collision_points = self.load_collision_points()

        self.hand_col_points = self.collision_points[-3:].copy()
        self.joint_names = self.robot_kinematics._joint_name[:]
        del self.joint_names[-3]  # remove dummy hand finger joint
        self.joint_lower_limit = np.array(
            [[self.robot_kinematics._joint_limits[n][0] for n in self.joint_names]]
        )
        self.joint_upper_limit = np.array(
            [[self.robot_kinematics._joint_limits[n][1] for n in self.joint_names]]
        )
        self.joint_lower_limit[:, :-2] += config.cfg.soft_joint_limit_padding
        self.joint_upper_limit[:, :-2] -= config.cfg.soft_joint_limit_padding
        
class Env(object):
    """
    Environment class
    """

    def __init__(self, cfg):
        self.robot = Robot(config.cfg.root_dir)
        self.config = config.cfg
        self.objects = []
        self.names = []
        self.indexes = []
        self.sdf_torch = None
        self.sdf_limits = None
        self.target_idx = 0

class PlanningScene(object):
    """
    Environment class
    """

    def __init__(self, cfg):
        self.traj = Trajectory(config.cfg.timesteps)
        print("Setting up env...")
        start_time = time.time()
        self.env = Env(config.cfg)
        print("env init time: {:.3f}".format(time.time() - start_time))
        config.cfg.ROBOT = self.env.robot.robot_kinematics  # hack for parallel ik
        if len(config.cfg.scene_file) > 0:
            self.planner = Planner(self.env, self.traj, lazy=config.cfg.default_lazy)
            if config.cfg.vis:
                self.setup_renderer()