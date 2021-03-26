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
    Robot kinematics and dynamics class
    """
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.kine_dyn = from_urdf_file(self.urdf_path)
        self.id = -1
        print(
            "Imported from "
            + self.urdf_path
            + ", robot with name "
            + self.kine_dyn.name.decode("utf-8")
        )
        # joints
        self.dof = self.kine_dyn.mb.nrDof()

        # set gravity direction (this is the acceleration at base joint for RNEA)
        self.kine_dyn.mbc.gravity = e.Vector3d(0, 0, 9.81)
        self.kine_dyn.mbc.zero(self.kine_dyn.mb)

        # robot limits
        self.lower_limit = e.VectorXd.Zero(self.dof)
        self.upper_limit = e.VectorXd.Zero(self.dof)
        for i, (k, v) in enumerate(self.kine_dyn.limits.lower.items()):
            self.lower_limit[i] = v
        for i, (k, v) in enumerate(self.kine_dyn.limits.upper.items()):
            self.upper_limit[i] = v

        # # setup spheres for collision detection
        # self.extents = np.loadtxt(config.cfg.robot_model_path + "/extents.txt")
        # self.sphere_size = (
        #     np.linalg.norm(self.extents, axis=-1).reshape([self.extents.shape[0], 1])
        #     / 2.0
        # )
        
    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def update_kinematics(self, q, dq):
        """
        Update kinematics using values from physics engine
        @param q A list of joint angles
        @param dq A list of joint velocities
        """
        self.kine_dyn.mbc.q = [
            [],
            [q[0]],
            [q[1]],
            [q[2]],
            [q[3]],
            [q[4]],
            [q[5]],
            [q[6]],
        ]
        self.kine_dyn.mbc.alpha = [
            [],
            [dq[0]],
            [dq[1]],
            [dq[2]],
            [dq[3]],
            [dq[4]],
            [dq[5]],
            [dq[6]],
        ]
        # forward kinematics
        rbd.forwardKinematics(p.mb, p.mbc)
        rbd.forwardVelocity(p.mb, p.mbc)

    def update_dynamics(self):
        """
        Update dynamics using values from physics engine to compute M, Minv and h
        @return M, Minv and h
        """
        # mass matrix
        fd = rbd.ForwardDynamics(p.mb)
        fd.computeH(p.mb, p.mbc)
        self.M = fd.H()
        self.Minv = self.M.inverse()
        # nonlinear effects vector
        fd.computeC(p.mb, p.mbc)
        self.h = fd.C()
        
        return M, Minv, h

    def _body_id_from_name(name, bodies):
        """
        Gets the body Id from the body name
        @param name The name of the body
        @param bodies The set of bodies provided by the multibody data structure
        @return Id of the body, -1 if not found
        """
        for bi, b in enumerate(bodies):
            if b.name().decode("utf-8") == name:
                return bi
                
        return -1

    def _sva_to_affine(sTransform):
        """
        Converts a spatial transform matrix to a homogeneous transform matrix
        @param sTransform Spatial transform
        @return Homogeneous transform matrix
        """
        m4d = e.Matrix4d.Identity()
        R = sTransform.rotation().transpose()
        p = sTransform.translation()
        for row in range(3):
            for col in range(3):
                m4d.coeff(row, col, R.coeff(row, col))
        for row in range(3):
            m4d.coeff(row, 3, p[row])

        return m4d

class Env(object):
    """
    Environment class
    """
    def __init__(self):
        pass

class PlanningScene(object):
    """
    Planning scene class
    """
    def __init__(self, cfg):
        self.traj = Trajectory(config.cfg.timesteps)
        print("Setting up env...")
        start_time = time.time()
        
        if len(config.cfg.scene_file) > 0:
            self.planner = Planner(self.env, self.traj, lazy=config.cfg.default_lazy)
            if config.cfg.vis:
                self.setup_renderer()