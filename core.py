from util import *
from config import *
from planner import Planner

import numpy as np
import eigen as e
import sva as s
import rbdyn as rbd
from rbdyn.parsers import *

import pybullet as b
import pybullet_data

import time
import torch

import matplotlib.pyplot as plt

torch.no_grad()


class Trajectory(object):
    """
    Trajectory class
    """

    def __init__(self, timesteps=100, dof=7):
        """
        Initialize fixed endpoint trajectory.
        """
        self.timesteps = cfg.timesteps
        self.dof = dof
        self.goal_set = []
        self.goal_quality = []

        # organize joint space trajectories as q_init^T, [q_1^T; q_2^T; ...; q_nTimestep^T] , q_goal^T
        # [2.96706  2.0944 2.96706  2.0944 2.96706  2.0944 3.05433]
        # [-2.96706  -2.0944 -2.96706  -2.0944 -2.96706  -2.0944 -3.05433]
        self.start = np.array([0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785])
        self.data = np.zeros([self.timesteps, dof])
        self.end = np.array([-0.99, 1.0, -0.61, 1.04, 0.88, 1.21, -1.12])

        # interpolate through the waypoints
        self.interpolate_waypoints(mode=cfg.trajectory_interpolate)

    def update(self, grad):
        """
        Update trajectory based on functional gradient.
        """
        self.data += grad

    def set(self, new_trajectory):
        """
        Set trajectory by given data.
        """
        self.data = new_trajectory

    def interpolate_waypoints(self, waypoints=None, mode="cubic"):
        """
        Interpolate the waypoints using interpolation with fixed/variable time steps
        """
        timesteps = cfg.timesteps

        # check if dynamic timesteps are desired, default to false
        if cfg.dynamic_timestep:
            timesteps = min(
                max(
                    int(np.linalg.norm(self.start - self.end) / cfg.trajectory_delta),
                    cfg.trajectory_min_step,
                ),
                cfg.trajectory_max_step,
            )
            cfg.timesteps = timesteps
            self.data = np.zeros([timesteps, self.dof])  # fixed start and end
            cfg.get_global_param(timesteps)
            self.timesteps = timesteps

        # interpolate through waypoints
        self.data = interpolate_waypoints(
            np.stack([self.start, self.end]), timesteps, self.start.shape[0], mode=mode
        )


class Simulation(object):
    def __init__(self, time_step, robot):
        # setup physics
        self.robot = robot
        self.time_step = time_step
        self.time = 0

        # client
        physics_client = b.connect(b.GUI)
        b.setAdditionalSearchPath(pybullet_data.getDataPath())
        b.setGravity(0, 0, -9.81)
        b.setRealTimeSimulation(0)
        b.setTimeStep(time_step)

        # import robot
        planeId = b.loadURDF("plane.urdf")
        startPos = [0, 0, 0]
        startOrientation = b.getQuaternionFromEuler([0, 0, 0])
        loadFlag = (
            b.URDF_USE_INERTIA_FROM_FILE | b.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        )
        robotId = b.loadURDF(
            self.robot.urdf_path, startPos, startOrientation, flags=loadFlag
        )
        self.robot.set_id(robotId)

        # unlock joints
        nDof = b.getNumJoints(self.robot.id)
        b.setJointMotorControlArray(
            robotId,
            range(nDof),
            b.VELOCITY_CONTROL,
            forces=[0] * nDof,
        )

    def get_time(self):
        return self.time

    def step_simulation(self):
        b.stepSimulation()
        self.time += self.time_step
        time.sleep(self.time_step)

    def step_simulation(self, q):
        self._set_robot_cfg(q)
        self.time += self.time_step
        time.sleep(self.time_step)

    def _set_robot_cfg(self, q):
        for i in range(b.getNumJoints(self.robot.id)):
            b.resetJointState(self.robot.get_id(), i, targetValue=q[i])

    def _update_simulation(self):
        joints_id = range(self.robot.dof)
        joint_states = b.getJointStates(self.robot.id, joints_id)
        # read state feedback
        q = [joint_states[i][0] for i in joints_id]
        dq = [joint_states[i][1] for i in joints_id]
        # update kinematics and dynamics properties
        self.robot.update_kinematics(q, dq)
        self.robot.update_dynamics()


class Environment(object):
    """
    Environment class storing robot and other objects.
    """

    def __init__(self, robot, cfg):
        self.robot = robot
        self.config = cfg


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

    def set_id(self, id):
        """
        Sets the robot id
        """
        self.id = id

    def get_id(self):
        """
        Gets the robot id
        """
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


class PlanningScene(object):
    """
    Planning scene containing the environment configuration, planner and trajectory.
    """

    def __init__(self, robot, cfg):
        self.trajectory = Trajectory(cfg.timesteps)
        self.env = Environment(robot, cfg)
        self.planner = Planner(self.env, self.trajectory)

    def update_planner(self):
        self.planner.update()

    def reset(self):
        """
        Resets scene for the next plan
        """
        self.planner = Planner(self.env, self.trajectory)

    def plan(self):
        """
        Run an optimization step
        """
        plan = self.planner.plan(self.trajectory)
        return plan


def plot_results(trajectory, upper_limit, lower_limit):
    """
    Plots the joint space trajectories
    """
    for i in range(robot.dof):
        plt.subplot(4, 3, i + 1)
        joint_trajectory = np.append(
            np.append(trajectory.start[i], trajectory.data[:, i]), trajectory.end[i]
        )
        counts = range(len(joint_trajectory))
        plt.plot(counts, joint_trajectory)
        plt.plot(
            counts, np.max(joint_trajectory) * np.ones(joint_trajectory.shape), "r"
        )
        plt.plot(
            counts, np.min(joint_trajectory) * np.ones(joint_trajectory.shape), "r"
        )
        print("Joint {}, min = {}, max = {}".format(i, np.min(joint_trajectory), np.max(joint_trajectory)))
    plt.show()


if __name__ == "__main__":
    import argparse

    step_time = 0.001
    total_time = 20

    robot = Robot("./assets/kuka_iiwa.urdf")
    sim = Simulation(step_time, robot)

    # configurations
    cfg.timesteps = 50
    cfg.get_global_param(cfg.timesteps)

    # get ik

    # planning phase
    planningScene = PlanningScene(robot, cfg)
    planningScene.plan()

    # load objects

    # results
    plot_results(planningScene.trajectory, robot.lower_limit, robot.upper_limit)

    # # run controller
    # # while sim.get_time() < total_time:
    # #     # sim.step_simulation()

    i = 0
    while i < cfg.timesteps:
        sim.step_simulation(planningScene.trajectory.data[i, :])
        i = i + 1
        time.sleep(0.05)
