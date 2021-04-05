import numpy as np
import eigen as e
import rbdyn as rbd
import sva as s
from rbdyn.parsers import *

import pybullet as b
import pybullet_data

import time


class Robot(object):
    """
    Robot class
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
        self.id = id

    def get_id(self):
        return self.id

    def update_kinematics(self, q, dq):
        """
        Update kinematics using values from physics engine
        @param q A list of joint angles
        @param dq A list of joint velocities
        """
        # self.kine_dyn.mbc.q = []
        # self.kine_dyn.mbc.alpha = []
        # self.kine_dyn.mbc.q.append([])
        # self.kine_dyn.mbc.alpha.append([])
        # for i in range(len(q)):
        #     self.kine_dyn.mbc.q.append([q[i]])
        #     self.kine_dyn.mbc.alpha.append([dq[i]])
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
        rbd.forwardKinematics(self.kine_dyn.mb, self.kine_dyn.mbc)
        rbd.forwardVelocity(self.kine_dyn.mb, self.kine_dyn.mbc)

    def update_dynamics(self):
        """
        Update dynamics using values from physics engine to compute M, Minv and h
        @return M, Minv and h
        """
        # mass matrix
        fd = rbd.ForwardDynamics(self.kine_dyn.mb)
        fd.computeH(self.kine_dyn.mb, self.kine_dyn.mbc)
        self.M = fd.H()
        self.Minv = self.M.inverse()
        # nonlinear effects vector
        fd.computeC(self.kine_dyn.mb, self.kine_dyn.mbc)
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


if __name__ == "__main__":
    import argparse

    step_time = 0.001
    total_time = 10

    robot = Robot("./assets/kuka_iiwa.urdf")
    sim = Simulation(step_time, robot)

    q = np.random.rand(7)
    dq = np.random.rand(7)

    robot.update_kinematics(q, dq)
    sv = robot.kine_dyn.mbc.bodyPosW[1]
    # print(sv.translation())
    # print(sv.rotation().transpose())

    test = np.array(sv.translation())
    test_ = np.reshape(test, (3,))
    print(test)
    print(test_)

    c = np.zeros([1,2,3,3])
    print(c[0,1,2,:])
    c[0,1,2,:] = test_
    print(c[0,1,2,:])

    for bi, bd in enumerate(robot.kine_dyn.mb.bodies()):
        print('body index: {}, body name = {}, body position = {}'.format(bi, bd.name().decode("utf-8"), robot.kine_dyn.mbc.bodyPosW[bi].translation().transpose()))

    # p = e.Vector3d.UnitX()
    # p_ = sv.rotation().transpose() * p + sv.translation()
    # print(p_)

    # example to add in a cube
    link1_id = b.loadURDF("./assets/cube.urdf")
    b.setCollisionFilterGroupMask(link1_id, -1, 0, 0)
    # b.resetBasePositionAndOrientation(position=)

    while sim.get_time() < total_time:
        # sim.step_simulation()
        sim.step_simulation(q)

    # s = Simulation(0.001, r)
