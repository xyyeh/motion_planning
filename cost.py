from util import *
import time
import torch
import numpy as np


class Cost(object):
    """
    Cost class that computes obstacle, grasp, and smoothness and their gradients.
    """

    def __init__(self, env):
        self.env = env
        self.cfg = env.config

    def functional_grad(self, v, a, JT, ws_cost, ws_grad):
        """
        Compute functional gradient based on workspace cost.
        """
        p = v.shape[-2]
        vel_norm = np.linalg.norm(v, axis=-1, keepdims=True)  # n x p x 1
        cost = np.sum(ws_cost * vel_norm[..., 0], axis=-1)  # n
        normalized_vel = safe_div(v, vel_norm)  # p x 3
        proj_mat = np.eye(3) - np.matmul(
            normalized_vel[..., None], normalized_vel[..., None, :]
        )  # p x 3 x 3
        scaled_curvature = ws_cost[..., None, None] * safe_div(
            np.matmul(proj_mat, a[..., None]), vel_norm[..., None] ** 2
        )  # p x 3 x 1
        projected_grad = np.matmul(proj_mat, ws_grad[..., None])  # p x 3 x 1
        grad = np.sum(
            np.matmul(JT, (vel_norm[..., None] * projected_grad - scaled_curvature)),
            axis=-1,
        )
        return cost, grad

    def compute_smooth_loss(self, xi, start, end):
        """
        Computes smoothness loss using (1), (2) of DOI 10.1109/robot.2009.5152817
        @param xi       The stack of configurations \in R^{nxd} for q_1 to q_n
        @param start    Start of trajectory, i.e. q_0
        @param end      End of trajectory, i.e. q_n+1
        """
        link_smooth_weight = np.array(self.cfg.link_smooth_weight)[None]
        ed = np.zeros(
            [xi.shape[0] + 1, xi.shape[1]]
        )  # one additional element for difference computation, \mathbb{R}^{n_time_steps+1 x 1}

        # first element: ed \in \mathbb{R}^{n_time_steps+1 x 1}
        ed[0] = (
            self.cfg.diff_rule[0][self.cfg.diff_rule_length // 2 - 1]  # -1
            * start
            / self.cfg.time_interval
        )

        # last element: if we are not using goal set, use the last trajectory point
        if not self.cfg.goal_set_proj:
            ed[-1] = (
                self.cfg.diff_rule[0][self.cfg.diff_rule_length // 2]  # 1
                * end
                / self.cfg.time_interval
            )

        # remaining velocities, mathbb{R}^{n_time_steps+1 x n_time_steps} x mathbb{R}^{n_time_steps}
        # velocity \in mathbb{R}^{n_time_steps+1}
        velocity = self.cfg.diff_matrices[0].dot(xi)

        # equation (1) with D = 1, \frac{1}{2} w_1 (\xi^T K_1^T + e_1^T) (K_1 \xi + e_1)
        # axis = 0 gets norm of each columns
        # axis = 1 gets norm of each rows
        # velocity_norm = (
        #     np.matmul(velocity.transpose(), velocity)
        #     + np.matmul(velocity.transpose(), ed)
        #     + np.matmul(ed.transpose(), velocity)
        #     + np.matmul(ed.transpose(), ed)
        # )
        # weighted_velocity_norm =

        weighted_velocity_norm = np.linalg.norm(
            (velocity + ed) * link_smooth_weight, axis=1
        )
        smoothness_loss = 0.5 * weighted_velocity_norm ** 2

        # equation (2): diff (1) wrt \xi, w_1 (K_1^T K_1 \xi + K_1^T e_1)
        smoothness_grad = self.cfg.A.dot(xi) + self.cfg.diff_matrices[0].T.dot(ed)
        smoothness_grad *= link_smooth_weight
        return smoothness_loss, smoothness_grad

    def forward_kinematics_obstacle(self, xi, start, end, arc_length=True):
        """
        Instead of computing C space trajectory and using Jacobian, we differentiate workspace positions for velocity and acceleration.
        """
        robot_pts = self.env.robot.collision_points  # n_dof x n_points x 3
        obstacle_pts = self.env.obstacle_points  # n_obs_points x 3

        n, m = xi.shape[0], xi.shape[1]  # n = n_waypoints, m = n_dof
        p = robot_pts.shape[1]  # n_points for a particular links

        robot_pts_pos = np.zeros([n, m, p, 3])
        robot_pts_vel = np.zeros_like(robot_pts_pos)
        robot_pts_acc = np.zeros_like(robot_pts_pos)

        # get the position, velocity and acceleration for each collision point using finite differencing
        for waypoints_idx in range(n):
            # update kinematics based on current set of way points
            self.env.robot.update_poses(xi[waypoints_idx, :])

            # update the obstacle's positions
            for bodies_idx in range(m):
                # get position from 1st link onwards
                pos = np.array(
                    self.env.robot.kine_dyn.mbc.bodyPosW[bodies_idx + 1].translation()
                )
                robot_pts_pos[waypoints_idx, bodies_idx, 0, :] = np.reshape(pos, (3,))

            # calculate potentials and gradients based on position of points
            (
                potentials,
                potential_grads,
                collide,
            ) = self.compute_obstacle_cost_and_gradient(robot_pts_pos, obstacle_pts)

        print(robot_pts_pos[0, 2, 0, :])

    def compute_obstacle_cost_and_gradient(
        self, robot_pts, obstacle_points, radius=0.1, epsilon=0.02
    ):
        """
        compute obstacle cost and gradient using (4) and Fig. 2 and Fig. 6 from 10.1177/0278364913488805
        """
        n_obstacle = obstacle_points.shape[0]
        n_dof = robot_pts.shape[0]
        n_collision_points = robot_pts.shape[1]

        # run through all obstacles and compute the potentials
        cost = np.zeros([n_dof, n_obstacle])
        for j in range(n_dof):
            for i in range(n_obstacle):
                # for k in range(n_collision_points):
                #     field_distance = np.norm(obstacle_points[i,:]-robot_pts[j,k,:]) - radius
                field_distance = np.linalg.norm(
                    obstacle_points[i, :] - robot_pts[j, 0, :]
                )  # single point
                D = field_distance - radius  # D(x)

                if D >= epsilon:
                    c = 0
                elif D >= 0:
                    c = 1 / 2 * (D - epsilon) ** 2 / epsilon
                else:
                    c = -D + 1 / 2 * epsilon

                cost[j, i] = c

        return 1, 2, 3

    def compute_collision_loss(self, xi, start, end):
        """
        Compute obstacle loss
        """
        n, m = xi.shape[0], xi.shape[1]
        obs_grad = np.zeros_like(xi)
        obs_cost = np.zeros([n, m + 1])

        self.forward_kinematics_obstacle(xi, start, end)

        print("compute collision loss .............................")

        return obs_cost, obs_grad

    def compute_total_loss(self, trajectory):
        """
        Compute total losses, gradients, and other info
        """

        # smoothness loss and gradient
        smoothness_loss, smoothness_grad = self.compute_smooth_loss(
            trajectory.data, trajectory.start, trajectory.end
        )
        smoothness_loss_sum = smoothness_loss.sum()

        # obstacle loss and gradient
        (obstacle_loss, obstacle_grad) = self.compute_collision_loss(
            trajectory.data, trajectory.start, trajectory.end
        )

        # total cost and vectorial summation of gradient
        # cost = weighted_obs + weighted_smooth
        # grad = weighted_obs_grad + weighted_smooth_grad

        # weights
        weighted_smooth = self.cfg.smoothness_weight * smoothness_loss_sum
        weighted_smooth_grad = self.cfg.smoothness_weight * smoothness_grad

        # total costs
        cost = weighted_smooth
        grad = weighted_smooth_grad

        cost_trajectory = self.cfg.smoothness_weight * smoothness_loss[:-1]

        if self.cfg.goal_set_proj:
            # goal_distance = (
            # np.linalg.norm(
            #     trajectory.data[-1] - trajectory.goal_set[trajectory.goal_idx]
            # )
            print("Not implemented - compute_total_loss")
            goal_distance = 0
        else:
            goal_distance = 0

        # goal_distance = (
        #     np.linalg.norm(
        #         trajectory.data[-1] - trajectory.goal_set[trajectory.goal_idx]
        #     )
        #     if self.cfg.goal_set_proj
        #     else 0
        # )

        # print(goal_distance)
        terminate = (
            self.cfg.pre_terminate
            and (goal_distance < 0.01)
            and smoothness_loss_sum < self.cfg.terminate_smooth_loss
        )

        info = {
            "smooth": smoothness_loss_sum,
            "gradient": grad,
            "cost": cost,
            "grad": np.linalg.norm(grad),
            "terminate": terminate,
            "goal_distance": goal_distance,
            "cost_trajectory": cost_trajectory,
        }

        return cost, grad, info