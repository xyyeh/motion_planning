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
        Instead of computing C space trajectory and using Jacobian, we differentiate workspace positoins for velocity and acceleration.
        """
        pass
        # robot_pts = robot.collision_points.

    def compute_collision_loss(self, xi, start, end):
        """
        Compute obstacle loss
        """
        n, m = xi.shape[0], xi.shape[1]
        obs_grad = np.zeros_like(xi)
        obs_cost = np.zeros([n, m + 1])
        # ()

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