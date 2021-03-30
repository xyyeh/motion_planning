# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from optimizer import Optimizer
from cost import Cost
from util import *

import config
import time
import torch
import multiprocessing

class Planner(object):
    """
    Planner class that plans the trajectory, grasp planning, and standoff pregrasp
    """

    def __init__(self, env, trajectory):
        self.cfg = config.cfg
        self.trajectory = trajectory

        self.cost = Cost(env)
        self.optim = Optimizer(env, self.cost)

        if self.cfg.goal_set_proj:
            print("Not implemented")
            # if self.cfg.scene_file == "" or self.cfg.traj_init == "grasp":
            #     self.load_grasp_set(env)
            #     self.setup_goal_set(env)
            # else:
            #     self.load_goal_from_scene()

            # self.grasp_init(env)
            # self.learner = Learner(env, trajectory, self.cost)
        else:
            self.trajectory.interpolate_waypoints()
        self.history_trajectories = []
        self.info = []
        self.ik_cache = []

    def update(self, env, trajectory):
        """
        Updates planner according to environment
        """
        self.cfg = config.cfg
        self.trajectory = trajectory

        # update cost
        # self.cost.env = env
        # self.cost.cfg = config.cfg
        # if len(self.env.objects) > 0:
        #     self.cost.target_obj = self.env.objects[self.env.target_idx]

        # update optimizer
        self.optim = Optimizer(env, self.cost)

        # load grasps if needed
        if self.cfg.goal_set_proj:
            print("Not implemented")
            # if self.cfg.scene_file == "" or self.cfg.traj_init == "grasp":
            #     self.load_grasp_set(env)
            #     self.setup_goal_set(env)
            # else:
            #     self.load_goal_from_scene()

            # self.grasp_init(env)
            # self.learner = Learner(env, trajectory, self.cost)
        else:
            self.trajectory.interpolate_waypoints()
        self.history_trajectories = []
        self.info = []
        self.ik_cache = []

    def plan(self, trajectory):
        """
        Run chomp optimizer to do trajectory optimization
        """
        self.history_trajectories = [np.copy(trajectory.data)]
        self.info = []
        self.selected_goals = []
        start_time_ = time.time()
        alg_switch = self.cfg.ol_alg != "Baseline" and self.cfg.ol_alg != "Proj"

        if (not self.cfg.goal_set_proj) or len(self.trajectory.goal_set) > 0:
            for t in range(self.cfg.optim_steps + self.cfg.extra_smooth_steps):
                start_time = time.time()
                if (
                    self.cfg.goal_set_proj
                    and alg_switch and t < self.cfg.optim_steps 
                ):
                    self.learner.update_goal()
                    self.selected_goals.append(self.trajectory.goal_idx)

                self.info.append(self.optim.optimize(trajectory, force_update=True))  
                self.history_trajectories.append(np.copy(trajectory.data))

                if self.cfg.report_time:
                    print("plan optimize:", time.time() - start_time)

                if self.info[-1]["terminate"] and t > 0:
                    break
 
            # compute information for the final
            if not self.info[-1]["terminate"]:
                self.info.append(self.optim.optimize(trajectory, info_only=True))  
            else:
                del self.history_trajectories[-1]

            plan_time = time.time() - start_time_
            res = (
                "SUCCESS BE GENTLE"
                if self.info[-1]["terminate"]
                else "FAIL DONT EXECUTE"
            )
            if not self.cfg.silent:
                print(
                "planning time: {:.3f} PLAN {} Length: {}".format(
                    plan_time, res, len(self.history_trajectories[-1])
                )
            )
            self.info[-1]["time"] = plan_time

        else:
            if not self.cfg.silent: print("planning not run...")
        return self.info