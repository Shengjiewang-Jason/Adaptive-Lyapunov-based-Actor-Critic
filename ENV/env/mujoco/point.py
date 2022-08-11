import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import math

class PointEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.max_reward = 1500.
        self.target_dist = 15
        mujoco_env.MujocoEnv.__init__(self, 'point.xml', 5)
        utils.EzPickle.__init__(self)
    def step(self, action):
        size=40
        # variables = [i for i in dir(self.sim.data)]
        # print(self.sim.get_state())
        # pos_before = np.copy(self.get_body_com("torso"))
        # self.do_simulation(action, self.frame_skip)
        # pos_after = np.copy(self.get_body_com("torso"))
        #action ballx and rot
        #qpos

        sim_state = self.sim.get_state()
        pos_before=np.copy(self.get_body_com("torso"))
        x_joint_i = self.sim.model.get_joint_qpos_addr("ballx")    #x位移
        y_joint_i = self.sim.model.get_joint_qpos_addr("bally")    #x位移
        z_joint_i = self.sim.model.get_joint_qpos_addr("rot")      #角度

        # 角度
        sim_state.qpos[z_joint_i] += action[1]
        ori = sim_state.qpos[z_joint_i]

        # compute increment in each direction
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]
        # ensure that the robot is within reasonable range
        sim_state.qpos[x_joint_i] =  np.clip(sim_state.qpos[x_joint_i] + dx, -size, size)
        sim_state.qpos[y_joint_i] =  np.clip(sim_state.qpos[y_joint_i] + dy, -size, size)
        # ensure that the robot is within reasonable range
        self.sim.set_state(sim_state)
        self.sim.forward()
        next_obs = self.get_current_obs()
        pos_after = np.copy(self.get_body_com("torso"))
        self.circle_mode=True

        if self.circle_mode:
            # vel = (pos_after-pos_before)/self.model.opt.timestep
            x, y = pos_after[0], pos_after[1]
            # dx, dy = vel[0], vel[1]
            # reward = -y * dx + x * dy
            # reward /= (1 + np.abs(np.sqrt(x ** 2 + y ** 2) - self.target_dist))
            reward = np.abs(np.sqrt(x ** 2 + y ** 2) - self.target_dist)

        next_obs = self.get_current_obs()
        done = False
        if abs(x) >3 :
            violation_of_constraint = 1
        else:
            violation_of_constraint = 0
        l_reward = max(abs(x)- 0.8*3., 0.)
        # if abs(x) >0.8*3:
        #     l_reward = 1.
        # else:
        #     l_reward = 0.
        return next_obs, reward, done, dict(l_rewards=l_reward,violation_of_constraint= violation_of_constraint)

    def get_current_obs(self):
        
        pos = np.copy(self.get_body_com("torso"))
        return np.concatenate([
            self.sim.data.qpos.flatten(),
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
            # [np.sqrt(pos[0] ** 2 + pos[1] ** 2) - self.target_dist],
        ])

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self.get_current_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
