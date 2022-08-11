import os

import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding

from gym.envs.robotics import utils
from gym.envs.robotics import rotations

import mujoco_py

PATH = os.path.dirname(__file__) # good way to get the target dirname base the file  
MODEL_XML_PATH = os.path.join(PATH,'mujoco_files','spacerobot','spacerobot_v3.xml')
DEFAULT_SIZE = 500


class RobotEnv(gym.Env):
    # n_actions is the number of actuator
    # n_substeps ?
    def __init__(self, model_path, initial_qpos, n_substeps):

        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()

        self._env_setup(initial_qpos=initial_qpos)  # set Robot initial joints
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.goal = self._sample_goal()  # sample goals
        obs = self._get_obs()

        self._set_action_space()  # set aciton space
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy()[:6]
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def _detecte_collision(self):
        self.collision = self.sim.data.ncon
        return self.collision

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):  
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)  
        self.sim.step()  
        self._step_callback() 
        obs = self._get_obs()

        success_info = self._is_success(obs[12:15], self.goal[:3])
        if success_info == 0.:
            done = False
        else:
            done = True
        info = {
            'is_success': success_info,
        }
        reward = self.compute_reward(obs[12:15], self.goal[:3], info)
        return obs, reward, done, info


    def reset(self):  
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues(e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        # super(RobotEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()  # _reset_sim()

        self.goal = self._sample_goal()  # every reset get another goal
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        # self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)  
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
    
    
    
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class SpacerobotEnv(RobotEnv):
    """Superclass for all SpaceRobot environments.
    """

    def __init__(
            self, model_path, n_substeps,
            distance_threshold, initial_qpos, reward_type, goal_type
    ):
        """Initializes a new Fetch environment.
        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            ? gripper_extra_height (float): additional height above the table when positioning the gripper
            ? block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            ? target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            ? target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            ? target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        #        self.gripper_extra_height = gripper_extra_height
        #        self.block_gripper = block_gripper
        #        self.has_object = has_object
        #        self.target_in_the_air = target_in_the_air
        #        self.target_offset = target_offset
        self.n_substeps = n_substeps
        #        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.goal_type = goal_type

        super(SpacerobotEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps,  # n_actions=4,
            initial_qpos=initial_qpos)

        # GoalEnv methods

    # ----------------------------

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, desired_goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        elif self.reward_type == 'cost':
            return d
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _set_action(self, act):
        # print('action',action)
        action = act
        if action.shape == (3,):
            action = np.concatenate((action, [0, 0, 0]))
        assert action.shape == (6,) 
        action = action.copy()  # ensure that we don't change the action outside of this scope
        # 测试值 0.05
        self.sim.data.ctrl[:6] = action * 0.1
        # self.sim.data.ctrl[:6] = np.zeros((6,),dtype=np.float)

        # if goal_distance (self.sim.data.get_body_xpos('tip_frame').copy(),self.goal[:3].copy()) < 0.02 \
        #         and goal_distance (rotations.quat2euler(self.sim.data.get_body_xquat('tip_frame').copy()),self.goal[3:].copy()) < 0.2:
        #     self.sim.data.ctrl[6:] = np.array([0.5,0.5,0.5,0],dtype=np.float)
        # else:
        #     self.sim.data.ctrl[6:] = np.array([0,0,0,0],dtype=np.float)
        # gripper
        if np.linalg.norm(action) < 0.1:
            # self.sim.data.ctrl[6:] = np.array([0.5,0.5,0.5,0],dtype=np.float)
            self.sim.data.ctrl[6:] = np.array([0, 0, 0, 0], dtype=np.float)
        else:
            self.sim.data.ctrl[6:] = np.array([0, 0, 0, 0], dtype=np.float)

        for _ in range(self.n_substeps):
            self.sim.step()

    def _get_obs(self):

        # positions
        grip_pos = self.sim.data.get_body_xpos('tip_frame').copy()
        grip_rot = self.sim.data.get_body_xquat('tip_frame').copy()
        grip_rot = gym.envs.robotics.rotations.quat2euler(grip_rot)

        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_body_xvelp('tip_frame') * dt
        robot_qpos, robot_qvel = gym.envs.robotics.utils.robot_get_obs(self.sim)
        # position = self.sim.data.qpos.flat.copy()
        # velocity = self.sim.data.qvel.flat.copy()

        object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-1:]
        gripper_vel = robot_qvel[-1:] * dt  # change to a scalar if the gripper is made symmetric

        achieved_goal1 = grip_pos.copy()
        achieved_goal2 = grip_rot.copy()

        qpos = self.sim.data.qpos[7:13].copy()
        qvel = self.sim.data.qvel[6:12].copy()

        obs1 = np.concatenate([
            qpos, qvel, grip_pos, self.goal[:3].copy()
        ])

        # # 观测量加入了goal
        # qpos = self.sim.data.qpos[10:13].copy()
        # qvel = self.sim.data.qvel[9:12].copy()
        # obs2 = np.concatenate([
        #     qpos, qvel, grip_rot, self.goal[3:6].copy()
        # ])

        obs = np.concatenate([qpos, qvel,grip_pos, self.goal[:3].copy()])

        return obs.copy()


    def _viewer_setup(self):
        #        body_id = self.sim.model.body_name2id('forearm_link')
        body_id = self.sim.model.body_name2id('wrist_3_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)  # self.initial_state在robot_env中定义;环境重置之后，机械臂的位置回到初始自然state
        self.sim.forward()
        return True

    def _sample_goal(self):
        
        if self.goal_type == 'multi':  # self.deterministic_reset in base_env.py
            target_pos = self.initial_gripper_xpos.copy()
            target_pos[0] += np.random.uniform(-0.3, -0.1)
            target_pos[1] += np.random.uniform(-0.2, 0.2)
            target_pos[2] += np.random.uniform(0, 0.3)

        elif self.goal_type == 'single':
            target_pos = self.initial_gripper_xpos.copy() + [-0.3, 0.2, 0.2]

        target_att = np.array([-1.42471829, -1.4207963, -3.11997619])

        # display target position
        site_id = self.sim.model.site_name2id('target0')  # set target position
        self.sim.model.site_pos[site_id] = target_pos
        self.sim.model.site_quat[site_id] = gym.envs.robotics.rotations.euler2quat(target_att)

        self.sim.forward()

        # goal_pos = self.sim.data.get_body_xpos('target0').copy()
        # goal_att = gym.envs.robotics.rotations.quat2euler(self.sim.data.get_body_xquat('target0').copy())
        # print('target_pos: {}'.format(target_pos))
        # print('target_att: {}'.format(target_att))

        goal = target_pos

        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)
        # return d

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():  
            self.sim.data.set_joint_qpos(name, value)
        gym.envs.robotics.utils.reset_mocap_welds(self.sim)

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_body_xpos('tip_frame').copy()
        grip_rot = self.sim.data.get_body_xquat('tip_frame').copy()
        grip_rot = gym.envs.robotics.rotations.quat2euler(grip_rot)
        self.initial_gripper_xrot = grip_rot

    def render(self, mode='human', width=500, height=500):
        return super(SpacerobotEnv, self).render(mode, width, height)    
    
    
    
class SpaceRandomEnv_cost(SpacerobotEnv, gym.utils.EzPickle):
    def __init__(self, reward_type='cost', goal_type='multi'):
        initial_qpos = {
            'arm:shoulder_pan_joint': 0,
            'arm:shoulder_lift_joint': 0,
            'arm:elbow_joint': 0.0,
            'arm:wrist_1_joint': 0.0,
            'arm:wrist_2_joint': 0.0,
            'arm:wrist_3_joint': 0.0
        }
        SpacerobotEnv.__init__(
            self, MODEL_XML_PATH, n_substeps=20, distance_threshold=0.01,
            initial_qpos=initial_qpos, reward_type=reward_type, goal_type=goal_type)
        gym.utils.EzPickle.__init__(self)