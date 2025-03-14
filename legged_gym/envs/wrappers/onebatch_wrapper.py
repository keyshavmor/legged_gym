import isaacgym
assert isaacgym
import torch
import gym

class OneBatchWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        self.obs_history_length = self.env.cfg.env.num_observation_history

        self.num_obs_history = self.obs_history_length * self.env.cfg.env.num_observations # 30 * 70 = 2100
        self.obs_history = torch.zeros(1, self.num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False) # [num_env, 2100]
        self.num_privileged_obs = self.env.cfg.env.num_privileged_obs

    def step(self, action):
        assert action.dim()==1, 'Batch size is not equal to 1'
        obs, privileged_obs, rew, done, info = self.env.step(action.unsqueeze(0))
        assert obs.shape[0]==1, 'Batch size is not equal to 1'

        self.obs_history = torch.cat((self.obs_history[:, self.env.cfg.env.num_observations:], obs), dim=-1)
        obs = obs.squeeze(0)
        privileged_obs = privileged_obs.squeeze(0)
        obs_history = self.obs_history.squeeze(0)

        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': obs_history}, rew, done, info

    def get_observations(self):
        obs = self.env.get_observations() # [num_env, 70]
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history = torch.cat((self.obs_history[:, self.env.cfg.env.num_observations:], obs), dim=-1)
        obs = obs.squeeze(0)
        privileged_obs = privileged_obs.squeeze(0)
        obs_history = self.obs_history.squeeze(0)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': obs_history}

    def reset_idx(self, env_ids):  # it might be a problem that this isn't getting called!!
        ret = super().reset_idx(env_ids)
        self.obs_history[env_ids, :] = 0
        return ret

    def reset(self):
        obs, privileged_obs = super().reset()
        
        self.obs_history[:, :] = 0
        obs = obs.squeeze(0)
        privileged_obs = privileged_obs.squeeze(0)
        obs_history = self.obs_history.squeeze(0)
        return obs, privileged_obs, obs_history


# if __name__ == "__main__":
#     from tqdm import trange
#     import matplotlib.pyplot as plt

#     import ml_logger as logger

#     from go1_gym_learn.ppo import Runner
#     from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
#     from go1_gym_learn.ppo.actor_critic import AC_Args

#     from go1_gym.envs.base.legged_robot_config import Cfg
#     from go1_gym.envs.mini_cheetah.mini_cheetah_config import config_mini_cheetah
#     config_mini_cheetah(Cfg)

#     test_env = gym.make("VelocityTrackingEasyEnv-v0", cfg=Cfg)
#     env = HistoryWrapper(test_env)

#     env.reset()
#     action = torch.zeros(test_env.num_envs, 12)
#     for i in trange(3):
#         obs, rew, done, info = env.step(action)
#         print(obs.keys())
#         print(f"obs: {obs['obs']}")
#         print(f"privileged obs: {obs['privileged_obs']}")
#         print(f"obs_history: {obs['obs_history']}")

#         img = env.render('rgb_array')
#         plt.imshow(img)
#         plt.show()
