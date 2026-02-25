import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, BaseCallback
from datetime import datetime
from navigation_env import NavWorldEnv


def make_env():
    return Monitor(NavWorldEnv(size=10))


if __name__ == "__main__":

    save_path = "../warehouse/nn/nav/" + str(datetime.now())
    env = make_vec_env(make_env, n_envs=4)

    eval_env = make_env()
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_path,
                                     log_path=save_path, eval_freq = 500,
                                     deterministic=True, render=False,)
    callback = CallbackList([eval_callback])

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        n_steps=256,
        batch_size=512,
        learning_rate=3e-4,
        gamma=0.95,
        ent_coef=0.02,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="../warehouse/nn/nav/logs"
    )

    model.learn(total_timesteps=2_000_000,
                progress_bar=True,
                callback=callback,
                log_interval=8)
    model.save(save_path + "/nav_policy_end")
