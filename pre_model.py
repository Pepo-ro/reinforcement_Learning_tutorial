import retro
import os
import time
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from baselines.common.retro_wrappers import *
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from policy import TransformerPolicy

env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
env = SonicDiscretizer(env)
env = RewardScaler(env, scale=0.01)
env = CustomRewardAndDoneEnv(env)
env = StochasticFrameSkip(env, n=4, stickprob=0.25)
env = Downsample(env, 2)
env = Rgb2gray(env)
env = FrameStack(env, 4)
env = ScaledFloatFrame(env)
env = TimeLimit(env, max_episode_steps=4500)
env = Monitor(env, log_dir, allow_early_resets=True)
print('行動空間: ', env.action_space)
print('状態空間: ', env.observation_space)

env.seed(1234)
set_global_seeds(1234)

env = DummyVecEnv([lambda: env])

model = PPO2(policy=CnnPolicy, env=env, verbose=0, learning_rate=2.5e-5, tensorboard_log=log_dir) #CnnPolicy
#model = PPO2(policy=TransformerPolicy, env=env, verbose=0, learning_rate=2.5e-5, tensorboard_log=log_dir) # TransformerPolicy
# model.learn(total_timesteps=200000, callback=callback) # 学習したいときは、こちら

model = PPO2.load('model', env=env, verbose=0) # 推論したいときは、こちら

state = env.reset()
env.close()

total_reward = 0

while True:
    # 環境の描画
    env.render()

    time.sleep(1/120)

    action, _ = model.predict(state)

    state, reward, done, info = env.step(action)
    total_reward += reward[0]

    if done:
        print('reward:', total_reward)
        state = env.reset()
        total_reward = 0
