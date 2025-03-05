# Code adapted from https://github.com/araffin/rl-baselines-zoo
# Author: Antonin Raffin
# Edited by: Javier Moralejo
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import argparse
import os
import time

import gym
import numpy as np
import random

from config import ENV_ID
from utils.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams

import glob

DEFAULT = ["Default"]
CLEAR = ["ClearNoon", "ClearSunset"]
CLOUD = ["CloudyNoon", "CloudySunset"]
WET = ["WetNoon", "WetSunset"]
CLOUDWET = ["WetCloudyNoon", "WetCloudySunset"]
STORM = [
    "HardRainNoon",
    "HardRainSunset",
    "MidRainSunset",
    "MidRainyNoon",
    "SoftRainNoon",
    "SoftRainSunset",
]

weather = [
    "ClearNoon",
    "ClearSunset",
    "CloudyNoon",
    "CloudySunset",
    "WetNoon",
    "WetSunset",
    "SoftRainNoon",
    "MidRainyNoon",
]

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs")
parser.add_argument(
    "--algo",
    help="RL Algorithm",
    default="sac",
    type=str,
    required=False,
    choices=list(ALGOS.keys()),
)
parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=8, type=int)
parser.add_argument(
    "--exp-id", help="Experiment ID (-1: no exp folder, 0: latest)", default=0, type=int
)
parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
parser.add_argument(
    "--no-render",
    action="store_true",
    default=False,
    help="Do not render the environment (useful for tests)",
)
parser.add_argument(
    "--deterministic", action="store_true", default=False, help="Use deterministic actions"
)
parser.add_argument(
    "--norm-reward",
    action="store_true",
    default=False,
    help="Normalize reward if applicable (trained with VecNormalize)",
)
parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
parser.add_argument("-vae", "--vae-path", help="Path to saved VAE", type=str, default="")
parser.add_argument("-p", "-path", help="Model path", type=str, default="")
# parser.add_argument('-best', '--best-model', action='store_true', default=False,
#                    help='Use best saved model of that experiment (if it exists)')
args = parser.parse_args()

algo = args.algo
folder = args.folder

"""
if args.exp_id == 0:
    args.exp_id = get_latest_run_id(os.path.join(folder, algo), ENV_ID)
    print('Loading latest experiment, id={}'.format(args.exp_id))

# Sanity checks
if args.exp_id > 0:
    log_path = os.path.join(folder, algo, '{}_{}'.format(ENV_ID, args.exp_id))
else:
    log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)

"""
log_path = "D:\path_to_folder\logs\sac\Experiment_Name"

print(log_path)
path_list = glob.glob(
    os.path.join(log_path, "DonkeyVae-v0-level-0_best_done_247.pkl")
)  #'{}_best_*.pkl'.format(ENV_ID)))
print(path_list)
# best_path = ''
# if args.best_model:
#    best_path = '_best'

# model_path = os.path.join(log_path, "{}_best_{}.pkl".format(ENV_ID, best_path))

for path in path_list:
    assert os.path.isfile(path), "No model found for {} on {}, path: {}".format(algo, ENV_ID, path)


stats_path = os.path.join(log_path, ENV_ID)
hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward)
if args.vae_path != "":
    hyperparams["vae_path"] = args.vae_path

log_dir = args.reward_log if args.reward_log != "" else None


if not os.path.isfile("{}\\test2.txt".format(log_path)):
    with open("{}\\test2.txt".format(log_path), "a+") as info:
        info.write("model,avg_reward,avg_length,passed_test\n")
        info.close()

env = create_test_env(
    stats_path=stats_path, seed=args.seed, log_dir=log_dir, hyperparams=hyperparams
)

# model = ALGOS[algo].load(model_path)
# env.envs[0].env.viewer.handler.seed = args.seed

# if args.n_timesteps == 1:
#    env.envs[0].env.viewer.handler.seed = args.seed

# Force deterministic for SAC and DDPG
deterministic = args.deterministic or algo in ["ddpg", "sac"]
if args.verbose >= 1:
    print("Deterministic actions: {}".format(deterministic))

running_reward = 0.0
ep_len = 0
episode = 0
passed_test = 0
total_reward = 0
total_length = 0
done = False
for path in path_list:
    print(path)
    model = ALGOS[algo].load(path)
    seed = 0

    while episode < args.n_timesteps:
        random.seed(seed + episode)
        np.random.seed(seed + episode)
        env.envs[0].env.viewer.world.next_weather(next_weather=weather[episode % len(weather)])
        obs = env.reset()
        for _ in range(3001):
            if not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                # Clip Action to avoid out of bound errors
                if isinstance(env.action_space, gym.spaces.Box):
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, done, infos = env.step(action)
                if not args.no_render:
                    env.render("human")
                running_reward += reward[0]
                ep_len += 1

                if done or ep_len >= 3000 or infos[0]["success"] == 1:
                    # NOTE: for env using VecNormalize, the mean reward
                    # is a normalized reward when `--norm_reward` flag is passed
                    print("Episode Reward: {:.2f}".format(running_reward))
                    print("Episode Length", ep_len)
                    total_reward += running_reward
                    total_length += ep_len
                    if ep_len >= 3000 or infos[0]["success"] == 1:
                        passed_test += 1
                    running_reward = 0.0
                    ep_len = 0
                    episode += 1
                    done = True
        done = False
    with open("{}\\test2.txt".format(log_path), "a") as info:
        info.write(
            "{},{},{},{}\n".format(
                path, total_reward / args.n_timesteps, total_length / args.n_timesteps, passed_test
            )
        )
        info.close()
    if passed_test < -1:
        os.rename(path, "{}_x".format(path))
    # env.envs[0].env.viewer.handler.seed = args.seed
    # obs = env.reset()
    episode = 0
    passed_test = 0
    total_reward = 0
    total_length = 0

    model = None

# env.reset()
env.envs[0].env.exit_scene()
# Close connection does work properly for now
# env.envs[0].env.close_connection()
time.sleep(0.5)
