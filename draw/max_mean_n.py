import pandas as pd
import argparse


parser = argparse.ArgumentParser(description="Moving average.")
# parser.add_argument("-f", "--file", required=True)

# file_name = vars(parser.parse_args())["file"]

window = 25
title = "Entrenamiento del Modelo Final"

file_name = "no_sem_final"

df = pd.read_csv(file_name + ".txt")

reward = df["acum_reward"]

reward_mean = reward.mean()
reward_max = reward.max()
n_success = df["success"].sum()
# ticks_max = ticks.max()

# time_per_interruption = 0.05 * ticks_max * 8 / (8-df["success"].max())

print("{} {} {} {}".format(file_name, reward_mean, reward_max, n_success))
