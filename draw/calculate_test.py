import pandas as pd
import argparse


parser = argparse.ArgumentParser(description="Moving average.")
# parser.add_argument("-f", "--file", required=True)

# file_name = vars(parser.parse_args())["file"]

window = 25
title = "Entrenamiento del Modelo Final"

file_name = "test2_tr_nosem"

df = pd.read_csv(file_name + ".txt")

reward = df["avg_reward"]
ticks = df["avg_length"]

reward_mean = reward.max()
ticks_mean = ticks.max()
n_success = df["passed_test"].max()

FPS = 15
time_per_tick = 1 / FPS

time_per_interruption = time_per_tick * ticks_mean * 8 / (8 - n_success)

autonomy = (8 * ticks_mean) / (8 * ticks_mean + 6 * FPS * (8 - n_success))

print(
    "{} {} {} {} {} {}".format(
        file_name, reward_mean, ticks_mean, n_success, time_per_interruption, autonomy
    )
)
