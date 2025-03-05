import pandas as pd
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description="Moving average.")
# parser.add_argument("-f", "--file", required=True)

# file_name = vars(parser.parse_args())["file"]

window = 25
title = "Sin IMU o Últimas Acciones"

file_names = ["sem_seg", "nospeedb", "commands0", "commands0_nospeed"]
names = [
    "Con IMU / Últimas Acciones",
    "Sin IMU",
    "Sin Últimas Acciones",
    "Sin IMU / Últimas Acciones",
]
rewards = []
ticks = []
ma_rewards = []
ma_ticks = []
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

with open(title + ".txt", "w") as f:
    f.close()

for file_name, name, color in zip(file_names, names, colors):
    df = pd.read_csv(file_name + ".txt")

    reward = df["acum_reward"]

    ticks = df["ticks"]

    ma_reward = df["acum_reward"].rolling(window).mean()

    ma_ticks = df["ticks"].rolling(window).mean()

    plt.figure(1)
    plt.plot(ma_reward, label=name, color=color)
    plt.plot(reward, color=color, alpha=0.2)

    plt.figure(2)
    plt.plot(ma_ticks, label=name, color=color)
    plt.plot(ticks, color=color, alpha=0.2)

    with open(title + ".txt", "a+") as f:
        f.write(
            "{},{},{},{}\n".format(
                df["acum_reward"].mean(),
                df["acum_reward"].std(),
                df["ticks"].mean(),
                df["ticks"].std(),
            )
        )
        f.close()

plt.figure(1)
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.title(title + " - Recompensa")
plt.legend()
plt.savefig(title + "_ma_reward.png")
# plt.clf()


plt.figure(2)
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.title(title + " - Recompensa")
plt.legend()
plt.savefig(title + "_ma_tick.png")
# plt.clf()
# plt.savefig("ma_reward_"+file_name+".png")
# plt.clf()

# plt.plot(ma_ticks)
# plt.savefig("ma_ticks_"+file_name+".png")
