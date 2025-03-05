import matplotlib.pyplot as plt

test_name = "VAE o Autoencoder Tradicional"

paths = ["aaaa", "NoVAE"]
names = ["VAE", "Autoencoder Tradicional"]
train_r_loss = []
train_g_loss = []
val_r_loss = []
val_g_loss = []
color = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
for path in paths:
    with open(path + "_train.txt", "r") as f:
        lines = f.readlines()
        f.close()

        train_r_loss.append([float(i) for i in lines[1].split(",")])
        train_g_loss.append([float(i) for i in lines[4].split(",")])

        if path == "green2" or path == "green2b":
            for idx, elem in enumerate(train_r_loss[1]):
                train_r_loss[1][idx] = train_r_loss[1][idx] - train_g_loss[1][idx]
        if path == "green3":
            for idx, elem in enumerate(train_r_loss[2]):
                train_r_loss[2][idx] = train_r_loss[2][idx] - train_g_loss[2][idx] * 2

    with open(path + "_val.txt", "r") as f:
        lines = f.readlines()
        f.close()

        val_r_loss.append([float(i) for i in lines[1].split(",")])
        val_g_loss.append([float(i) for i in lines[4].split(",")])


# print("REC")
for p, n, t, v, c in zip(paths, names, train_r_loss, val_r_loss, color):
    plt.plot(t, label=n + " T.", color=c)
    plt.plot(v, label=n + " V.", color=c, linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(test_name + " - Error de Reconstrucción")
    plt.ylim(0, 8000)
    plt.legend()
    print(n + " " + str(min(v)))
plt.savefig(test_name + "2_rec")
plt.clf()
plt.close()

# print("GREEN")
for p, n, t, v, c in zip(paths, names, train_g_loss, val_g_loss, color):
    plt.plot(t, label=n + " T.", color=c)
    plt.plot(v, label=n + " V.", color=c, linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(test_name + " - Error en las Líneas")
    plt.ylim(200, 500)
    plt.legend()
    print(n + " " + str(min(v)))
plt.savefig(test_name + "2_green")
plt.clf()
plt.close()
