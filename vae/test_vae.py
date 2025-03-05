"""
Test a trained vae
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from vae.controller import VAEController
from vae.data_loader import DataLoader
from vae.model import ConvVAE


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--folder", help="Log folder", type=str, default="C://Nueva carpeta/vae/val"
    )
    parser.add_argument(
        "-vae",
        "--vae-path",
        help="Path to saved VAE",
        type=str,
        default="D:/TFM/Carla/vae/green/green-2-big",
    )
    parser.add_argument("--n-samples", help="Max number of samples", type=int, default=10000)
    parser.add_argument("--n", help="Vae to test", type=int, default=50)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--sem-seg", help="Semantic Segmentation", type=int, default=1)
    parser.add_argument(
        "--kl-tolerance", help="KL tolerance (to cap KL loss)", type=float, default=0.5
    )  # 0.5
    parser.add_argument("--beta", help="Weight for kl loss", type=float, default=1.0)  # 1
    parser.add_argument("--batch-size", help="Batch size", type=int, default=64)
    parser.add_argument("-g", "--green", help="green mult", type=float, default=2.0)

    args = parser.parse_args()

    if not args.folder.endswith("/"):
        args.folder += "/"

    x_images_val = ["x/" + im for im in os.listdir(args.folder + "x/") if im.endswith(".png")]
    x_images_val = np.array(x_images_val)
    n_samples_val = len(x_images_val)

    if args.sem_seg == 1:
        y_images_val = ["y/" + im for im in os.listdir(args.folder + "y/") if im.endswith(".png")]
        y_images_val = np.array(y_images_val)

    if args.n_samples > 0:
        n_samples_val = min(n_samples_val, args.n_samples)
        x_images_val = x_images_val[0:n_samples_val]
        if args.sem_seg == 1:
            y_images_val = y_images_val[0:n_samples_val]

    print("{} images".format(n_samples_val))

    indices_val = np.arange(n_samples_val, dtype="int64")

    minibatchlist_val = [
        np.array(sorted(indices_val[start_idx : start_idx + args.batch_size]))
        for start_idx in range(0, len(indices_val) - args.batch_size + 1, args.batch_size)
    ]

    if args.sem_seg == 1:
        data_loader_val = DataLoader(
            minibatchlist_val,
            x_images_val,
            y_images_path=y_images_val,
            n_workers=-1,
            folder=args.folder,
        )
    else:
        data_loader_val = DataLoader(
            minibatchlist_val, x_images_val, n_workers=-1, folder=args.folder
        )

    vae_controller = VAEController(z_size=128)
    vae_controller.target_vae = ConvVAE(
        z_size=128,
        batch_size=args.batch_size,
        kl_tolerance=args.kl_tolerance,
        beta=args.beta,
        is_training=2,
        green=args.green,
        reuse=False,
    )

    val_error = []
    val_r_loss = []
    val_kl_loss = []
    val_r_loss_r = []
    val_r_loss_g = []
    val_r_loss_b = []
    for i in range(args.n):
        vae_path = args.vae_path + "-{:02d}.pkl".format(i + 49 + 1)  # 1
        vae_controller.load(vae_path, is_training=2)
        pbar = tqdm(total=len(minibatchlist_val))

        val_error_avg = []
        val_r_loss_avg = []
        val_kl_loss_avg = []
        val_r_loss_r_avg = []
        val_r_loss_g_avg = []
        val_r_loss_b_avg = []

        print("Epoch: ", str(i + 1))

        for obs, obj in data_loader_val:
            feed = {
                vae_controller.target_vae.input_tensor: obs,
                vae_controller.target_vae.objective_tensor: obj,
            }
            (train_loss, r_loss, kl_loss, r_loss_b, r_loss_g, r_loss_r) = (
                vae_controller.target_vae.sess.run(
                    [
                        vae_controller.target_vae.loss,
                        vae_controller.target_vae.r_loss,
                        vae_controller.target_vae.kl_loss,
                        vae_controller.target_vae.r_loss_b,
                        vae_controller.target_vae.r_loss_g,
                        vae_controller.target_vae.r_loss_r,
                    ],
                    feed,
                )
            )
            val_error_avg.append(train_loss)
            val_r_loss_avg.append(r_loss)
            val_kl_loss_avg.append(kl_loss)
            val_r_loss_r_avg.append(r_loss_r)
            val_r_loss_g_avg.append(r_loss_g)
            val_r_loss_b_avg.append(r_loss_b)
            pbar.update(1)
        pbar.close()
        train_loss = sum(val_error_avg) / len(val_error_avg)
        r_loss = sum(val_r_loss_avg) / len(val_r_loss_avg)
        kl_loss = sum(val_kl_loss_avg) / len(val_kl_loss_avg)
        r_loss_r = sum(val_r_loss_r_avg) / len(val_r_loss_r_avg)
        r_loss_g = sum(val_r_loss_g_avg) / len(val_r_loss_g_avg)
        r_loss_b = sum(val_r_loss_b_avg) / len(val_r_loss_b_avg)
        print(
            "VAE: Validation loss: ",
            train_loss,
            " r_loss: ",
            r_loss,
            " kl_loss: ",
            kl_loss,
            " r_loss_b: ",
            r_loss_b,
            " r_loss_g: ",
            r_loss_g,
            " r_loss_r: ",
            r_loss_r,
        )
        print(tf.Session().run(tf.contrib.memory_stats.MaxBytesInUse()))
        val_error.append(train_loss)
        val_r_loss.append(r_loss)
        val_kl_loss.append(kl_loss)
        val_r_loss_g.append(r_loss_g)
        val_r_loss_r.append(r_loss_r)
        val_r_loss_b.append(r_loss_b)

        np.random.seed(42)
        for j in range(10):
            image_idx = np.random.randint(n_samples_val)
            image_path = args.folder + x_images_val[image_idx]
            image = cv2.imread(image_path)
            # r = ROI
            # im = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            im = image

            print(vae_controller.target_vae.z_size)

            # encoded = vae_controller.encode(im[40:120,:,:])
            encoded = vae_controller.encode(im)

            reconstructed_image = vae_controller.decode(encoded)[0]
            g_image = vae_controller.decode_g(encoded)[0]

            encoded = vae_controller.encode(im)
            reconstructed_image_b = vae_controller.decode(encoded)[0]
            # Plot reconstruction
            print(tf.Session().run(tf.contrib.memory_stats.MaxBytesInUse()))
            print((reconstructed_image == reconstructed_image_b).all())
            cv2.imwrite("video/original{}.png".format(j), image)
            cv2.imwrite("video/recons{}.png".format(j), reconstructed_image)
            cv2.imwrite("video/lines{}.png".format(j), g_image)
            image_path_y = args.folder + y_images_val[image_idx]
            image_y = cv2.imread(image_path_y) * 255
            cv2.imwrite("video/target{}.png".format(j), image_y)
            # cv2.imshow("Original", image)
            # cv2.imshow("Reconstruction", reconstructed_image)
            # cv2.imshow("Lines", g_image)
            cv2.waitKey(0)

    plt.plot(val_error, label="Val loss")
    plt.plot(val_r_loss, label="Rec loss")
    plt.plot(val_kl_loss, label="KL Loss")
    plt.plot(val_r_loss_g, label="G Loss")
    print("Best val error:", min(val_error))

    error_list = [val_error, val_r_loss, val_kl_loss, val_r_loss_r, val_r_loss_g, val_r_loss_b]

    with open(args.vae_path + "_val.txt", "w") as f:
        for l in error_list:
            for idx, e in enumerate(l):
                text = ""
                if idx == len(l) - 1:
                    text = str(e) + "\n"
                else:
                    text = str(e) + ","
                f.write(text)
        f.close()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation error")
    plt.ylim(0, 5000)
    plt.legend()
    plt.savefig(args.vae_path + "_val.png")
