"""
Train a VAE model using saved images in a folder
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from stable_baselines.common import set_global_seeds
from tqdm import tqdm

# from config import ROI
from vae.controller import VAEController
from vae.data_loader import DataLoader
from vae.model import ConvVAE

TITLE = "softmax"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--folder",
        help="Path to a folder containing images for training",
        type=str,
        default="vae/train/",
    )
    # default='vae/train')
    parser.add_argument("--z-size", help="Latent space", type=int, default=128)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--n-samples", help="Max number of samples", type=int, default=-1000)
    parser.add_argument("--batch-size", help="Batch size", type=int, default=64)
    parser.add_argument("--learning-rate", help="Learning rate", type=float, default=1e-4)
    parser.add_argument(
        "--kl-tolerance", help="KL tolerance (to cap KL loss)", type=float, default=0.5
    )  # 0.5
    parser.add_argument("--beta", help="Weight for kl loss", type=float, default=1.0)  # 1
    parser.add_argument("--n-epochs", help="Number of epochs", type=int, default=50)
    parser.add_argument("--verbose", help="Verbosity", type=int, default=0)
    parser.add_argument("--sem-seg", help="Semantic Segmentation", type=int, default=1)
    parser.add_argument("--val", help="Validation", type=str, default="")
    parser.add_argument("--aug", help="Data Augmentation", type=int, default=0)
    parser.add_argument("-t", "--title", help="Title", type=str, default="")
    parser.add_argument("-g", "--green", help="green mult", type=float, default=1.0)

    args = parser.parse_args()

    set_global_seeds(args.seed)

    if args.title != "":
        TITLE = args.title

    if not args.folder.endswith("/"):
        args.folder += "/"

    vae = ConvVAE(
        z_size=args.z_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        kl_tolerance=args.kl_tolerance,
        beta=args.beta,
        is_training=1,
        green=args.green,
        reuse=False,
    )

    target_vae = ConvVAE(
        z_size=args.z_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        kl_tolerance=args.kl_tolerance,
        beta=args.beta,
        is_training=2,
        green=args.green,
        reuse=False,
    )

    x_images = ["x/" + im for im in os.listdir(args.folder + "x/") if im.endswith(".png")]
    x_images = np.array(x_images)
    n_samples = len(x_images)

    if args.sem_seg == 1:
        y_images = ["y/" + im for im in os.listdir(args.folder + "y/") if im.endswith(".png")]
        y_images = np.array(y_images)

    if args.n_samples > 0:
        n_samples = min(n_samples, args.n_samples)
        x_images = x_images[0:n_samples]
        if args.sem_seg == 1:
            y_images = y_images[0:n_samples]

    print("{} images".format(n_samples))

    # indices for all time steps where the episode continues
    indices = np.arange(n_samples, dtype="int64")
    np.random.shuffle(indices)

    # split indices into minibatches. minibatchlist is a list of lists; each
    # list is the id of the observation preserved through the training
    minibatchlist = [
        np.array(sorted(indices[start_idx : start_idx + args.batch_size]))
        for start_idx in range(0, len(indices) - args.batch_size + 1, args.batch_size)
    ]

    if args.aug == 0:
        data_aug = False
    else:
        data_aug = True

    if args.sem_seg == 1:
        data_loader = DataLoader(
            minibatchlist,
            x_images,
            y_images_path=y_images,
            n_workers=-1,
            folder=args.folder,
            is_training=True,
            data_augmentation=data_aug,
        )
    else:
        data_loader = DataLoader(
            minibatchlist,
            x_images,
            n_workers=-1,
            folder=args.folder,
            is_training=True,
            data_augmentation=data_aug,
        )

    train_error = []

    if args.val != "":
        if not args.val.endswith("/"):
            args.val += "/"

        x_images_val = ["x/" + im for im in os.listdir(args.val + "x/") if im.endswith(".png")]
        x_images_val = np.array(x_images_val)
        n_samples_val = len(x_images_val)

        if args.sem_seg == 1:
            y_images_val = ["y/" + im for im in os.listdir(args.val + "y/") if im.endswith(".png")]
            y_images_val = np.array(y_images_val)

        print("{} images".format(n_samples_val))

        # indices for all time steps where the episode continues
        indices_val = np.arange(n_samples_val, dtype="int64")

        # split indices into minibatches. minibatchlist is a list of lists; each
        # list is the id of the observation preserved through the training
        minibatchlist_val = [
            np.array(sorted(indices_val[start_idx : start_idx + args.batch_size]))
            for start_idx in range(0, len(indices_val) - args.batch_size + 1, args.batch_size)
        ]

        # if args.sem_seg == 1:
        #    data_loader_val = DataLoader(minibatchlist_val, x_images_val, y_images_path=y_images_val, n_workers=-1, folder=args.val)
        # else:
        #    data_loader_val = DataLoader(minibatchlist_val, x_images_val, n_workers=-1, folder=args.val)

        val_error = []

    vae_controller = VAEController(z_size=args.z_size)
    vae_controller.vae = vae
    vae_controller.target_vae = target_vae

    train_error = []
    train_r_loss = []
    train_kl_loss = []
    train_r_loss_r = []
    train_r_loss_g = []
    train_r_loss_b = []
    if args.val != "":
        val_error = []

    save_path = "vae/logs/vae-{}-{}".format(args.z_size, TITLE)
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(args.n_epochs):
        pbar = tqdm(total=len(minibatchlist))

        train_error_avg = []
        train_r_loss_avg = []
        train_kl_loss_avg = []
        train_r_loss_r_avg = []
        train_r_loss_g_avg = []
        train_r_loss_b_avg = []

        for obs, obj in data_loader:

            feed = {vae.input_tensor: obs, vae.objective_tensor: obj}
            (train_loss, r_loss, kl_loss, r_loss_b, r_loss_g, r_loss_r, train_step, _) = (
                vae.sess.run(
                    [
                        vae.loss,
                        vae.r_loss,
                        vae.kl_loss,
                        vae.r_loss_b,
                        vae.r_loss_g,
                        vae.r_loss_r,
                        vae.global_step,
                        vae.train_op,
                    ],
                    feed,
                )
            )
            train_error_avg.append(train_loss)
            train_r_loss_avg.append(r_loss)
            train_kl_loss_avg.append(kl_loss)
            train_r_loss_r_avg.append(r_loss_r)
            train_r_loss_g_avg.append(r_loss_g)
            train_r_loss_b_avg.append(r_loss_b)
            pbar.update(1)
        train_loss = sum(train_error_avg) / len(train_error_avg)
        r_loss = sum(train_r_loss_avg) / len(train_r_loss_avg)
        kl_loss = sum(train_kl_loss_avg) / len(train_kl_loss_avg)
        r_loss_r = sum(train_r_loss_r_avg) / len(train_error_avg)
        r_loss_g = sum(train_r_loss_g_avg) / len(train_r_loss_g_avg)
        r_loss_b = sum(train_r_loss_b_avg) / len(train_r_loss_b_avg)
        pbar.close()
        print("Epoch {:3}/{}".format(epoch + 1, args.n_epochs))
        print(
            "VAE: train loss: ",
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
            " optimization step: ",
            (train_step + 1),
        )
        print(tf.Session().run(tf.contrib.memory_stats.MaxBytesInUse()))
        train_error.append(train_loss)
        train_r_loss.append(r_loss)
        train_kl_loss.append(kl_loss)
        train_r_loss_r.append(r_loss_r)
        train_r_loss_g.append(r_loss_g)
        train_r_loss_b.append(r_loss_b)
        # Update params
        # vae_controller.set_target_params()
        vae_controller.save_train("{}/{}-{:02d}".format(save_path, TITLE, epoch + 1))
        # Load test image

        if args.val != "":
            pbar = tqdm(total=len(minibatchlist_val))
            for obs, obj in data_loader_val:
                feed = {vae.input_tensor: obs, vae.objective_tensor: obj}
                (train_loss, r_loss, kl_loss) = vae.sess.run(
                    [vae.loss, vae.r_loss, vae.kl_loss], feed
                )
                pbar.update(1)
            pbar.close()
            print(
                "VAE: validation loss: ",
                train_loss,
                " r_loss_val: ",
                r_loss,
                " kl_loss_val: ",
                kl_loss,
            )
            print(tf.Session().run(tf.contrib.memory_stats.MaxBytesInUse()))
            val_error.append(train_loss)

        if args.verbose >= 1:
            if args.val == "":
                image_idx = np.random.randint(n_samples)
                image_path = args.folder + x_images[image_idx]
            else:
                image_idx = np.random.randint(n_samples_val)
                image_path = args.val + x_images_val[image_idx]
            image = cv2.imread(image_path)
            # r = ROI
            # im = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            im = image

            # encoded = vae_controller.encode(im[40:120,:,:])
            encoded = vae_controller.encode_train(im)

            reconstructed_image = vae_controller.decode_train(encoded)[0]
            g_image = vae_controller.decode_train_g(encoded)[0]

            encoded = vae_controller.encode_train(im)
            reconstructed_image_b = vae_controller.decode_train(encoded)[0]
            # Plot reconstruction
            print(tf.Session().run(tf.contrib.memory_stats.MaxBytesInUse()))
            print((reconstructed_image == reconstructed_image_b).all())
            cv2.imshow("Original", image)
            cv2.imshow("Reconstruction", reconstructed_image)
            cv2.imshow("Lines", g_image)
            cv2.waitKey(1)

    error_list = [
        train_error,
        train_r_loss,
        train_kl_loss,
        train_r_loss_r,
        train_r_loss_g,
        train_r_loss_b,
    ]

    print("Saving to {}".format(save_path + "/" + TITLE))
    vae_controller.set_target_params()
    vae_controller.save(save_path + "/" + TITLE)

    plt.plot(train_error, label="Train")
    with open(save_path + "/" + TITLE + "_train.txt", "w") as f:
        for l in error_list:
            for idx, e in enumerate(l):
                text = ""
                if idx == len(l) - 1:
                    text = str(e) + "\n"
                else:
                    text = str(e) + ","
                f.write(text)
        f.close()

    if args.val != "":
        plt.plot(val_error, label="Validation")
        print("Best val error:", min(val_error))
        with open(save_path + "/" + TITLE + ".txt", "w") as f:
            f.write(str(min(val_error)))
            f.close()

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(TITLE)
    plt.ylim(0, 5000)
    plt.legend()
    plt.savefig(save_path + "/" + TITLE + "_train")
