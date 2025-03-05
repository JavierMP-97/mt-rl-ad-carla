# Original author: Roma Sokolkov
# Edited by Antonin Raffin
import os
import random
import warnings

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.utils import seeding

from config import (
    ACTION_NOISE,
    INCLUDE_ACCEL,
    INCLUDE_ANGLE_DIFF,
    INCLUDE_CTE,
    INCLUDE_JUNCTION,
    INCLUDE_SPEED,
    INCLUDE_VAE,
    INPUT_DIM,
    JERK_REWARD_WEIGHT,
    LEADING_INSTRUCTIONS,
    MAX_STEERING,
    MAX_STEERING_DIFF,
    MAX_THROTTLE,
    MIN_STEERING,
    MIN_THROTTLE,
)
from vae.controller import VAEController

from .carla_sim import ROAD_OPTIONS, CarlaSimContoller


def load_vae(path=None, z_size=None):
    """
    :param path: (str)
    :param z_size: (int)
    :return: (VAEController)
    """
    # z_size will be recovered from saved model
    if z_size is None:
        assert path is not None

    vae = VAEController(z_size=z_size)
    if path is not None:
        vae.load(path)
    print("Dim VAE = {}".format(vae.z_size))
    return vae


def semantic_transformation(img):
    new_img_b = (img[:, :, 2] == 7).reshape((img.shape[0], img.shape[1], 1))
    new_img_g = (img[:, :, 2] == 6).reshape((img.shape[0], img.shape[1], 1))
    new_img_r = (
        ((img[:, :, 2] < 6) & (img[:, :, 2] > 0))
        + ((img[:, :, 2] > 7) & (img[:, :, 2] < 10))
        + ((img[:, :, 2] > 10) & (img[:, :, 2] < 13))
        + (img[:, :, 2] > 13)
    ).reshape((img.shape[0], img.shape[1], 1))
    new_img = np.concatenate((new_img_b, new_img_g, new_img_r), axis=-1)
    new_img = new_img.astype(int)
    return new_img


class DonkeyVAEEnv(gym.Env):
    """
    Gym interface for DonkeyCar with support for using
    a VAE encoded observation instead of raw pixels if needed.

    :param level: (int) DonkeyEnv level
    :param frame_skip: (int) frame skip, also called action repeat
    :param vae: (VAEController object)
    :param const_throttle: (float) If set, the car only controls steering
    :param min_throttle: (float)
    :param max_throttle: (float)
    :param max_cte_error: (float) Max cross track error before ending an episode
    :param n_command_history: (int) number of previous commmands to keep
        it will be concatenated with the vae latent vector
    :param n_stack: (int) Number of frames to stack (used in teleop mode only)
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        level=0,
        frame_skip=2,
        vae=None,
        const_throttle=None,
        min_throttle=0.2,
        max_throttle=0.5,
        max_cte_error=3.0,
        n_command_history=0,
        n_stack=1,
        port=2000,
        save_for_vae=False,
        save_obs=True,
        show_vae=False,
    ):
        self.vae = vae
        self.z_size = None
        if vae is not None:
            self.z_size = vae.z_size

        self.const_throttle = const_throttle
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self.np_random = None

        # Save last n commands (throttle + steering)
        self.n_commands = 2
        self.command_history = np.zeros((1, self.n_commands * n_command_history))
        self.n_command_history = n_command_history
        # Custom frame-stack
        self.n_stack = n_stack
        self.stacked_obs = None

        self.n_image = 0
        self.save_obs = save_obs

        self.image_number = 80000
        self.ep_len = 0
        self.save_for_vae = save_for_vae
        self.im_counter = 0

        if self.save_obs == True:
            self.second_vae = load_vae("vae-128-no-sem-seg-49.pkl")
            if not os.path.isfile("video/info.txt"):
                with open("video/info.txt", "a+") as info:
                    info.write("ticks,speed,cte,angle_diff,throttle,steering\n")
                    info.close()

        # start simulation com
        if show_vae:
            self.viewer = CarlaSimContoller(
                level=level, port=port, max_cte_error=max_cte_error, vae=self.vae
            )
        else:
            self.viewer = CarlaSimContoller(
                level=level, port=port, max_cte_error=max_cte_error, vae=None
            )

        if const_throttle is not None:
            # steering only
            self.action_space = spaces.Box(
                low=np.array([-MAX_STEERING]), high=np.array([MAX_STEERING]), dtype=np.float32
            )
        else:
            # steering + throttle, action space must be symmetric
            self.action_space = spaces.Box(
                low=np.array([-MAX_STEERING, -1]),
                high=np.array([MAX_STEERING, 1]),
                dtype=np.float32,
            )

        if vae is None:
            # Using pixels as input
            if n_command_history > 0:
                warnings.warn(
                    "n_command_history not supported for images"
                    "(it will not be concatenated with the input)"
                )
            self.observation_space = spaces.Box(low=0, high=255, shape=INPUT_DIM, dtype=np.uint8)
        else:
            # z latent vector from the VAE (encoded input image)
            self.observation_space = spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                # shape=(1, self.z_size),
                shape=(
                    1,
                    self.z_size * INCLUDE_VAE
                    + self.n_commands * n_command_history
                    + LEADING_INSTRUCTIONS * len(ROAD_OPTIONS)
                    + INCLUDE_SPEED * 3
                    + INCLUDE_SPEED * 3
                    + INCLUDE_CTE
                    + INCLUDE_ANGLE_DIFF
                    + INCLUDE_JUNCTION,
                ),
                dtype=np.float32,
            )

        # Frame-stacking with teleoperation
        if n_stack > 1:
            obs_space = self.observation_space
            low = np.repeat(obs_space.low, self.n_stack, axis=-1)
            high = np.repeat(obs_space.high, self.n_stack, axis=-1)
            self.stacked_obs = np.zeros(low.shape, low.dtype)
            self.observation_space = spaces.Box(low=low, high=high, dtype=obs_space.dtype)

        self.seed()
        # Frame Skipping
        self.frame_skip = frame_skip
        # wait until loaded
        self.viewer.wait_until_loaded()

    def close_connection(self):
        return self.viewer.close_connection()

    def exit_scene(self):
        self.viewer.quit()

    def jerk_penalty(self):
        """
        Add a continuity penalty to limit jerk.
        :return: (float)
        """
        jerk_penalty = 0
        if self.n_command_history > 1:
            # Take only last command into account
            for i in range(int(self.n_command_history / 2)):
                steering = self.command_history[0, -2 * (i + 1)]
                prev_steering = self.command_history[0, -2 * (i + 2)]
                steering_diff = abs((prev_steering - steering) / (MAX_STEERING - MIN_STEERING))

                max_steer_diff = 0.1

                if steering_diff > max_steer_diff:
                    error = steering_diff - max_steer_diff
                    jerk_penalty += JERK_REWARD_WEIGHT * error
                else:
                    jerk_penalty += 0
        return jerk_penalty

    def postprocessing_step(self, action, observation, reward, done, info):
        """
        Update the reward (add jerk_penalty if needed), the command history
        and stack new observation (when using frame-stacking).

        :param action: ([float])
        :param observation: (np.ndarray)
        :param reward: (float)
        :param done: (bool)
        :param info: (dict)
        :return: (np.ndarray, float, bool, dict)
        """
        # Update command history
        if self.n_command_history > 0:
            self.command_history = np.roll(self.command_history, shift=-self.n_commands, axis=-1)
            self.command_history[..., -self.n_commands :] = action

            if self.vae is not None:
                observation = np.concatenate((observation, self.command_history), axis=-1)

        if self.vae is not None:
            aditional_info = []
            if INCLUDE_CTE:
                aditional_info += [info["cte"]]
            if INCLUDE_ANGLE_DIFF:
                aditional_info += [info["angle_diff"]]
            if INCLUDE_JUNCTION:
                aditional_info += [info["is_junction"]]
            if INCLUDE_SPEED:
                aditional_info += list(info["vel"])
            if INCLUDE_ACCEL:
                aditional_info += list(info["accel"])
            if LEADING_INSTRUCTIONS > 0:
                aditional_info += list(info["road_opt"])
            aditional_info = np.array([aditional_info])
            observation = np.concatenate((observation, aditional_info), axis=-1)

        jerk_penalty = self.jerk_penalty()
        # Cancel reward if the continuity constrain is violated
        # if jerk_penalty > 0 and reward > 0:
        #    reward = 0
        reward -= jerk_penalty

        if self.n_stack > 1:
            self.stacked_obs = np.roll(self.stacked_obs, shift=-observation.shape[-1], axis=-1)
            if done:
                self.stacked_obs[...] = 0
            self.stacked_obs[..., -observation.shape[-1] :] = observation
            return self.stacked_obs, reward, done, info

        return observation, reward, done, info

    def step(self, action):
        """
        :param action: (np.ndarray)
        :return: (np.ndarray, float, bool, dict)
        """
        # action[0] is the steering angle
        # action[1] is the throttle
        if self.const_throttle is not None:
            action = np.concatenate([action, [self.const_throttle]])
        else:
            # Convert from [-1, 1] to [0, 1]
            t = (action[1] + 1) / 2
            # Convert from [0, 1] to [min, max]
            action[1] = (1 - t) * self.min_throttle + self.max_throttle * t

        # Clip steering angle rate to enforce continuity
        if self.n_command_history > 0:
            prev_steering = self.command_history[0, -2]
            max_diff = (MAX_STEERING_DIFF - 1e-5) * (MAX_STEERING - MIN_STEERING)
            diff = np.clip(action[0] - prev_steering, -max_diff, max_diff)
            action[0] = prev_steering + diff

        if ACTION_NOISE > 0:
            action[0] = action[0] + random.gauss(0, ACTION_NOISE) * (MAX_STEERING - MIN_STEERING)
            action[1] = action[1] + random.gauss(0, ACTION_NOISE) * (MAX_THROTTLE - MIN_THROTTLE)

        action[0] = np.clip(action[0], MIN_STEERING, MAX_STEERING)
        action[1] = np.clip(action[1], MIN_THROTTLE, MAX_THROTTLE)

        # Repeat action if using frame_skip
        for _ in range(self.frame_skip):
            self.viewer.take_action(action)
            observation, reward, done, info = self.observe()

        return self.postprocessing_step(action, observation, reward, done, info)

    def reset(self, next_weather=None):
        self.viewer.reset(next_weather=next_weather)
        self.command_history = np.zeros((1, self.n_commands * self.n_command_history))
        observation, reward, done, info = self.observe()

        if self.n_command_history > 0:
            if self.vae is not None:
                observation = np.concatenate((observation, self.command_history), axis=-1)

        if self.vae is not None:
            aditional_info = []
            if INCLUDE_CTE:
                aditional_info += [info["cte"]]
            if INCLUDE_ANGLE_DIFF:
                aditional_info += [info["angle_diff"]]
            if INCLUDE_JUNCTION:
                aditional_info += [info["is_junction"]]
            if INCLUDE_SPEED:
                aditional_info += list(info["vel"])
            if INCLUDE_ACCEL:
                aditional_info += list(info["accel"])
            if LEADING_INSTRUCTIONS > 0:
                aditional_info += list(info["road_opt"])
            aditional_info = np.array([aditional_info])
            observation = np.concatenate((observation, aditional_info), axis=-1)

        if self.n_stack > 1:
            self.stacked_obs[...] = 0
            self.stacked_obs[..., -observation.shape[-1] :] = observation
            return self.stacked_obs

        return observation

    def render(self, mode="human"):
        """
        :param mode: (str)
        """
        if mode == "rgb_array":
            return self.viewer.current_img
        elif mode == "semantic_array":
            return self.viewer.current_sem
        return None

    def observe(self):
        """
        Encode the observation using VAE if needed.

        :return: (np.ndarray, float, bool, dict)
        """
        observation, reward, done, info = self.viewer.observe()
        # Learn from Pixels
        if self.vae is None:
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            observation = cv2.Canny(observation, 100, 250)
            observation = observation / 255
            observation = observation.reshape(INPUT_DIM)
            return observation, reward, done, info
        # Encode the image

        encoded = self.vae.encode(observation)

        if self.save_for_vae == True:
            self.im_counter += 1
            if self.im_counter >= 10:
                current_sem = semantic_transformation(self.viewer.current_sem)
                cv2.imwrite(
                    "vae/train/x/img_{:05d}.png".format(self.image_number), self.viewer.current_img
                )
                cv2.imwrite("vae/train/y/img_{:05d}.png".format(self.image_number), current_sem)
                with open("vae/train/d/data_{:05d}.txt".format(self.image_number), "w") as f:
                    f.write(
                        "{},{},{},{},{},{},{},{},{},{},{},{}".format(
                            info["cte"],
                            info["angle_diff"],
                            info["is_junction"],
                            info["speed"],
                            info["throttle"],
                            info["steering"],
                            info["road_opt"][0],
                            info["road_opt"][1],
                            info["road_opt"][2],
                            info["road_opt"][3],
                            info["road_opt"][4],
                            info["road_opt"][5],
                        )
                    )

                self.image_number += 1
                self.im_counter = 0

        if self.save_obs == True and self.viewer.controller.record == True:

            # decoded = np.squeeze(self.vae.decode(encoded))

            # decoded_2 = np.squeeze(self.second_vae.decode(self.second_vae.encode(self.viewer.current_img)))

            # plt.imshow(observation)
            cv2.imwrite("video/obs{:04d}.png".format(self.n_image), observation)

            # plt.imshow(decoded)
            # cv2.imwrite("video/dec{:04d}.png".format(self.n_image), decoded)

            # plt.imshow(decoded)
            # cv2.imwrite("video/nosem{:04d}.png".format(self.n_image), decoded_2)

            # plt.imshow(decoded)
            # cv2.imwrite("video/obj{:04d}.png".format(self.n_image), 255 * semantic_transformation(self.viewer.current_sem))

            with open("video/info.txt", "a") as inf:
                inf.write(
                    "{},{},{},{},{}\n".format(
                        self.n_image,
                        info["speed"],
                        info["cte"],
                        info["angle_diff"],
                        info["throttle"],
                        info["steering"],
                    )
                )
                inf.close()

            self.n_image += 1

        if not INCLUDE_VAE:
            encoded = np.array([[]])

        return encoded, reward, done, info

    def close(self):
        if self.unity_process is not None:
            self.unity_process.quit()
        self.viewer.quit()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_vae(self, vae):
        """
        :param vae: (VAEController object)
        """
        self.vae = vae
