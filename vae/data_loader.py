# Original code from https://github.com/araffin/robotics-rl-srl
# Authors: Antonin Raffin, René Traoré, Ashley Hill
# Edited: Javier Moralejo
import queue
import time
from multiprocessing import Process, Queue

import cv2
import numpy as np
from joblib import Parallel, delayed

# from config import IMAGE_WIDTH, IMAGE_HEIGHT, ROI

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 128

import imgaug.augmenters as iaa


def preprocess_input(x, mode="rl", data_augmentation=False):
    """
    Normalize input.

    :param x: (np.ndarray) (RGB image with values between [0, 255])
    :param mode: (str) One of "image_net", "tf" or "rl".
        - rl: divide by 255 only (rescale to [0, 1])
        - image_net: will zero-center each color channel with
            respect to the ImageNet dataset,
            with scaling.
        - tf: will scale pixels between -1 and 1,
            sample-wise.
    :return: (np.ndarray)
    """
    assert x.shape[-1] == 3, "Color channel must be at the end of the tensor {}".format(x.shape)

    aug = iaa.Sequential(
        [
            iaa.SomeOf(
                (0, 5),
                [
                    iaa.OneOf(
                        [
                            iaa.GaussianBlur(
                                (0, 3.0)
                            ),  # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(
                                k=(2, 7)
                            ),  # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(
                                k=(3, 11)
                            ),  # blur image using local medians with kernel sizes between 2 and 7
                        ]
                    ),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                    iaa.SimplexNoiseAlpha(
                        iaa.OneOf(
                            [
                                iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                            ]
                        )
                    ),
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                    ),  # add gaussian noise to images
                    iaa.OneOf(
                        [
                            iaa.Dropout(
                                (0.01, 0.1), per_channel=0.5
                            ),  # randomly remove up to 10% of the pixels
                            iaa.CoarseDropout(
                                (0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2
                            ),
                        ]
                    ),
                    iaa.Invert(0.05, per_channel=True),  # invert color channels
                    iaa.Add(
                        (-10, 10), per_channel=0.5
                    ),  # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                    iaa.OneOf(
                        [
                            iaa.Multiply((0.5, 1.5), per_channel=0.5),
                            iaa.FrequencyNoiseAlpha(
                                exponent=(-4, 0),
                                first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                second=iaa.LinearContrast((0.5, 2.0)),
                            ),
                        ]
                    ),
                    iaa.LinearContrast(
                        (0.5, 2.0), per_channel=0.5
                    ),  # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                ],
                random_order=True,
            )
        ]
    )

    if data_augmentation:
        x = x.astype(np.uint8)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        x = aug(image=x)

        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        x = (np.clip(x, 0, 255)).astype(np.float32)

    # RL mode: divide only by 255
    x /= 255.0

    if mode == "tf":
        x -= 0.5
        x *= 2.0
    elif mode == "image_net":
        # Zero-center by mean pixel
        x[..., 0] -= 0.485
        x[..., 1] -= 0.456
        x[..., 2] -= 0.406
        # Scaling
        x[..., 0] /= 0.229
        x[..., 1] /= 0.224
        x[..., 2] /= 0.225
    elif mode == "rl":
        pass
    else:
        raise ValueError("Unknown mode for preprocessing")
    return x


def denormalize(x, mode="rl"):
    """
    De normalize data (transform input to [0, 1])

    :param x: (np.ndarray)
    :param mode: (str) One of "image_net", "tf", "rl".
    :return: (np.ndarray)
    """

    if mode == "tf":
        x /= 2.0
        x += 0.5
    elif mode == "image_net":
        # Scaling
        x[..., 0] *= 0.229
        x[..., 1] *= 0.224
        x[..., 2] *= 0.225
        # Undo Zero-center
        x[..., 0] += 0.485
        x[..., 1] += 0.456
        x[..., 2] += 0.406
    elif mode == "rl":
        pass
    else:
        raise ValueError("Unknown mode for denormalize")
    # Clip to fix numeric imprecision (1e-09 = 0)
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def preprocess_image(image, convert_to_rgb=False, data_augmentation=False):
    """
    Crop, resize and normalize image.
    Optionnally it also converts the image from BGR to RGB.

    :param image: (np.ndarray) image (BGR or RGB)
    :param convert_to_rgb: (bool) whether the conversion to rgb is needed or not
    :return: (np.ndarray)
    """
    # Crop
    # Region of interest
    # r = ROI
    # image = image[40:,:,:]
    image = image[:, :, :]
    # Resize
    # im = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    im = image
    # Convert BGR to RGB
    if convert_to_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # Normalize
    im = preprocess_input(im.astype(np.float32), mode="rl", data_augmentation=data_augmentation)

    return im


class DataLoader(object):
    def __init__(
        self,
        minibatchlist,
        x_images_path,
        y_images_path=None,
        n_workers=1,
        folder="logs/recorded_data/",
        infinite_loop=True,
        max_queue_len=32,
        is_training=False,
        data_augmentation=False,
        load_everything=True,
    ):
        """
        A Custom dataloader to preprocessing images and feed them to the network.

        :param minibatchlist: ([np.array]) list of observations indices (grouped per minibatch)
        :param images_path: (np.array) Array of path to images
        :param n_workers: (int) number of preprocessing worker (load and preprocess each image)
        :param folder: (str)
        :param infinite_loop: (bool) whether to have an iterator that can be resetted, set to False, it
        :param max_queue_len: (int) Max number of minibatches that can be preprocessed at the same time
        :param is_training: (bool)
        """
        super(DataLoader, self).__init__()
        self.n_workers = n_workers
        self.infinite_loop = infinite_loop
        self.n_minibatches = len(minibatchlist)
        self.minibatchlist = minibatchlist
        self.images_path = x_images_path
        self.y_images_path = None
        if y_images_path is None:
            self.y_images_path = None
        else:
            self.y_images_path = y_images_path
        self.shuffle = is_training
        self.data_augmentation = data_augmentation
        self.folder = folder
        self.queue = Queue(max_queue_len)

        self.load_everything = load_everything

        if self.load_everything:
            self.x_images = []
            for idx, x in enumerate(self.images_path):
                image_path = self.folder + x
                with open(image_path, "rb") as f:
                    im = f.read()
                    f.close()
                im = np.fromstring(im, np.uint8)

                self.x_images.append(im)
                if idx % 10000 == 0:
                    print(idx)

            if self.y_images_path is not None:
                self.y_images = []
                for idx, y in enumerate(self.y_images_path):
                    image_path = self.folder + y
                    with open(image_path, "rb") as f:
                        im = f.read()
                        f.close()
                    im = np.fromstring(im, np.uint8)
                    self.y_images.append(im)
                    if idx % 10000 == 0:
                        print(idx)
            else:
                self.y_images = None

        self.process = None
        self.start_process()

    @staticmethod
    def create_minibatch_list(n_samples, batch_size):
        """
        Create list of minibatches.

        :param n_samples: (int)
        :param batch_size: (int)
        :return: ([np.array])
        """
        minibatchlist = []
        for i in range(n_samples // batch_size + 1):
            start_idx = i * batch_size
            end_idx = min(n_samples, (i + 1) * batch_size)
            minibatchlist.append(np.arange(start_idx, end_idx))
        return minibatchlist

    def start_process(self):
        """Start preprocessing process"""
        self.process = Process(target=self._run)
        # Make it a deamon, so it will be deleted at the same time
        # of the main process
        self.process.daemon = True

        self.process.start()
        # self._run()

    def _run(self):

        start = True

        with Parallel(n_jobs=self.n_workers, batch_size="auto", backend="threading") as parallel:
            while start or self.infinite_loop:
                start = False

                if self.shuffle:
                    indices = np.random.permutation(self.n_minibatches).astype(np.int64)
                else:
                    indices = np.arange(len(self.minibatchlist), dtype=np.int64)

                for minibatch_idx in indices:

                    if self.load_everything:
                        images = [self.x_images[i] for i in self.minibatchlist[minibatch_idx]]
                        if self.y_images_path is not None:
                            images_y = [self.y_images[i] for i in self.minibatchlist[minibatch_idx]]
                    else:
                        images = list(self.images_path[self.minibatchlist[minibatch_idx]])
                        if self.y_images_path is not None:
                            images_y = list(self.y_images_path[self.minibatchlist[minibatch_idx]])

                    if self.y_images_path is None:
                        images_y = [None] * len(images)

                    zipped_im = zip(images, images_y)

                    if self.load_everything:
                        batch = parallel(
                            delayed(self._make_batch_element_from_bytes)(
                                self.folder,
                                x_image,
                                y_image,
                                data_augmentation=self.data_augmentation,
                            )
                            for x_image, y_image in zipped_im
                        )
                    else:
                        batch = parallel(
                            delayed(self._make_batch_element)(
                                self.folder,
                                image_path,
                                y_image_path,
                                data_augmentation=self.data_augmentation,
                            )
                            for image_path, y_image_path in zipped_im
                        )

                    x_set = []
                    y_set = []
                    for t, v in batch:
                        x_set.append(t)
                        y_set.append(v)

                    batch = np.array([x_set, y_set])

                    # batch = np.concatenate(batch, axis=0)

                    if self.shuffle:
                        # self.queue.put((minibatch_idx, batch))
                        self.queue.put(batch)
                    else:
                        self.queue.put(batch)

                    # Free memory
                    del batch

                self.queue.put(None)

    @classmethod
    def _make_batch_element(cls, folder, image_path, y_image_path=None, data_augmentation=False):
        """
        :param image_path: (str) path to an image (without the 'data/' prefix)
        :return: (np.ndarray)
        """

        image_path = folder + image_path

        im = cv2.imread(image_path)
        if im is None:
            raise ValueError("tried to load {}.jpg, but it was not found".format(image_path))

        im = preprocess_image(im, data_augmentation=data_augmentation)

        if y_image_path is None:
            y_im = im
        else:
            y_image_path = folder + y_image_path

            y_im = cv2.imread(y_image_path) * 255

            if y_im is None:
                raise ValueError("tried to load {}.jpg, but it was not found".format(y_image_path))

            y_im = preprocess_image(y_im)
        # elem = np.array([im, im])

        # elem = elem.reshape((1,) + elem.shape)

        return im, y_im

    @classmethod
    def _make_batch_element_from_bytes(
        cls, folder, image_raw, y_image_raw=None, data_augmentation=False
    ):
        """
        :param image_path: (str) path to an image (without the 'data/' prefix)
        :return: (np.ndarray)
        """

        im = cv2.imdecode(image_raw, cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError("tried to load, but it was not found")

        im = preprocess_image(im, data_augmentation=data_augmentation)

        if y_image_raw is None:
            y_im = im
        else:
            y_im = cv2.imdecode(y_image_raw, cv2.IMREAD_COLOR) * 255
            if y_im is None:
                raise ValueError("tried to load, but it was not found")

            y_im = preprocess_image(y_im)

        return im, y_im

    def __len__(self):
        return self.n_minibatches

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                val = self.queue.get_nowait()
                break
            except queue.Empty:
                time.sleep(0.001)
                continue
        if val is None:
            raise StopIteration
        return val

    def __del__(self):
        if self.process is not None:
            self.process.terminate
