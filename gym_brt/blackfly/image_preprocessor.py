import time
import numpy as np
import cv2
import os
import shutil
from tqdm import tqdm

"""
If preprocess=True the image gets mapped to a one channel image where non-red areas are replaced by 0. If no 
preprocessing is needed object still needs to be used for normalization and reshaping.

The object can be used for different operations:
1.  Real time inference: preprocess_and_normalize_image()
2.  Preprocessing whole datasets: preprocess_dataset()
3.  for learning (with and without data generators): normalize_image() and load_and_normalize_image()
    It is important to use the right functions for training and real time use. The images get preprocessed 
    differently, even if it is used for training with generators and without generators. Visualize the images before 
    feeding them to the training. Range needs to be in (0,1) not (0,255) and channel order needs to be consistent.
"""

IMAGE_SHAPE = (220, 220, 3)


class ImagePreprocessor:

    def __init__(self, preprocess, image_shape=IMAGE_SHAPE):
        self.image_shape = image_shape
        self.preprocess = preprocess
        if preprocess:
            image_depth_preprocessed = 1  # number of channels after preprocessing, needs to be aligned with the preprocessing fct.
            self.color_mode = 'grayscale'
        else:
            image_depth_preprocessed = 3
            self.color_mode = 'rgb'
        self.image_shape_preprocessed = (image_shape[0], image_shape[1], image_depth_preprocessed)

    def preprocess_image(self, image):
        # if image.shape[0] != self.image_shape_preprocessed[0]:
        #     image = cv2.resize(image, self.image_shape[:-1])
        # else:
        #     pass
        #     # image = image[:, :, ::-1]
        image = cv2.resize(image, self.image_shape[:-1])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.preprocess and image.shape[2] != 1:
            img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([15, 255, 255])
            mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

            lower_red = np.array([165, 50, 50])
            upper_red = np.array([180, 255, 255])
            mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

            mask = mask0 + mask1
            image[np.where(mask == 0)] = 0

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            image = clahe.apply(image)

        # image = cv2.Canny(image, 100, 200)

        image = np.reshape(image, self.image_shape_preprocessed)
        return image

    def normalize_image(self, image):
        # for generators
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image / 255.

    def load_and_normalize_image(self, path):
        # for data loading without generators
        if self.preprocess:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            image = np.reshape(image, self.image_shape_preprocessed)
        else:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
        return image / 255.

    def preprocess_and_normalize_image(self, image):
        # for real time inference
        image = self.preprocess_image(image)
        return image / 255.

    def _load_preprocess_save(self, path):
        image = cv2.imread(path)
        image = self.preprocess_image(image)
        cv2.imwrite(path, image)

    def preprocess_dataset(self, path):
        print("Preprocessing all images ...")
        if not os.path.exists(path + "/img_raw"):
            os.mkdir(path + "/img_raw")
            path += "/img/"
            img_paths = [f for f in os.listdir(path)]
            for i, f in enumerate(tqdm(img_paths, desc='Preprocess image')):
                f = path + f
                f_new = f.replace("img", "img_raw")
                if not os.path.exists(f_new):
                    shutil.copyfile(f, f_new)
                self._load_preprocess_save(f)
                # image = cv2.imread(f)
                # cv2.imshow("train", image)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
        else:
            path += "/img_raw/"
            img_paths = [f for f in os.listdir(path)]
            for i, f in enumerate(tqdm(img_paths, desc='Preprocess image')):
                f = path + f
                f_new = f.replace("img_raw", "img")
                shutil.copyfile(f, f_new)
                self._load_preprocess_save(f_new)

        if self.preprocess:
            print("Images in " + path + " resized and preprocessed, still need to be NORMALIZED")
        else:
            print("Images in " + path + " resized, still need to be NORMALIZED")
