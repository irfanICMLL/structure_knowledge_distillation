import os
import os.path as osp
import numpy as np
import random
# import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data

class CitySegmentationCrop(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321),
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, use_aug=False, extra_aug=False):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.use_aug = use_aug
        self.extra_aug = extra_aug
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        # if not max_iters == None:
        #     self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for item in self.img_ids:
            image_path, label_path = item
            name = osp.splitext(osp.basename(label_path))[0]
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
                "weight": 1
            })
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        # f_scale = 0.5 + random.randint(0, 16) / 10.0
        f_scale = 0.5 + random.randint(0, 1) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def RotationAug(self, image, target, max_angel=10):
        """Randomly rotates the image.
        Args:
            image: The image.
            target: The target image.
            max_angel: The maximum angel by which the image is rotated.
        Returns:
            A tuple of augmented image and target image.
        """

        # Sample the rotation factor.
        factor = np.random.uniform(-max_angel, max_angel)
        if factor < 0:
            factor += 360.0

        # Get the rotation matrix.
        h, w = image.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), factor, 1)

        image = cv2.warpAffine(image, m, (w, h))
        target = cv2.warpAffine(
            target, m, (w, h), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        return image, target

    def SaturationAug(self, image, target, delta=(0.5, 1.5)):
        """Randomly alters the image saturation.
        Args:
            image: The image.
            target: The target image.
            min_delta: Minimum deviation in the color space.
            max_delta: Maximum deviation in the color space.
        Returns:
            A tuple of augmented image and target image.
        """
        # Sample the color factor.
        factor = np.random.uniform(delta[0], delta[1])

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hsv_image[:, :, 1] *= factor
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0.0, 1.0)

        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return image, target

    def HueAug(self, image, target, delta=(-30, 30)):
        """Randomly alters the image hue.
        Args:
            image: The image.
            target: The target image.
            min_delta: Minimum deviation in the color space.
            max_delta: Maximum deviation in the color space.
        Returns:
            A tuple of augmented image and target image.
        """
        # Sample the color factor.
        factor = np.random.uniform(delta[0], delta[1])

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hsv_image[:, :, 0] += factor

        # Make sure the values are in [-360, 360].
        hsv_image[:, :, 0] += 360 * (hsv_image[:, :, 0] < 360)
        hsv_image[:, :, 0] -= 360 * (hsv_image[:, :, 0] > 360)

        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return image, target

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        if self.use_aug:  # the augmented data gt label map has been transformed
            label = label
        else:
            label = self.id2trainId(label)
        size = image.shape
        name = datafiles["name"]
        # print(name)
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        #mean = (72.41519599,82.93553322,73.18188461)
        image = image[:, :, ::-1]
        from torchvision import transforms
        data_transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize([0.290101, 0.328081, 0.286964],
                                                                   [0.182954, 0.186566, 0.184475])])

        #image-= mean



        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        #image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.extra_aug:
            # if self.random_rotate:
            image, label = self.RotationAug(image, label)
            # if self.random_saturation:
            image, label = self.SaturationAug(image, label)
            # if self.random_hue::
            image, label = self.SaturationAug(image, label)

        from PIL import Image
        image = Image.fromarray(np.uint8(image))
        image = data_transforms(image)

        return image, label.copy(), np.array(size), name



class CityscapesLoader(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128),
                 scale=True, mirror=True, ignore_label=255, use_aug=False, data_aug={}, data_norm='SUB_MEAN'):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.use_aug = use_aug
        self.data_aug = data_aug  # Dict: {'Gamma':g, 'Rotation': angel, 'Saturation': tuple, 'Hue': tuple}
        self.data_norm = data_norm
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for item in self.img_ids:
            image_path, label_path = item
            name = osp.splitext(osp.basename(label_path))[0]
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
                "weight": 1
            })
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    # Data Augmentation
    # Modified by Ke
    def generate_scale_label(self, image, label):

        f_scale = 0.5 + random.randint(0, 16) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        """
        size = (np.int(image.shape[1] * f_scale), np.int(image.shape[0] * f_scale))

        # resize image
        new_image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

        # resize target
        fy, fx = f_scale, f_scale
        H, W = target.shape
        h, w = (np.int(H * fy), np.int(W * fx))

        m = np.min(target)
        M = np.max(target)
        if m == M:
            M = m + 1

        assert -1 <= m, "Labels should not have values below -1"

        # Count the number of occurences of the labels in each "fy x fx cell"
        label_sums = np.zeros((h, w, M + 2))
        mx, my = np.meshgrid(np.arange(w), np.arange(h))
        for dy in range(fy):
            for dx in range(fx):
                label_sums[my, mx, target[dy::fy, dx::fx]] += 1

        # "Don't know" don't count.
        label_sums = label_sums[:, :, :-1]

        # Use the highest-occurence label.
        new_targets = np.argsort(label_sums, 2)[:, :, -1].astype("uint8")

        # But turn "uncertain" cells into "don't know" label.
        counts = label_sums[my, mx, new_targets]
        hit_counts = np.sum(label_sums, 2) * 0.25
        new_targets[counts <= hit_counts] = 255
        """
        return image, label

    def GammaAug(self, image, target, gamma_range=0.05):
        """Performs random gamma augmentation.
        Args:
            image: training sample
            gamma_range: The range from which to sample gamma.
        Returns:
            A tuple of augmented image.
        """
        assert 0.0 <= gamma_range <= 0.5, "Invalid gamma parameter."

        # Sample a gamma factor.
        gamma = np.random.uniform(-gamma_range, gamma_range)

        # Apply the non-linear transformation
        gamma = np.log(
            0.5 + 1 / np.sqrt(2) * gamma) / np.log(0.5 - 1 / np.sqrt(2) * gamma)

        # Perform the gamma correction.
        cv2.pow(image, gamma, image)

        return image, target

    def RotationAug(self, image, target, max_angel=10):
        """Randomly rotates the image.
        Args:
            image: The image.
            target: The target image.
            max_angel: The maximum angel by which the image is rotated.
        Returns:
            A tuple of augmented image and target image.
        """

        # Sample the rotation factor.
        factor = np.random.uniform(-max_angel, max_angel)
        if factor < 0:
            factor += 360.0

        # Get the rotation matrix.
        h, w = image.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), factor, 1)

        image = cv2.warpAffine(image, m, (w, h))
        target = cv2.warpAffine(
            target, m, (w, h), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        return image, target

    def SaturationAug(self, image, target, delta=(0.5, 1.5)):
        """Randomly alters the image saturation.
        Args:
            image: The image.
            target: The target image.
            min_delta: Minimum deviation in the color space.
            max_delta: Maximum deviation in the color space.
        Returns:
            A tuple of augmented image and target image.
        """
        # Sample the color factor.
        factor = np.random.uniform(delta[0], delta[1])

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hsv_image[:, :, 1] *= factor
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0.0, 1.0)

        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return image, target

    def HueAug(self, image, target, delta=(-30, 30)):
        """Randomly alters the image hue.
        Args:
            image: The image.
            target: The target image.
            min_delta: Minimum deviation in the color space.
            max_delta: Maximum deviation in the color space.
        Returns:
            A tuple of augmented image and target image.
        """
        # Sample the color factor.
        factor = np.random.uniform(delta[0], delta[1])

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hsv_image[:, :, 0] += factor

        # Make sure the values are in [-360, 360].
        hsv_image[:, :, 0] += 360 * (hsv_image[:, :, 0] < 360)
        hsv_image[:, :, 0] -= 360 * (hsv_image[:, :, 0] > 360)

        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return image, target

    # Data Augmentation End

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        if self.use_aug:  # the augmented data gt label map has been transformed
            label = label
        else:
            label = self.id2trainId(label)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        # Data Augmentation
        # if self.data_norm == 'SUB_MEAN':
        #     image -= self.mean
        # elif self.data_norm == '01_NORM':
        #     image = image / 255.0
        # else:
        #     raise Exception('unknown data normalization: %s' % self.data_norm)
        image_t = image - self.mean

        s_mean = (73.18188461, 82.93553322, 72.41519599)
        # image_s = image[:, :, ::-1]
        image_s = image

        image_s -= s_mean

        image_s /= 255.0
        # image_s=image/255.0
        #
        # s_mean=[0.485, 0.456, 0.406]
        # s_std=[0.229, 0.224, 0.225]
        # image_s-=s_mean
        # image_s/=s_std


        # change to BGR
        image_s = image_s.transpose((2, 0, 1))
        image_t = image_t.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image_s = image_s[:, :, ::flip]
            image_t = image_t[:, :, ::flip]
            label = label[:, ::flip]

        return image_t.copy(), image_s.copy(), label.copy(), np.array(size), name

class CitySegmentationTrain(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321),
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, use_aug=False, extra_aug=False):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.use_aug = use_aug
        self.extra_aug = extra_aug
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for item in self.img_ids:
            image_path, label_path = item
            name = osp.splitext(osp.basename(label_path))[0]
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
                "weight": 1
            })
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        # f_scale = 0.5 + random.randint(0, 16) / 10.0
        f_scale = 0.5 + random.randint(0, 16) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def RotationAug(self, image, target, max_angel=10):
        """Randomly rotates the image.
        Args:
            image: The image.
            target: The target image.
            max_angel: The maximum angel by which the image is rotated.
        Returns:
            A tuple of augmented image and target image.
        """

        # Sample the rotation factor.
        factor = np.random.uniform(-max_angel, max_angel)
        if factor < 0:
            factor += 360.0

        # Get the rotation matrix.
        h, w = image.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), factor, 1)

        image = cv2.warpAffine(image, m, (w, h))
        target = cv2.warpAffine(
            target, m, (w, h), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        return image, target

    def SaturationAug(self, image, target, delta=(0.5, 1.5)):
        """Randomly alters the image saturation.
        Args:
            image: The image.
            target: The target image.
            min_delta: Minimum deviation in the color space.
            max_delta: Maximum deviation in the color space.
        Returns:
            A tuple of augmented image and target image.
        """
        # Sample the color factor.
        factor = np.random.uniform(delta[0], delta[1])

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hsv_image[:, :, 1] *= factor
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0.0, 1.0)

        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return image, target

    def HueAug(self, image, target, delta=(-30, 30)):
        """Randomly alters the image hue.
        Args:
            image: The image.
            target: The target image.
            min_delta: Minimum deviation in the color space.
            max_delta: Maximum deviation in the color space.
        Returns:
            A tuple of augmented image and target image.
        """
        # Sample the color factor.
        factor = np.random.uniform(delta[0], delta[1])

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hsv_image[:, :, 0] += factor

        # Make sure the values are in [-360, 360].
        hsv_image[:, :, 0] += 360 * (hsv_image[:, :, 0] < 360)
        hsv_image[:, :, 0] -= 360 * (hsv_image[:, :, 0] > 360)

        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return image, target

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        if self.use_aug:  # the augmented data gt label map has been transformed
            label = label
        else:
            label = self.id2trainId(label)
        size = image.shape
        name = datafiles["name"]
        # print(name)
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        # image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.extra_aug:
            # if self.random_rotate:
            image, label = self.RotationAug(image, label)
            # if self.random_saturation:
            image, label = self.SaturationAug(image, label)
            # if self.random_hue::
            image, label = self.SaturationAug(image, label)

        return image.copy(), label.copy(), np.array(size), name


class CitySegmentationTest(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for item in self.img_ids:
            image_path = item
            name = osp.splitext(osp.basename(image_path[0]))[0]
            img_file = osp.join(self.root, image_path[0])
            self.files.append({
                "img": img_file,
                "name": name
            })
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        size = image.shape
        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        image -= self.mean
        # pdb.set_trace()
        img_h, img_w, _ = image.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
        else:
            img_pad = image

        img_h, img_w, _ = img_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        image = image.transpose((2, 0, 1))
        return image.copy(), np.array(size), name

if __name__ == '__main__':
    dst = CSDataSet("./cityscapes/", "./list/cityscapes/val.lst", crop_size=(1024, 2048))
    # trainloader = data.DataLoader(CSDataSet(args.data_dir, args.data_list, max_iters=args.num_steps*args.batch_size, crop_size=input_size,
    #             scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
    #             batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    trainloader = data.DataLoader(dst, batch_size=1, num_workers=0)

    with open("./list/cityscapes/val.lst") as f:
        train_list = f.readlines()
    train_list = [x.strip() for x in train_list]

    f_w = open("./list/cityscapes/single_cls_truck.lst", "w")
    cnt = np.zeros(19)
    for i, data in enumerate(trainloader):
        freq = np.zeros(19)
        imgs, labels, _, _ = data
        n, h, w = labels.shape
        print('prcessing {}th images ...'.format(i))
        for k in [17]:
            mask = (labels[:, :, :] == k)
            freq[k] += torch.sum(mask)
            if freq[k] > 200:
                f_w.writelines(train_list[i] + "\n")
                cnt[k] += 1
                print('# images of class {}: {}'.format(k, cnt[k]))
    print(cnt)
