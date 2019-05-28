import os
import torch
import numpy
import random
from torch.utils import data
from PIL import Image, ImageOps, ImageFilter
try:
    import accimage
except ImportError:
    accimage = None

N_CLASS = 19
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
color_list = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
color_map = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
             (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
             (255,  0,  0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_label_file(filename):
    return filename.endswith('gtFine_labelIds.png') or filename.endswith('gtCoarse_labelIds.png')


def make_image_list(dir):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images


def make_label_list(dir):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_label_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images


def change_label(label):
    shape = label.shape
    out = numpy.ones(shape=(shape[0], shape[1]), dtype=numpy.float32) * (-1)

    for i in range(N_CLASS):
        class_id = color_list[i]
        out[label[:] == class_id] = i

    return out


class SegDataLoader(data.Dataset):

    def __init__(self, data_path, label_path, transform=None):
        self.imgs = make_image_list(data_path)
        self.labels = make_label_list(label_path)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_file = self.imgs[index]
        lbl_file = self.labels[index]

        # load image and label
        img = Image.open(img_file)
        lbl = Image.open(lbl_file)
        lbl = lbl.convert('L')

        if self.transform is not None:
            img, lbl = self.transform(img, lbl)

        lbl = numpy.array(lbl, dtype=numpy.uint8)
        lbl = change_label(lbl)
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


class RandomCrop(object):
    """ Crop the given PIL.Image at a random location. """

    def __init__(self, crop_size, is_padding=True):
        """
        :param crop_size:
            Desired output size of the crop. If size is an int instead of sequence like (h, w),
            a square crop (size, size) is made.
        :param is_padding:
            (int or sequence, optional): Optional padding on each border of the image. Default is 0, i.e no padding.
            If a sequence of length 4 is provided, it is used to pad left, top, right, bottom borders respectively.
        """
        if isinstance(crop_size, int):
            self.size = (crop_size, crop_size)
        else:
            self.size = crop_size

        self.padding = is_padding

    def __call__(self, img, label):
        """
        :param img:
            (PIL.Image), image to be cropped.
        :param label:
            (PIL.Image), label to be cropped.
        :return:
            cropped img
            cropped label
        """
        w, h = img.size
        tw, th = self.size

        if w == tw and h == th:
            return img, label

        if self.padding and w < tw:
            w_p = tw - w
            img = ImageOps.expand(img, border=(w_p, w_p, 0, 0), fill=0)
            label = ImageOps.expand(label, border=(w_p, w_p, 0, 0), fill=0)
            w, h = img.size
        if self.padding and h < th:
            h_p = th - h
            img = ImageOps.expand(img, border=(0, 0, h_p, h_p), fill=0)
            label = ImageOps.expand(label, border=(0, 0, h_p, h_p), fill=0)
            w, h = img.size

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        return img.crop((x1, y1, x1 + tw, y1 + th)), label.crop((x1, y1, x1 + tw, y1 + th))


class Normalize(object):
    """ Normalize an tensor image with mean and standard deviation. """

    def __init__(self, mean, std):
        """
        Given mean: (R, G, B) and std: (R, G, B), will normalize each channel of the torch.*Tensor, i.e.
            channel = (channel - mean) / std
        :param mean:
            (sequence), Sequence of means for R, G, B channels respecitvely.
        :param std:
            (sequence), Sequence of standard deviations for R, G, B channels
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor, label):
        """
        :param tensor:
            Tensor image of size (C, H, W) to be normalized.
        :param label:
            ndarray label, will not be processed
        :return:
            Normalized image, label
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)

        return tensor, label


class RandomBrightness(object):
    """ Randomly adjust the brightness of input image."""

    def __init__(self, shift_value=10):
        """
        :param shift_value: random number of [-shift_value, shift_value] will be chose to adjust the brightness.
        """
        self.shift_value = shift_value

    def __call__(self, img, label):
        """
        :param img:
            (PIL.Image), image to be adjust brightness.
        :param label:
            ndarray label, will not be processed.
        :return:
            adjusted image, label.
        """
        shift = numpy.random.uniform(-self.shift_value, self.shift_value, size=1)
        image = numpy.array(img, dtype=float)
        image[:, :, :] += shift
        image = numpy.around(image)
        image = numpy.clip(image, a_min=0, a_max=255)
        image = image.astype(numpy.uint8)
        image = Image.fromarray(image)

        return image, label


class RandomGaussNoise(object):
    """ Randomly add noise on input image."""

    def __init__(self, shift_value=3):
        """
        :param shift_value: sigma of the sigma of added noise.
        """
        self.shift_value = shift_value

    def __call__(self, img, label):
        """
        :param img:
            (PIL.Image), image to be adjust brightness.
        :param label:
            ndarray label, will not be processed.
        :return:
            adjusted image, label.
        """
        image = numpy.array(img, dtype=float)
        sigma = abs(numpy.random.normal(0, self.shift_value))
        if sigma > 20:
            sigma = 20
        shift = numpy.random.normal(0, sigma, size=image.shape)
        image[:, :, :] += shift
        image = numpy.around(image)
        image = numpy.clip(image, a_min=0, a_max=255)
        image = image.astype(numpy.uint8)
        image = Image.fromarray(image)

        return image, label


class RandomGaussBlur(object):
    """ Randomly blur image. """

    def __init__(self, max_blur=4):
        """
        :param max_blur:
            max value used for blur.
        """
        self.max_blur = max_blur

    def __call__(self, img, label):
        """
        :param img:
            (PIL.Image), image to be blurred.
        :param label:
            ndarray label, will not be processed.
        :return:
            adjusted image, label.
        """
        if self.max_blur > 0:
            blur_value = numpy.random.uniform(0, self.max_blur)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_value))

        return img, label


class RandomHorizontalFlip(object):
    """ Horizontally flip the given PIL.Image randomly with a probability of 0.5. """

    def __call__(self, img, label):
        """
        :param img:
            (PIL.Image), Image to be flipped.
        :param label:
            (PIL.Image), label to be flipped.
        :return:
            flipped image and label.
        """
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img, label


class RandomScale(object):
    """ random Rescale the input image and correspond label to the given size. """

    def __init__(self, min_size, max_size, interpolation=Image.BICUBIC):
        """
        :param min_size:
            (int), Desired  min output size.
        :param max_size:
            (int), Desired  max output size.
        :param interpolation:
            Desired interpolation for image, for not introduce unknown label, the label using NEAREST as default.
        """
        self.minsize = min_size
        self.maxsize = max_size
        self.interpolation = interpolation

    def __call__(self, img, label):
        """
        :param img:
            (PIL.Image), image to be scaled.
        :param label:
            (PIL.Image), label to be scaled.
        :return:
            Rescaled image and label.
        """
        w, h = img.size
        ratio = numpy.random.uniform(self.minsize, self.maxsize)

        ow = int(w * ratio)
        oh = int(h * ratio)
        return img.resize((ow, oh), self.interpolation), label.resize((ow, oh), Image.NEAREST)


class ToTensor(object):
    """ Converts a PIL.Image in the range [0, 255] to a torch.FloatTensor of shape (CxHxW) in the range [0., 1.]. """

    def __call__(self, pic, label):
        """
        :param pic:
            (PIL.Image): Image to be converted to tensor.
        :param label:
            ndarray label, will not be processed.
        :return:
            Converted image, label.
        """
        pic = numpy.array(pic)  # change PIL.Image into numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))

        return img.float().div(255), label


class Compose(object):
    """ Composes several transforms together. """

    def __init__(self, transforms):
        """
        :param transforms:
            A list of transformations used for data augmentation.
        """
        self.transforms = transforms

    def __call__(self, img, label):
        """
        :param img:
            image to be transformed.
        :param label:
            correspond label.
        :return:
            transformed image and label.
        """
        for t in self.transforms:
            img, label = t(img, label)

        return img, label


