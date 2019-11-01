import numpy as np
import torch
import random
import numbers
from skimage import transform
from PIL import Image
from . import functional as F


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class x1y1x2y2(object):
    # orig_type 0 = x1, y1, W, H
    # orig_type 1 = x1, y1, x2, y2
    # orig_type 2 = cx, cy, W, H

    def __init__(self, orig_type):
        assert isinstance(orig_type, int)
        self.orig_type = orig_type

    def __call__(self, sample):
        image, bbox = sample['data'], sample['bbox']

        if self.orig_type == 0:  # Left(0), Top(1), Width(2), Height(3)
            bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
            bbox[:, 3] = bbox[:, 1] + bbox[:, 3]

        else:
            bbox[:, 0] = bbox[:, 0] - int(bbox[:, 2]/2)
            bbox[:, 1] = bbox[:, 1] - int(bbox[:, 3]/2)
            bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
            bbox[:, 3] = bbox[:, 1] + bbox[:, 3]

        return {'data': image, 'bbox': bbox}


class y1x1y2x2(object):
    # orig_type 0 = x1, y1, W, H
    # orig_type 1 = x1, y1, x2, y2
    # orig_type 2 = cx, cy, W, H

    def __init__(self, orig_type):
        assert isinstance(orig_type, int)
        self.orig_type = orig_type

    def __call__(self, sample):
        image, bbox = sample['data'], sample['bbox']

        if self.orig_type == 0:  # Left, Top, Width, Height
            bbox[:, [0, 1]] = bbox[:, [1, 0]]
            bbox[:, 2] = bbox[:, 1] + bbox[:, 2]
            bbox[:, 3] = bbox[:, 0] + bbox[:, 3]
            bbox[:, [2, 3]] = bbox[:, [3, 2]]

        elif self.orig_type == 1:
            bbox[:, [0, 1]] = bbox[:, [1, 0]]
            bbox[:, [2, 3]] = bbox[:, [3, 2]]

        else:
            bbox[:, 0] = bbox[:, 0] - int(bbox[:, 2]/2)
            bbox[:, 1] = bbox[:, 1] - int(bbox[:, 3]/2)
            bbox[:, [0, 1]] = bbox[:, [1, 0]]
            bbox[:, 2] = bbox[:, 1] + bbox[:, 2]
            bbox[:, 3] = bbox[:, 0] + bbox[:, 3]
            bbox[:, [2, 3]] = bbox[:, [3, 2]]

        return {'data': image, 'bbox': bbox}


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)

        return {'data': np.array(transform(sample['data'])), 'bbox': sample['bbox']}

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
            self.width = output_size
            self.height = output_size
        else:
            assert len(output_size) == 2
            self.width = random.uniform(output_size[0], output_size[1])
            self.height = random.uniform(output_size[0], output_size[1])
            self.output_size = output_size

    def __call__(self, sample):
        image, bbox = sample['data'], sample['bbox']
        h, w = image.shape[:2]
        new_h, new_w = int(h*self.height), int(w*self.width)

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        crop_image = image[top: top + new_h, left: left + new_w]

        crop_box = []
        for box in bbox:
            c_x = int((box[3]-box[1])/2)
            c_y = int((box[2]-box[0])/2)

            if left < c_x < left+new_w and top < c_y < top+new_h:
                x1 = box[1] - left
                y1 = box[0] - top
                x2 = box[3] - left
                y2 = box[2] - top
                if max(left, box[1]) == left:
                    x1 = 0
                if max(top, box[0]) == top:
                    y1 = 0
                if min(left+new_w, box[3]) == left+new_w:
                    x2 = new_w
                if min(top+new_h, box[2]) == top+new_h:
                    y2 = new_h

                box = np.array([y1, x1, y2, x2])
                crop_box.append(box)

        crop_box = np.array(crop_box)

        if len(crop_box) == 0:
            crop_box = bbox
            crop_image = image

        return {'data': crop_image, 'bbox': crop_box}

def annotransform(boxes, im_width, im_height):

    boxes[:, 0] = boxes[:, 0]/im_width
    boxes[:, 1] = boxes[:, 1]/im_height
    boxes[:, 2] = boxes[:, 2]/im_width
    boxes[:, 3] = boxes[:, 3]/im_height

    return boxes


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, bbox = sample['data'], sample['bbox']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        print(len(bbox), len(bbox[0]))
        print(bbox[:])
        print(bbox[:, :4])
        bbox[:, :4] = bbox[:, :4] * [new_w / w, new_h / h, new_w / w, new_h / h]
        bbox = bbox.astype(float)
        bbox[:, :4] = annotransform(bbox[:, :4], 640, 640)
        # bbox = np.hstack((bbox[:, 4][:, np.newaxis], bbox[:, :4]))
        return {'data': img, 'bbox': bbox}


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample):
        image = sample['data']
        bbox = sample['bbox']

        if random.random() < self.prob:
            height, width = image.shape[-2:]
            img_center = np.array([int(height/2), int(width/2)])
            img_center = np.hstack((img_center, img_center))
            image = np.fliplr(image).copy()
            bbox[:, [1, 3]] += 2 * (img_center[[0, 2]] - bbox[:, [1, 3]])
            bbox[:, [1, 3]] = bbox[:, [3, 1]]
        return {'data': image, 'bbox': bbox}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bbox = sample['data'], sample['bbox']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = F.to_tensor(image)
        return {'data': image, 'bbox': torch.FloatTensor(bbox)}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        tensor = sample['data']
        bbox = sample['bbox']
        tensor = F.normalize(tensor, self.mean, self.std, self.inplace)
        return {'data': tensor, 'bbox': bbox}

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
