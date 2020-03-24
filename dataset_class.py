from __future__ import print_function, division
import glob
import torch
import torch.nn as nn
import torch.nn.functional as NN_Func
from torch.utils.data import Dataset
from torchvision import utils
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import settings



class Enchancer_Dataset(Dataset):
    def __init__(self, img_path_list, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_path_list = img_path_list
        self.transform = transform

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        img_name = self.image_path_list[idx]
        image_input  = io.imread(img_name)
        image_output = io.imread(img_name)
        sample = {'input': image_input, 'output': image_output}
        if self.transform:
            sample = self.transform(sample)
        return sample

class RandomCropBoth_transform(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """
    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, sample):
        image_input, image_output = sample['input'], sample['output']
        if image_input.shape[:2] == image_output.shape[:2]:
            h, w   = image_input.shape[:2]
            new_h, new_w = self.size
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            img_input  = image_input[top: top + new_h, left: left + new_w]
            img_output = image_output[top: top + new_h, left: left + new_w]
        else:
            print('no random crop proceed, diff sizes')
            img_input  = image_input
            img_output = image_output
        return {'input': img_input, 'output': img_output}

class Rescale_transform(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, size_input, size_output):
        assert isinstance(size_input, (int, tuple))
        assert isinstance(size_output, (int, tuple))
        self.size_input = size_input
        self.size_output = size_output

    def __call__(self, sample):
        image_input, image_output  = sample['input'], sample['output']
        h, w = image_input.shape[:2]
        if isinstance(self.size_input, int):
            if h > w:
                new_h, new_w = self.size_input * h / w, self.size_input
            else:
                new_h, new_w = self.size_input, self.size_input * w / h

        else:
            new_h, new_w = self.size_input
        new_h, new_w = int(new_h), int(new_w)
        img_input = transform.resize(image_input, (new_h, new_w))
        # io.imshow(img_input)
        # io.show()
        h, w = image_output.shape[:2]
        if isinstance(self.size_output, int):
            if h > w:
                new_h, new_w = self.size_output * h / w, self.size_output
            else:
                new_h, new_w = self.size_output, self.size_output * w / h
        else:
            new_h, new_w = self.size_output
        new_h, new_w = int(new_h), int(new_w)
        img_output = transform.resize(image_output, (new_h, new_w))
        # io.imshow(img_output)
        # io.show()
        return {'input': img_input, 'output': img_output}

class ToTensor_transform(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image_input, image_output = sample['input'], sample['output']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img_input = torch.from_numpy(image_input.transpose((2, 0, 1)))
        img_output = torch.from_numpy(image_output.transpose((2, 0, 1)))
        return {'input': img_input, 'output': img_output}


def padd_input_to_output_size(input_tensor, input_img_size=settings.input_img_size, output_img_size=settings.output_img_size, printing=False):
    size_diff = int(output_img_size - input_img_size)
    if size_diff % 2 == 0:
        paddler = [int(size_diff / 2), int(size_diff / 2), int(size_diff / 2), int(size_diff / 2)]
    else:
        paddler = [int(size_diff / 2), int(size_diff / 2), int(size_diff / 2) + 1, int(size_diff / 2) + 1]
    if printing: print('padding L,R,U,D: ' + str(paddler))
    padd_input_img  = NN_Func.pad(input_tensor, paddler)# , mode='replicate')
    return padd_input_img


# size_diff = int((settings.output_img_size - settings.input_img_size) / 2)
# padder = nn.ReplicationPad2d(size_diff)

# def padd_input_to_output_size(sample, printing=False):
#     input_img = sample['input'].unsqueeze(1)
#     if printing: print('input size: ' + str(input_img.size()))
#     new_input_img = padder(input_img)
#     if printing: print('output size:' + str(new_input_img.size()))
#     sample['input'] = new_input_img
#     return sample
