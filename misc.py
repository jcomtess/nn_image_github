from __future__ import print_function, division
import os
import glob
import torch
from torch import load as nn_load
from torch import save as nn_save
from torchvision import utils
from skimage import io

# import wandb
import dataset_class
import settings
import warnings
warnings.filterwarnings("ignore")

#%%
# Разные переменные которым не нашлось места в других файлах
HIDDEN = True  # Отключить!
if HIDDEN:
    train_image_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                    'data', '.xxx', 'train_img', '*')  # Hidden train, Caution
else:
    train_image_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                    'data', 'train_img', '*')          # Normal train
train_img_path_list  = glob.glob(train_image_path)


# Разные функции которым не нашлось места в других файлах
#%%
# Отобразить образец batch-а в плоте matplotlib
def show_batch(pred, sample_batched, printing=False):
    if printing:
        print('input sample size : ' + str(sample_batched['input'][0].size())[11:-1])
        print('output sample size: ' + str(sample_batched['output'][0].size())[11:-1])
        print('predicted size    : ' + str(pred[0].size())[11:-1])
    image_list = []
    for n_image in range(settings.batch_size):
        image_list.append(dataset_class.padd_input_to_output_size(sample_batched['input'][n_image]))
        image_list.append(sample_batched['output'][n_image])
        image_list.append(pred[n_image])
    if settings.batch_size % 15 == 0:
        n_row = int(settings.batch_size / 5)
    elif settings.batch_size % 12 == 0:
        n_row = int(settings.batch_size / 4)
    elif settings.batch_size % 9 == 0:
        n_row = int(settings.batch_size / 3)
    elif settings.batch_size % 6 == 0:
        n_row = int(settings.batch_size / 2)
    elif settings.batch_size % 3 == 0:
        n_row = settings.batch_size
    else:
        n_row = 3
    grid = utils.make_grid(image_list, nrow=n_row)
    io.imshow(grid.detach().numpy().transpose((1, 2, 0)))
    io.show()
    pass


#%%
# Загрузить модель из файла, необходимо также передать саму модель и оптимизатор
def load_model_from_state_file(model, optim):
    if HIDDEN:
        state_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                  'data', '.xxx', 'saved_model', model.model_name)
    else:
        state_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                  'data', 'saved_model', model.model_name)
    if os.path.isfile(state_path):
        saved_model = nn_load(state_path)
        model.load_state_dict(saved_model['model'])
        optim.load_state_dict(saved_model['optim'])
        epoch = saved_model['epoch']
        model.eval()
        print('success read saved model')
    else:
        epoch = 0
        print('no saved model found! creating new')
    return epoch


#%%
# Сохранить модель в файл с уникальным именем, необходимо также передать саму модель и оптимизатор
def save_model_to_file(model, optim, n_epoch):
    model_name = model.model_name
    if HIDDEN:
        state_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                  'data', '.xxx', 'saved_model', model_name)
    else:
        state_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                  'data', 'saved_model', model_name)
    nn_save({'epoch': n_epoch, 'model': model.state_dict(), 'optim': optim.state_dict()}, state_path)


#%%
# Высчитывает среднее арифметическое из любого списка
def mean_list(any_list):
    summ = 0
    n_val = 0
    for n_val, val in enumerate(any_list):
        summ += val
    n_val += 1
    mean = int(summ / n_val)
    return mean


# Считает вых размера для свёрточных сетей. Не уверен одинаково верно ли для Conv2d и MaxPool2d
# https://algorithmia.com/blog/convolutional-neural-nets-in-pytorch
def calc_output_pixels(input_size, kernel_size, stride, padding):
    output = int((input_size - kernel_size + 2 * padding) / stride) + 1
    return output


#%%
# workaround
def read_img_size(img_path):
    img = io.imread(img_path)
    sizes_list = list(torch.from_numpy(img.transpose((2, 0, 1))).size())
    return sizes_list


#%%
# Определяет ориентацию изображения
def calc_orientation(img_path):
    channel, width, height = read_img_size(img_path)
    if width > int(height * 1.6):
        orient = 'landscape'
    elif width > int(height * 1.3):
        orient = 'album'
    elif height > int(width * 1.3):
        orient = 'portrait'
    elif height > int(width * 1.6):
        orient = 'scroll'
    else:
        orient = 'square'
    return orient


#%%
# Фильтр изображений по заданному миним. размеру
def filter_dataset_by_img_min_size(size, img_path_list):
    for img_path in img_path_list:
        c, w, h = read_img_size(img_path)
        if c > 3:
            print('chanel number error ' + str(img_path))
        if w < h:
            min_side = w
        else:
            min_side = h
        if size > min_side:
            img_path_list.remove(img_path)
    return img_path_list
