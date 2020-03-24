import glob
from skimage import io
import os, glob, csv
import torch
import json
import random
import settings, misc


#%%
# Читает набор тегов из .csv файла
def read_csv(proj_path=os.path.abspath(os.path.dirname(__file__))):
    csv_file = open(os.path.join(proj_path, 'data', 'img_tags.csv'))
    csv_reader = csv.reader(csv_file, dialect='csv')

    csv_file.close()

#%%
# Возвращает кол-во изобр. минимальный размер которых больше или равен аргументу
def img_min_size_numbering(size, printing=False):
    N = 0
    for n_img, img_path in enumerate(misc.train_img_path_list):
        c, w, h = misc.read_img_size(img_path)
        if c > 3: print('chanells error ' + str(img_path))
        if w < h: min_side = w
        else: min_side = h
        if size <= min_side: N += 1
    if printing: print(str(N) + ' images bigger then ' + str(size) + ' pixels')
    return N

#%%
# Распределяет изображения по группам размерам
def group_img_by_min_size(printing=False):
    size_dict         = {'100':0,'200':0,'300':0,'400':0,'500':0,'600':0,'700':0,'800':0,'1000':0}
    for n_img, img_path in enumerate(misc.train_img_path_list):
        c, w, h = misc.read_img_size(img_path)
        if w < h: min_side = w
        else: min_side = h
        for keys in size_dict.keys():
            if min_side <= int(keys):
                size_dict[keys] += 1
        orient = misc.calc_orientation(img_path)
        if printing: print('IMG_' + str(n_img) + ' W_' + str(w) + ' H_' + str(h) + ' orient: ' + orient)
    print(size_dict)

#%%

if __name__=='__main__':
    print(misc.train_img_path_list)
    group_img_by_min_size(printing=True)
    Cumul_list = []
    for x in range(0, 1100, 50):
        n = img_min_size_numbering(x, printing=True)
        Cumul_list.append(n)
    print(Cumul_list)
