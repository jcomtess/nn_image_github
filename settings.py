import os
from torch import nn, optim
Name = 'NeuroImage'
wandb_key = '98d89cb3a608069e728ad7f3b7389fce4b6f057a'

# size_dict = {'small':0, 'medium':1, 'large':2, 'wallpaper':3}
# training settings
nn_total_layers  = 8
nn_conv_layers   = 3
number_of_worker = 1
learning_rate    = 0.000001
momentum         = 0.5
dropout_var      = 0.005
iteration_over_data = 1000
show_n_epoch        = 20
save_n_epoch        = 10

#%%
# Различные варианты обучения модели Loss-функция
loss_set = 'MSELoss'              # Mean Square Error Loss
# loss_set = 'L1Loss'             # Mean Absolute Error
# loss_set = 'SmoothL1Loss'       # Smooth L1 Loss, Huber loss
# loss_set = 'NLLLoss'            # Negative Log-Likelihood Loss
# loss_set = 'CrossEntropyLoss'   # Cross-Entropy Loss
# Оптимизатор
optim_set = 'SGD'
# optim_set = 'RMSprop'

#%%
# Кодер\Декодер модели
input_img_size   = 160       # если используется конволюция, то должен делиться на 2 без остатка N раз
n_vars           = 20        # где N - количество сверточных/конволюционных слоев
batch_size       = 12
output_img_size  = input_img_size   # на самом деле не используется в этом блоке, только ниже)))

#%%
# Линейные модели улучшения
# High RAM-memory 57 600
#input_img_size   = 60    # должен делиться на 4 без остатка. так надо для конволюции
#output_img_size  = 80
#batch_size       = 12    # должен делиться на 3 без остатка, так надо для вывода образцовых батчей
# Low RAM-memory 18 432
# input_img_size   = 32    # должен делиться на 4 без остатка. так надо для конволюции
# output_img_size  = 48
# batch_size       = 12     # должен делиться на 3 без остатка, так надо для вывода образцовых батчей
# Tiny RAM-memory 10 368
# input_img_size   = 24    # должен делиться на 4 без остатка. так надо для конволюции
# output_img_size  = 36
# batch_size       = 12     # должен делиться на 3 без остатка, так надо для вывода образцовых батчей













#%%
#
small = (800, 600)
medium = (1600, 1200)

#%%
# Tags для простых изображений 50х50
tglist_shape_color = ['black', 'red', 'green', 'blue', 'yellow', 'purple', 'cyan']
tglist_shape_type  = ['star', 'grid', 'diag_grid', 'ball', 'hex']
                     #['vertical_left', 'vertical_right', 'horizontal_up', 'horizontal_down',
                     # 'diagonal_1', 'diagonal_2', 'diagonal_3', 'diagonal_4']





















#%%
# list of tag class
# тип изображения
tglist_img_type   = ['foto', 'gif', 'draw', '3d_render', 'hentai', 'simple_shape']
# заполнение изображения
tglist_img_fill   = ['face', 'upper_body', 'down_body', 'whole_body', 'multiple_body', 'body_n_surroundings']
# ориентация изображ
tglist_img_orient = ['scroll', 'portrait', 'square', 'album', 'landscape']
# размер изображения
tglist_img_size   = ['small', 'medium', 'large', 'wallpaper', 'ultra_large']
# качество изображения
tglist_img_dpi    = ['small', 'medium', 'large']
#%%
# tgdict_img_type
# tgdict_img_fill
# tgdict_img_orient = {el:0 for el in tglist_img_orient}
# tgdict_img_size
# tgdict_img_dpi
#%%

# внешний вид актёров, цвет волос, макияж, одежда, прочее
# тип волос = длинные, короткие, прямые, вьющиеся, кудрявые, сложная _прическа, отсутствуютп
tglist_hair_type         = ['long', 'short', 'straight', 'waving', 'curly', 'haircut', 'none']
# цвет волос = брюнетка, русая, блонди, рыжая/красная, другой_цвет, нет_волос
tglist_hair_color        = ['brunette', 'brown', 'blonde', 'redhair', 'other', 'nohair']
# макияж_общий = нету, легкий, средний, тяжелый
tglist_makeup_total      = ['none', 'light', 'medium', 'heavy']
# макияж глаз (н,л,с,т) и цвет макияжа = нету, натуральный, черный
tglist_eye_makeup_amount = ['none', 'light', 'medium', 'heavy']
tglist_eye_makeup_color  = ['none', 'natural', 'black']
# цвет зрачков глаз = карие, голубые, зеленые, особые
tglist_eye_ocular_color  = ['brown', 'blue', 'green', 'specific']
# цвет губной помады
tglist_lips_makeup_color = ['none', 'red', 'pink', 'dark']
# размер губ 
tglist_lips_size         = ['thin', 'middle', 'fat']
# загар/цвет_кожи = бледный, средний, загорелый, метис, негритян, азиатка
tglist_skin_color_tan    = ['white', 'middle', 'tanned', 'black', 'asian']
# тип одежды = нету, обычная, обтягивающее, блестящее, латекс, чулки, косплей_персонажа
tglist_dress_type        = ['none', 'casual', 'tight', 'shiny', 'latex', 'stockings', 'pantyhose', 'cosplay']
# цвет одежды = нету, чёрная, белая, красная/яркая, прочие_цвета, радуга
tglist_dress_color       = ['none', 'black', 'white', 'red', 'other', 'rainbow']

# тип совокупления = классика, лесбиянки, трансы, геи, бисекс, стриптиз
tglist_sex_type          = ['straight', 'lesbian', 'tranny', 'gay', 'bisex', 'tease']
# позы совокупления = миссионер, наездница, раком, трутца, сидеть_лицо, отсос, дрочка, поцелуи, страпон_Ж+Ж, страпон_Ж+М, секс_игрушки
tglist_sex_pose_type     = ['missionere', 'riding', 'behind', 'grinding', 'facesitting', 'blowjob', 'handjob', 'kissing', 'strapon', 'pegging', 'toys']
# фетиши  = чулки, удушение, масло, красота, сперма на лице, доминирование
tglist_fetish            = ['stockings', 'strangle', 'oily', 'beauty', 'bukkake', 'femdom']

#позы фотомодели
pose_of_model            = ['standing', 'sit_on_chair', 'sit_on_squat', 'kneels', 'lay_on_stomach', 'lay_on_back']

# ярковыраженный фетиш

#stockings
#%%

import misc
