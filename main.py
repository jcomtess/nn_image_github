# requirements.txt
from __future__ import print_function, division
import glob
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms, utils


import dataset_class
from models.nn_Linear_Tahn import LinearNN
import settings
import misc

# weights and biases
# import wandb
# wandb.init(project="image_enchancer_Linear")

# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

#%%
# Создаем модель по свойствам из settings и misc
train_image_path_list = misc.filter_dataset_by_img_min_size(settings.input_img_size, misc.train_img_path_list)
n_images_in_dataset   = len(train_image_path_list)
print('Total number of images in dataset = ' + str(n_images_in_dataset))
crop                  = dataset_class.RandomCropBoth_transform(settings.output_img_size)
scale                 = dataset_class.Rescale_transform(settings.input_img_size, settings.output_img_size)
tensr                 = dataset_class.ToTensor_transform()
composed_transform    = transforms.Compose([crop, scale, tensr])
dataset               = dataset_class.Enchancer_Dataset(img_path_list=train_image_path_list,
                                                        transform=composed_transform)
dataloader            = DataLoader(dataset, batch_size=settings.batch_size,
                                   shuffle=True, num_workers=settings.number_of_worker)
NNmodel               = LinearNN(printing=True, dropout_var=settings.dropout_var)

# Создание Loss-функции из файла настроек settings
if settings.loss_set == 'L1Loss':
    loss_func = nn.L1Loss(reduction='sum')                        # Mean Absolute Error
elif settings.loss_set == 'MSELoss':
    loss_func = nn.MSELoss(reduction='sum')                    # Mean Square Error Loss
elif settings.loss_set == 'SmoothL1Loss':
    loss_func = nn.SmoothL1Loss(reduction='sum')          # Smooth L1 Loss, Huber loss
elif settings.loss_set == 'NLLLoss':
    loss_func = nn.NLLLoss(reduction='sum')                    # Negative Log-Likehood loss
elif settings.loss_set == 'CrossEntropyLoss':
    loss_func = nn.CrossEntropyLoss(reduction='sum')  # Cross-Entropy Loss
else:
    print('wrong Loss-function type, exiting')
    exit()
# Аналогично - оптимизатор из settings
if settings.optim_set == 'SGD':
    optimizer = optim.SGD(NNmodel.parameters(), lr=settings.learning_rate,
                          momentum=settings.momentum, nesterov=False)
elif settings.optim_set == 'RMSprop':
    optimizer = optim.RMSprop(NNmodel.parameters(), lr=settings.learning_rate,
                              alpha=0.99, eps=1e-08, weight_decay=0, momentum=settings.momentum, centered=False)
else:
    print('Wrong optimizer type, exiting')
    exit()
epoch = misc.load_model_from_state_file(model=NNmodel, optim=optimizer)

#%%
# Начинаем циклы обучения нашей модели
# wandb.watch(NNmodel)
for n_data in range(settings.iteration_over_data):
    for sample_batched in dataloader:
        if sample_batched['input'].size()[0] != settings.batch_size:
            break  # Если в датасете не хватает объектов для полного Батча - прервать цикл
        epoch += 1
        pred = NNmodel.forward(sample_batched['input'])
        if epoch % settings.show_n_epoch == 0:
            misc.show_batch(pred, sample_batched, printing=True)  # . Вывести вход/выход/прогноз
        loss = NNmodel.backward_run(predicted=pred, solution=sample_batched['output'],
                                    loss_function=loss_func, optimizer=optimizer)
        # loss_list['loss'].append(loss.item())
        print('epoch= ' + str(epoch) + '   loss = ' + str(int(loss.item())))
        # wandb.log({"Loss": loss})
        if epoch % settings.save_n_epoch == 0:
            misc.save_model_to_file(model=NNmodel, optim=optimizer, n_epoch=epoch)
            print('saved model at epoch ' + str(epoch))




