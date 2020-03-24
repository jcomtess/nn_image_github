# requirements.txt
from __future__ import print_function, division
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torchvision import transforms
# from torchsummary import summary

import dataset_class
from models.nn_Conv_Linear import ConvNN
from models.nn_Coder_Linear import CoderNN
from models.nn_Decoder_Linear import DecoderNN
import settings
import misc

# weights and biases
# import wandb
# wandb.init(project="image_enchancer_Convolution")

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
nn_Coder_layer     = settings.nn_total_layers - settings.nn_conv_layers
# исходный размер поделённые на 2 столько раз, сколько сверточных слоев в модели
convolved_size      = int(settings.input_img_size / (settings.nn_conv_layers * 2))
print('total layers: ' + str(settings.nn_total_layers) + ' Convol: '
      + str(settings.nn_conv_layers) + ' Coder: ' + str(nn_Coder_layer))
print('input size:   ' + str(settings.input_img_size) + ' Convolved size: ' + str(convolved_size))

NNConv              = ConvNN(n_layer=settings.nn_conv_layers, input_size=settings.input_img_size, printing=True)
NNcoder             = CoderNN(n_layer=nn_Coder_layer, input_size=convolved_size, printing=True)
NNdecoder           = DecoderNN(n_layer=settings.nn_total_layers, output_size=settings.output_img_size, printing=True)
# print(summary(NNConv, (3, settings.input_img_size, settings.input_img_size), settings.batch_size, device='cpu'))
# print(summary(NNcoder, (3, convolved_size, convolved_size), settings.batch_size, device='cpu'))
# print(summary(NNdecoder, (3, settings.input_img_size, settings.input_img_size), settings.batch_size, device='cpu'))
# Создание Loss-функции
loss_func = nn.MSELoss(reduction='sum')
optimizer = optim.SGD([{'params': NNConv.parameters(), 'lr': settings.learning_rate, 'momentum': settings.momentum},
                      {'params': NNcoder.parameters(), 'lr': settings.learning_rate, 'momentum': settings.momentum},
                      {'params': NNdecoder.parameters(), 'lr': settings.learning_rate, 'momentum': settings.momentum}])
epoch_Cnv = misc.load_model_from_state_file(model=NNConv, optim=optimizer)
epoch_Cd  = misc.load_model_from_state_file(model=NNcoder, optim=optimizer)
epoch_Dd  = misc.load_model_from_state_file(model=NNdecoder, optim=optimizer)


#%%
# Общий для Кодера и Декодера цикл обратной прогонки (обучение градиентом)
def coder_decoder_backward_run(predicted_img, input_img):
    calc_loss = loss_func(predicted_img, input_img)
    optimizer.zero_grad()
    calc_loss.backward()
    optimizer.step()
    return calc_loss


#%%
# Начинаем циклы обучения нашей модели
# wandb.watch(NNmodel)
for n_data in range(settings.iteration_over_data):
    for sample_batched in dataloader:
        if sample_batched['input'].size()[0] != settings.batch_size:
            break  # Если в датасете не хватает объектов для полного Батча - прервать цикл
        epoch_Cnv += 1
        epoch_Cd  += 1
        epoch_Dd  += 1
        pred_conv = NNConv.forward(sample_batched['input'])
        pred_vars = NNcoder.forward(pred_conv)
        pred_img  = NNdecoder.forward(pred_vars)
        if epoch_Cd % settings.show_n_epoch == 0 or epoch_Dd % settings.show_n_epoch == 0:
            misc.show_batch(pred_img, sample_batched, printing=False)
        loss = coder_decoder_backward_run(pred_img, sample_batched['input'])
        print('epoch= ' + str(epoch_Cd) + '   loss = ' + str(int(loss.item())))
        # wandb.log({"Loss": loss})
        if epoch_Cd % settings.save_n_epoch == 0 or epoch_Dd % settings.save_n_epoch == 0:
            misc.save_model_to_file(model=NNConv, optim=optimizer, n_epoch=epoch_Cnv)
            misc.save_model_to_file(model=NNcoder, optim=optimizer, n_epoch=epoch_Cd)  # loss_list=loss_list)
            misc.save_model_to_file(model=NNdecoder, optim=optimizer, n_epoch=epoch_Dd)
            print('saved model at epoch ' + str(epoch_Dd))
