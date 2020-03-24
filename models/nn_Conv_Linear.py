from torch import nn
import settings
import misc
#%%
# Просто чтоб не забыть какие использую паоаметры
Conv_params = '''
kernel_size = 3
stride = 1
padding = 2
dilation = 1
groups = 1
'''
MaxPool_params = '''
kernel_size = 2
stride = 2
padding = 0
dilation = 1
'''

#%%
class ConvNN(nn.Module):
    def __init__(self, n_layer, input_size=settings.input_img_size, n_channel=3, channel_step=9,
                 printing=False, dropout_var=settings.dropout_var):
        super(ConvNN, self).__init__()
        self.n_layer          = n_layer         # Каждый сверточный слой сокращает размерность изображение
        self.input_channels   = n_channel       # в 2 раза по w и по h, т.е. пикселей в 4 раза!!!
        self.channel_step     = channel_step    # но увеличивает кол-во каналов на такую величину
        self.printing         = printing        # Verbose
        self.dropout_var      = dropout_var
        self.input_size       = input_size      # Сторона изобрж.
        self.input_pixels     = input_size ^ 2  # Количество входных нейронов (пикселей изобр)
        self.output_pixels    = misc.calc_output_pixels(input_size=self.input_pixels,
                                                        kernel_size=3, stride=1, padding=2)
        self.output_chanel    = self.input_channels + self.channel_step * self.n_layer
        self.model_name       = 'Conv' + str(self.n_layer) + 'lay_' + str(self.input_size) + 'in_' \
                                + str(self.output_pixels) + 'out_' + str(self.output_chanel) + 'outCh_' \
                                + str(settings.batch_size) + 'bat_' + settings.optim_set

        print(self.model_name)
        print('---------------------------------------------------------------')
        print(self.output_pixels)
        print(self.output_chanel)
        self.layer_dict = {}
        self.layer_creation()
        if self.check_sizes():
            print('++++++++++++++++++++++++++')


#  Convolution info
#    Kernel Size – the size of the filter.
#    Kernel Type – the values of the actual filter. Some examples include identity, edge detection, and sharpen.
#    Stride – the rate at which the kernel passes over the input image.
#       A stride of 2 moves the kernel in 2-pixel increments.
#    Padding – we can add layers of 0s to the outside of the image in order to make sure
#       that the kernel properly passes over the edges of the image.
#    Output Layers – how many different kernels are applied to the image.

    def layer_creation(self):
        for layer in range(self.n_layer):
            if layer == 0:                   # . Если первый слой. Нет нормализации
                self.layer_dict[layer] = nn.Sequential(
                    nn.Conv2d(self.input_channels,
                              self.input_channels + self.channel_step * (layer + 1),
                              kernel_size=3, stride=1, padding=2, dilation=1, groups=1).double(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1).double(),
                    nn.Tanh().double(),
                    nn.Dropout(self.dropout_var))
            elif layer == self.n_layer - 1:      # . Если последний слой. Нет дропаута
                self.layer_dict[layer] = nn.Sequential(
                    nn.LayerNorm(self.input_channels + self.channel_step * layer, elementwise_affine=True).double(),
                    nn.Conv2d(self.input_channels + self.channel_step * layer,
                              self.output_chanel,
                              kernel_size=3, stride=1, padding=2, dilation=1, groups=1).double(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1).double(),
                    nn.Tanh().double())
            else:                           # . Если любой другой
                self.layer_dict[layer] = nn.Sequential(
                    nn.LayerNorm(self.input_channels + self.channel_step * layer, elementwise_affine=True).double(),
                    nn.Conv2d(self.input_channels + self.channel_step * layer,
                              self.input_channels + self.channel_step * (layer + 1),
                              kernel_size=3, stride=1, padding=2, dilation=1, groups=1).double(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1).double(),
                    nn.Tanh().double(),
                    nn.Dropout(self.dropout_var))

    def check_sizes(self):
        for layer in range(self.n_layer):
            if self.printing:
                print('check layer: ' + str(layer + 1) + ' of the model: ' + self.model_name)
                print(str(self.layer_dict[layer]))
            if settings.input_img_size % 2 != 0:
                print('at layer N ' + str(layer) + ' divided to 2')
                return False
        return True

    def forward(self, input_batch):
        out = input_batch.double()
        for layer in range(self.n_layer):
            out = self.layer_dict[layer](out)
            if self.printing: print('layer: ' +str(layer + 1) + ' out size: ' + str(out.size()))
        # output = out.reshape(settings.batch_size, self.n_input_channels + self.channel_step * self.n_layer)
        return out
