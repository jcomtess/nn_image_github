from torch import nn
import settings


#%%
class CoderNN(nn.Module):
    def __init__(self, n_layer, input_size=settings.input_img_size, n_channel=3, output_vars=settings.n_vars,
                 printing=False, dropout_var=settings.dropout_var):
        super(CoderNN, self).__init__()
        self.n_layer = n_layer
        self.printing    = printing                  # Verbose
        self.dropout_var = dropout_var               # Вероятность DropOut Нейронов
        self.input_size  = input_size                # Размер (ширина или высота, одно число) изображениия
        self.input_pixels = input_size * input_size  # Количество входных нейронов (пикселей изобр)
        self.output_vars = output_vars
        self.coder_step_size = int(round((self.input_pixels - self.output_vars) / self.n_layer))
        self.n_channels  = n_channel
        self.model_name = 'Coder' + str(self.n_layer) + 'lay_' + str(self.input_size) + 'in_'\
                          + str(self.output_vars) + 'var_' + str(settings.batch_size) + 'bat_' + settings.optim_set
        self.layer_dict = {}
        self.layer_creation()
        if self.check_sizes():
            print('++++++++++++++++++++++++++')

    def layer_creation(self):
        for layer in range(self.n_layer):
            if layer == 0:                   # . Если первый слой. Нет нормализации
                self.layer_dict[layer] = nn.Sequential(
                    nn.Linear(self.input_pixels,
                              self.input_pixels - self.coder_step_size * (layer + 1)).double(),
                    nn.Tanh().double(),
                    nn.Dropout(self.dropout_var))
            elif layer == self.n_layer - 1:      # . Если последний слой. Нет дропаута.  -1 т.к. range считает до числа
                self.layer_dict[layer] = nn.Sequential(
                    nn.LayerNorm(self.input_pixels - self.coder_step_size * layer, elementwise_affine=True).double(),
                    nn.Linear(self.input_pixels - self.coder_step_size * layer,
                              self.output_vars).double(),
                    nn.Tanh().double())
            else:                            # . Если любой другой
                self.layer_dict[layer] = nn.Sequential(
                    nn.LayerNorm(self.input_pixels - self.coder_step_size * layer, elementwise_affine=True).double(),
                    nn.Linear(self.input_pixels - self.coder_step_size * layer,
                              self.input_pixels - self.coder_step_size * (layer + 1)).double(),
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
        out = input_batch.reshape(settings.batch_size, 3, self.input_pixels)
        for layer in range(self.n_layer):
            out = self.layer_dict[layer](out)
        return out
