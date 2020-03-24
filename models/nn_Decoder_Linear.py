from torch import nn
import settings


#%%
class DecoderNN(nn.Module):
    def __init__(self, n_layer, output_size=settings.input_img_size, printing=False, dropout_var=settings.dropout_var):
        super(DecoderNN, self).__init__()
        self.n_layer     = n_layer
        self.printing    = printing                   # Verbose
        self.dropout_var = dropout_var               # Вероятность DropOut Нейронов
        self.input_vars  = settings.n_vars   # Количество входных нейронов (пикселей изобр)
        self.output_size = output_size
        self.output_pixels = output_size * output_size
        self.coder_step_size = int(round((self.output_pixels - self.input_vars) / self.n_layer))
        self.n_channels  = 3
        self.model_name = 'Decoder_' + str(self.n_layer) + 'lay_' + str(self.input_vars) + 'in_'\
                          + str(self.output_size) + 'out_' + str(settings.batch_size) + 'bat_' + settings.optim_set
        self.layer_dict = {}
        self.layer_creation()
        if self.check_sizes():
            print('++++++++++++++++++++++++++')

    def layer_creation(self):
        for layer in range(self.n_layer):
            if layer == 0:    # . Если первый слой. Нет нормализации
                self.layer_dict[layer] = nn.Sequential(
                    nn.Linear(self.input_vars,
                              self.input_vars + self.coder_step_size * (layer + 1)).double(),
                    nn.Tanh().double(),
                    nn.Dropout(self.dropout_var))
            elif layer == self.n_layer - 1:    # . Если последний слой. Нет дропаута
                self.layer_dict[layer] = nn.Sequential(
                    nn.LayerNorm(self.input_vars + self.coder_step_size * layer, elementwise_affine=True).double(),
                    nn.Linear(self.input_vars + self.coder_step_size * layer,
                              self.output_pixels).double(),
                    nn.Tanh().double(),
                    nn.Dropout(self.dropout_var))
            else:   # . Если любой другой
                self.layer_dict[layer] = nn.Sequential(
                    nn.LayerNorm(self.input_vars + self.coder_step_size * layer, elementwise_affine=True).double(),
                    nn.Linear(self.input_vars + self.coder_step_size * layer,
                              self.input_vars + self.coder_step_size * (layer + 1)).double(),
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
        out = input_batch
        for layer in range(self.n_layer):
            out = self.layer_dict[layer](out)
        output = out.reshape(settings.batch_size, 3, self.output_size, self.output_size)
        return output
