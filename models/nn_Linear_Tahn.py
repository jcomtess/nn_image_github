from torch import nn
import settings
import misc


#%%
class LinearNN(nn.Module):
    def __init__(self, n_layer, printing=False, dropout_var=settings.dropout_var):
        super(LinearNN, self).__init__()
        self.n_layer     = n_layer
        self.printing    = printing                   # Verbose
        self.dropout_var = dropout_var               # Вероятность DropOut Нейронов
        self.n_channels  = 3
        self.input_size  = settings.input_img_size * settings.input_img_size    # Количество входных нейронов, пикселей
        self.output_size = settings.output_img_size * settings.output_img_size  # Количество выходящих Н-ов (пикселей)
        self.middle_size = self.output_size + misc.mean_list([self.input_size,  # Кол-во Нейронов средних слоев
                                                            self.output_size])  # N_сред = N_выход+(N_вход+N_выход)/2
        self.size_step   = int(round((self.output_size - self.input_size) / self.n_layer))
        if self.printing: print('Input: ' + str(self.input_size) +              # 1ый слой = N_вход
                                ' Middl: ' + str(self.middle_size) +            # 2-3 слой = N_сред
                                ' Out: ' + str(self.output_size))               # 4-8 слой = N_выход
        self.model_name = 'Linear' + str(self.n_layer) + 'lay_' + str(self.input_size) + 'in_'\
                          + str(self.output_size) + 'out_' + str(settings.batch_size) + 'bat_' + settings.optim_set
        self.layer_dict = {}
        self.layer_creation()

    def layer_creation(self):
        if self.printing: print('creating layer for model:')
        if self.printing: print(self.model_name)
        for layer in range(self.n_layer):
            if layer == 0:
                self.layer_dict[layer] = nn.Sequential(
                    nn.Linear(self.input_size + self.size_step * layer,
                              self.input_size + self.size_step * (layer + 1)).double(),
                    nn.Tanh().double(),
                    nn.Dropout(self.dropout_var))
            elif layer == self.n_layer:
                self.layer_dict[layer] = nn.Sequential(
                    nn.LayerNorm(self.input_size + self.size_step * layer, elementwise_affine=True).double(),
                    nn.Linear(self.input_size + self.size_step * layer,
                              self.output_size).double(),
                    nn.Tanh().double(),
                    nn.Dropout(self.dropout_var))
            else:
                self.layer_dict[layer] = nn.Sequential(
                    nn.LayerNorm(self.input_size + self.size_step * layer, elementwise_affine=True).double(),
                    nn.Linear(self.input_size + self.size_step * layer,
                              self.input_size + self.size_step * (layer + 1)).double(),
                    nn.Tanh().double(),
                    nn.Dropout(self.dropout_var))
            if self.printing: print('Linear layer N ' + str(layer +1) + ' with '
                                    + str(self.input_size + self.size_step * layer) + ' input pixel')
        if self.printing: print('layer creation done')

    def forward(self, input_batch):
        out = input_batch.reshape(settings.batch_size, 3, self.input_size)    # Уменьшаем размерность изображения
        for layer in self.layer_dict.items():
            out = layer(out)
        output = out.reshape(settings.batch_size,                       # Восстанавливаем 2х мерную размерность изображения
                             3,
                             settings.output_img_size,
                             settings.output_img_size)
        return output
