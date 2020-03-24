from torch import nn, optim, device
import settings, misc

# Класс десятислойной полносвязанной нейросетки,
class Enchancer_model(nn.Module):
    def __init__(self, printing=False, dropout_var=settings.dropout_var):
        super(Enchancer_model, self).__init__()
        self.model_name  = 'Linear10L'
        self.printing    = printing                   # Verbose
        self.dropout_var = dropout_var               # Вероятность DropOut Нейронов
        self.n_channels  = 3
        self.input_size  = settings.input_img_size * settings.input_img_size      # Количество входных нейронов (пикселей изобр)
        self.output_size = settings.output_img_size * settings.output_img_size    # Количество выходящих Н-ов (пикселей)
        self.middle_size = self.output_size + misc.mean_list([self.input_size,    # Кол-во Нейронов средних слоев
                                                            self.output_size])    # N_сред = N_выход + (N_вход + N_выход)/2
        if self.printing: print('Input: ' + str(self.input_size) +              # 1ый слой = N_вход
                                ' Middl: ' + str(self.middle_size) +            # 2-4 слой = N_сред
                                ' Out: ' + str(self.output_size))               # 5-10 слой = N_выход
        # self.cuda = device('cuda')
        # self.cuda0 = device('cuda:0')
        # self.cuda2 = device('cuda:2')  # GPU 2 (these are 0-indexed)
        self.layer_creation()


    def layer_creation(self):
        self.lay_01 = nn.Sequential(
            nn.Linear(self.input_size, self.middle_size).double(),
            nn.Tanh().double(),
            nn.Dropout(self.dropout_var))
        self.lay_02 = nn.Sequential(
            nn.LayerNorm(self.middle_size, elementwise_affine=True).double(),
            nn.Linear(self.middle_size, self.middle_size).double(),
            nn.Tanh().double(),
            nn.Dropout(self.dropout_var))
        self.lay_03 = nn.Sequential(
            nn.LayerNorm(self.middle_size, elementwise_affine=True).double(),
            nn.Linear(self.middle_size, self.middle_size).double(),
            nn.Tanh().double(),
            nn.Dropout(self.dropout_var))
        self.lay_04 = nn.Sequential(
            nn.LayerNorm(self.middle_size, elementwise_affine=True).double(),
            nn.Linear(self.middle_size, self.output_size).double(),
            nn.Tanh().double(),
            nn.Dropout(self.dropout_var))
        self.lay_05 = nn.Sequential(
            nn.LayerNorm(self.output_size, elementwise_affine=True).double(),
            nn.Linear(self.output_size, self.output_size).double(),
            nn.Tanh().double(),
            nn.Dropout(self.dropout_var))
        self.lay_06 = nn.Sequential(
            nn.LayerNorm(self.output_size, elementwise_affine=True).double(),
            nn.Linear(self.output_size, self.output_size).double(),
            nn.Tanh().double(),
            nn.Dropout(self.dropout_var))
        self.lay_07 = nn.Sequential(
            nn.LayerNorm(self.output_size, elementwise_affine=True).double(),
            nn.Linear(self.output_size, self.output_size).double(),
            nn.Tanh().double(),
            nn.Dropout(self.dropout_var))
        self.lay_08 = nn.Sequential(
            nn.LayerNorm(self.output_size, elementwise_affine=True).double(),
            nn.Linear(self.output_size, self.output_size).double(),
            nn.Tanh().double())
        self.lay_09 = nn.Sequential(
            nn.LayerNorm(self.output_size, elementwise_affine=True).double(),
            nn.Linear(self.output_size, self.output_size).double(),
            nn.Tanh().double())
        self.lay_10 = nn.Sequential(
            nn.LayerNorm(self.output_size, elementwise_affine=True).double(),
            nn.Linear(self.output_size, self.output_size).double(),
            nn.Tanh().double(),
            nn.LayerNorm(self.output_size, elementwise_affine=True).double())
        if self.printing: print('layer creation done')

    def forward(self, input):
        # out = input.to(device=self.cuda)
        out_gpu = input.reshape(settings.batch_size, 3, self.input_size)    # Уменьшаем размерность изображения
        out_gpu = self.lay_01(out_gpu)                                          # B_size/Channel/Width/Height
        out_gpu = self.lay_02(out_gpu)                                          # B_size/Channel/Pixels
        out_gpu = self.lay_03(out_gpu)                                          #
        out_gpu = self.lay_04(out_gpu)                                          #
        out_gpu = self.lay_05(out_gpu)                                          #
        out_gpu = self.lay_06(out_gpu)
        out_gpu = self.lay_07(out_gpu)
        out_gpu = self.lay_08(out_gpu)
        out_gpu = self.lay_09(out_gpu)
        out_gpu = self.lay_10(out_gpu)
        # out = out_gpu.to(device=self.cuda2)
        output = out_gpu.reshape(settings.batch_size,                       # Восстанавливаем 2х мерную размерность изображения
                             3,
                             settings.output_img_size,
                             settings.output_img_size)
        return output

    def backward_run(self, predicted, solution, loss_function, optimizer):
        loss = loss_function(predicted, solution)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss