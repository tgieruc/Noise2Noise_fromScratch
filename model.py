import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

# ---- BLOCKS -----
class Module(object):
    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        return input

    def backward(self, gradwrtoutput):
        pass

    def get_params(self):
        return []

    def set_params(self, params):
        pass

    def cuda(self):
        pass

    def params(self):
        return []

    def eval(self):
        pass

    def train(self):
        pass



class ReLU(Module):

    def __init__(self):
        self.zero_mask = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.zero_mask = x > 0
        return x * self.zero_mask

    def backward(self, gradwrtoutput):
        result = self.zero_mask * gradwrtoutput
        return result



class LeakyRelu(Module):

    def __init__(self, slope):
        self.slope = slope
        self.zero_mask = None

    def forward(self, x):
        zero_mask = x > 0
        self.zero_mask = zero_mask.float()
        result = zero_mask * (1 - self.slope)
        result += self.slope
        return x * result

    def backward(self, gradwrtoutput):
        result = self.zero_mask * (1 - self.slope)
        result += self.slope 
        return result
    


class Sigmoid(Module):

    def __init__(self):
        self.__e = torch.e
        self.forward_sigm = None
        self.mode = "train"

    def __sig(self, x):
        return 1 / (1 + self.__e ** -x)

    def forward(self, x):
        sig = self.__sig(x)
        if self.mode == "train":
            self.forward_sigm = sig
        return sig

    def backward(self, gradwrtoutput):
        result = gradwrtoutput * self.forward_sigm * (1 - self.forward_sigm)
        return result
    


class MaxPool2d(Module):

    def __init__(self):
        self.index = None
        self.mask = None

    def forward(self, x):
        n, c, W, H = x.shape
        xmax = x.view(n, c, W // 2, 2, H // 2, 2).max(5).values.max(3)
        xmax_max = xmax.values.repeat_interleave(2, axis=2).repeat_interleave(2, axis=3)
        self.mask = xmax_max == x
        return xmax.values

    def backward(self, gradwrtoutput):
        result = self.mask * gradwrtoutput.repeat_interleave(2, axis=2).repeat_interleave(2, axis=3)
        return result    
    


class Conv2d(Module):

    def __init__(self, channels_in=None, channels_out=None, kernel_size=(3, 3), stride=1, padding=0, dilation=1,
                 bias=True):
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.is_bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.device = "cpu"
        self.mode = "train"
        self.input_unfolded = None
        self.input_shape = None

        # Xavier weight initialization
        xavier_bound = (2 / (channels_out + channels_in) ** 0.5)
        self.weight = torch.empty((self.channels_out, self.channels_in, self.kernel_size[0], self.kernel_size[1])).uniform_(-xavier_bound, xavier_bound).to(self.device)
        self.bias = torch.empty(self.channels_out).uniform_(-xavier_bound, xavier_bound).to(self.device)

        self.dB = self.bias.clone()
        self.dW = self.weight.clone()
        self.prev_update_bias = self.bias * 0
        self.prev_update_conv = self.weight * 0

    def forward(self, input):
        input = input.to(self.device)
        N, _, _, _ = input.shape
        self.input_unfolded = torch.nn.functional.unfold(input, kernel_size=self.kernel_size, padding=self.padding,
                                                         stride=self.stride, dilation=self.dilation).to(self.device)
        wxb = self.weight.view(self.channels_out, -1) @ self.input_unfolded + self.bias.view(1, -1, 1)
        result = wxb.view(N, self.channels_out,
                          ((input.shape[2] - self.dilation * self.kernel_size[
                              0] + 2 * self.padding) // self.stride + 1),
                          ((input.shape[3] - self.dilation * self.kernel_size[
                              1] + 2 * self.padding) // self.stride + 1))

        if self.mode == "train":
            self.input_shape = input.shape

        return result

    def backward(self, gradwrtoutput):
        N, C, H, W = self.input_shape

        # dW
        grad_reshaped = gradwrtoutput.permute(1, 2, 3, 0).reshape(self.channels_out, -1)
        dW = grad_reshaped @ self.input_unfolded.permute(1, 2, 0).reshape(self.input_unfolded.shape[1], -1).T
        self.dW *= 0
        self.dW += dW.reshape(self.weight.shape)

        # dB
        self.dB *= 0
        self.dB += gradwrtoutput.sum(axis=(0, 2, 3)).view(-1)

        # dX
        weight_reshaped = self.weight.view(self.channels_out, -1)
        dX_col = weight_reshaped.T @ grad_reshaped
        out_h = (H - self.dilation * (self.kernel_size[0] - 1) + 2 * self.padding - 1) // self.stride + 1
        out_w = (W - self.dilation * (self.kernel_size[1] - 1) + 2 * self.padding - 1) // self.stride + 1
        dX_col_reshaped = dX_col.view(C * self.kernel_size[0] * self.kernel_size[1], out_w * out_h, N)
        dX = torch.nn.functional.fold(dX_col_reshaped.permute(2, 0, 1),
                                      self.input_shape[2:], kernel_size=self.kernel_size, padding=self.padding,
                                      stride=self.stride, dilation=self.dilation)
        
        return dX
    
    def get_params(self):
        return [self.weight, self.bias, self.is_bias]

    def set_params(self, params):
        self.weight = params[0].to(self.device)
        self.bias = params[1].to(self.device)
        self.is_bias = params[2]

    def cuda(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weight = self.weight.to(self.device)
        self.bias = self.bias.to(self.device)
        self.dW = self.dW.to(self.device)
        self.dB = self.dB.to(self.device)
        self.prev_update_conv = self.prev_update_conv.to(self.device)
        self.prev_update_bias = self.prev_update_bias.to(self.device)

    def params(self):
        return [[self.weight, self.dW], [self.bias, self.dB]]



class UpSampling2D(Module):
    """ Nearest neighbor up sampling of the input. Repeats the rows and columns of the data by size[0] and size[1] respectively. Parameters: ----------- size: tuple (size_y, size_x) - The number of times each axis will be repeated. """

    def __init__(self, scale_factor=2):
        self.scale_factor = scale_factor
        self.device = "cpu"

    def forward(self, x):
        return x.repeat_interleave(self.scale_factor, dim=2).repeat_interleave(self.scale_factor, dim=3)

    def backward(self, dy):
        N, C, H, W = dy.shape
        dx = torch.zeros((N, C, H//self.scale_factor, W//self.scale_factor)).to(self.device)
        for i in range(H//self.scale_factor):
            for j in range(W//self.scale_factor):
                dx[:, :, i, j] = dy[:, :, i*self.scale_factor:(i+1)*self.scale_factor, j*self.scale_factor:(j+1)*self.scale_factor].sum(dim=(-1, -2))
        return dx

    def cuda(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"



class Sequential(Module):

    def __init__(self, *functions):
        self.functions = functions

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for func in self.functions:
            x = func.forward(x)
        return x

    def backward(self, grad):
        for func in reversed(self.functions):
            grad = func.backward(grad)
        return grad

    def get_params(self):
        return [func.get_params() for func in self.functions]

    def set_params(self, params):
        for i, param in enumerate(params):
            self.functions[i].set_params(param)

    def cuda(self):
        for func in self.functions:
            func.cuda()

    def params(self):
        parameters = []
        for func in self.functions:
            parameters += func.params()
        return parameters

    def eval(self):
        for func in self.functions:
            func.eval()

    def train(self):
        for func in self.functions:
            func.train()



class SkipConnection(Module):

    def __init__(self, module):
        self.module = module
        self.size_in = None
        self.size_out = None

    def forward(self, x):
        self.size_in = x.shape
        skip_connection = x
        x = self.module.forward(x)
        x =  torch.cat((x, skip_connection), dim=1)
        self.size_out = x.shape
        return x
    
    def backward(self, dz):
        N, C, H, W = dz.shape
        dy = dz[:, :C - self.size_in[1], :, :]
        dx = dz[:, C - self.size_in[1]:, :, :]
        dy = self.module.backward(dy)
        dx += dy
        return dx

    def get_params(self):
        return self.module.get_params()

    def set_params(self, params):
        self.module.set_params(params)

    def cuda(self):
        self.module.cuda()

    def params(self):
        return self.module.params()

    def eval(self):
        self.module.eval()

    def train(self):
        self.module.train()



class MSELoss:

    def __init__(self):
        self.grad = None

    def __call__(self, model_output, ground_truth):
        return self.forward(model_output, ground_truth)

    def forward(self, model_output, ground_truth):
        self.grad = 2 * (model_output - ground_truth) / model_output.numel()
        return ((ground_truth - model_output) ** 2).mean()

    def backward(self):
        return self.grad
    



class Optimizer(object):

    def __init__(self, params, lr):
        self.params = params
        self.lr = lr



    def step(self):
        for param in self.params:
            param[0] -= self.lr * param[1]


class SGDMomentumOptimizer(object):

    def __init__(self, params, lr, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [[0] * len(param) for param in params]

    def step(self):
        for i, param in enumerate(self.params):
            for j in range(len(param)):
                self.velocities[i][j] = self.momentum * self.velocities[i][j] - self.lr * param[j]
                param[j] += self.velocities[i][j]

# class SGDMomentumOptimizer(object):

#     def __init__(self, params, lr, momentum=0.9):
#         self.params = params    
#         self.lr = lr
#         self.momentum = momentum
#         self.velocities = [0.0] * len(params)


#     def step(self):
#         for i, param in enumerate(self.params):
#             gradient = param[1]

#             # Update the velocity
#             self.velocities[i] = self.momentum * self.velocities[i] - self.lr * gradient

#             # Update the parameter
#             param[0] += self.velocities[i]

# Noise2Noise network

def convblock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, activation=LeakyRelu(0.1)):
    return Sequential(
        Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        activation
    )


class Model(Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.encoder01 = Sequential(
            convblock(in_channels=3, out_channels=48, kernel_size=3, stride=1, padding=1, bias=False, activation=LeakyRelu(0.1)),
            convblock(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, bias=False, activation=LeakyRelu(0.1)),
            MaxPool2d()
        )

        self.encoder2 = Sequential(
            convblock(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, bias=False, activation=LeakyRelu(0.1)),
            MaxPool2d()
        )

        self.encoder3 = Sequential(
            convblock(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, bias=False, activation=LeakyRelu(0.1)),
            MaxPool2d()
        )

        self.encoder4 = Sequential(
            convblock(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, bias=False, activation=LeakyRelu(0.1)),
            MaxPool2d()
        )

        self.encoder56 = Sequential(
            convblock(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, bias=False, activation=LeakyRelu(0.1)),
            MaxPool2d(),
            convblock(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, bias=False, activation=LeakyRelu(0.1)),
            UpSampling2D(scale_factor=2),

        )

        self.decoder5ab = Sequential(
            convblock(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False, activation=LeakyRelu(0.1)),
            UpSampling2D(scale_factor=2),
        )

        self.decoder4ab = Sequential(
            convblock(in_channels=144, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False, activation=LeakyRelu(0.1)),
            convblock(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False, activation=LeakyRelu(0.1)),
            UpSampling2D(scale_factor=2),
        )

        self.decoder3ab = Sequential(
            convblock(in_channels=144, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False, activation=LeakyRelu(0.1)),
            convblock(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False, activation=LeakyRelu(0.1)),
            UpSampling2D(scale_factor=2),
        )

        self.decoder2ab = Sequential(
            convblock(in_channels=144, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False, activation=LeakyRelu(0.1)),
            convblock(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False, activation=LeakyRelu(0.1)),
            UpSampling2D(scale_factor=2),
        )

        self.decoder1abc = Sequential(
            convblock(in_channels=99, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False, activation=LeakyRelu(0.1)),
            convblock(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False, activation=LeakyRelu(0.1)),
            Conv2d(channels_in=32, channels_out=3, kernel_size=3, stride=1, padding=1, bias=True),
        )


        self.skip1 = SkipConnection(self.encoder56)
        self.skip2 = SkipConnection(Sequential(self.encoder4, self.skip1, self.decoder5ab))
        self.skip3 = SkipConnection(Sequential(self.encoder3, self.skip2, self.decoder4ab))
        self.skip4 = SkipConnection(Sequential(self.encoder2, self.skip3, self.decoder3ab))
        self.skip5 = SkipConnection(Sequential(self.encoder01, self.skip4, self.decoder2ab))
        self.net = Sequential(self.skip5, self.decoder1abc)

        self.lr = 1e-10
        self.momentum = 0.9

        self.loss = MSELoss()
        self.optimizer = SGDMomentumOptimizer(self.net.params(), lr=self.lr, momentum=self.momentum)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_pretrained_model(self, model_path) -> None:
        ## This loads the parameters saved in bestmodel .pth into the model
        params = torch.load(model_path, map_location=self.device)
        self.net.set_params(params)

    def train(self, train_input, train_target, num_epochs, batch_size = 64) -> None:
        train_input = (train_input.float() / 255).to(self.device)
        train_target = (train_target.float() / 255).to(self.device)
        self.net.cuda()
        self.optimizer = Optimizer(self.net.params(), lr=self.lr)
        self.net.train()
        n_data = len(train_input)
        batch_size = batch_size
        self.loss_history = []

        for epoch in range(num_epochs):
            print(f'EPOCH: {epoch + 1}/{num_epochs}')
            loss = []
            pbar = tqdm(range(0, n_data, batch_size), total=n_data // batch_size, unit="batch")
            for first in pbar:
                with torch.no_grad():
                    last = first + batch_size
                    x_batch, y_batch = train_input[first:last], train_target[first:last]

                    results = self.net.forward(x_batch)
                    loss_ = self.loss(results, y_batch)
                    loss.append(loss_.item())
                    self.net.backward(self.loss.backward())
                    self.optimizer.step()
                
                pbar.set_description(f"loss: {np.mean(loss):.4f}")
            self.loss_history.append(np.mean(loss))
            sum = 0
            for val in loss:
                sum += val
            print(
                f'{(first / batch_size)} / {(n_data // batch_size)} | loss: {(sum / (first / batch_size + 1))}')

    def predict(self, test_input) -> torch.Tensor:
        #: test_input : tensor of size (N1 , C, H, W) with values in range 0 -255 that has tobe denoised by the trained or the loaded network
        #: returns a tensor of the size (N1 , C, H, W) with values in range 0 -255.
        self.net.eval()

        def normalization_cut(imgs):
            imgs_shape = imgs.shape
            imgs = imgs.flatten()
            imgs[imgs < 0] = 0
            imgs[imgs > 1] = 1
            imgs = imgs.view(imgs_shape)
            return imgs

        return 255 * normalization_cut(self.net((test_input.float() / 255).to(self.device))).cpu()

    def save(self, model_path):
        pck_file = self.net.get_params()
        torch.save(pck_file, model_path)
    
