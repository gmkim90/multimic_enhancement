import numpy as np
import torch
import math
import pdb
from torch.nn.modules.utils import _single

def _istuple(x):   return isinstance(x, tuple)
def _mktuple2d(x): return x if _istuple(x) else (x,x)

def complex_rayleigh_init(Wr, Wi, fanin=None, gain=1):
    if not fanin:
        fanin = 1
        for p in W1.shape[1:]:
            fanin *= p

    scale = float(gain)/float(fanin)
    theta  = torch.empty_like(Wr).uniform_(-math.pi/2, +math.pi/2)
    #theta = torch.zeros(Wr.size()).uniform_(-math.pi / 2, +math.pi / 2)
    rho = np.random.rayleigh(scale, tuple(Wr.shape))
    rho = torch.tensor(rho).to(Wr)
    Wr.data.copy_(rho*theta.cos())
    Wi.data.copy_(rho*theta.sin())


def CMatMul(Ar, Ai, Br, Bi):
    #pdb.set_trace()
    #Crr = torch.nn.functional.linear(Ar, Br, None)
    #Cri = torch.nn.functional.linear(Ar, Bi, None)
    #Cir = torch.nn.functional.linear(Ai, Br, None)
    #Cii = torch.nn.functional.linear(Ai, Bi, None)

    Crr = torch.mm(Ar, Br)
    Cri = torch.mm(Ar, Bi)
    Cir = torch.mm(Ai, Br)
    Cii = torch.mm(Ai, Bi)

    Cr = Crr - Cii
    Ci = Cri + Cir

    return Cr, Ci

def CConv1dFunc(Xr, Xi, Wr, Wi):
    #pdb.set_trace()
    Crr = torch.nn.functional.conv1d(Xr, Wr, None, 1, 0, 1, 1)
    Cri = torch.nn.functional.conv1d(Xr, Wi, None, 1, 0, 1, 1)
    Cir = torch.nn.functional.conv1d(Xi, Wr, None, 1, 0, 1, 1)
    Cii = torch.nn.functional.conv1d(Xi, Wi, None, 1, 0, 1, 1)

    Cr = Crr - Cii
    Ci = Cri + Cir

    return Cr, Ci

def Hermitian(Xr, Xi):
    Yr = Xr.transpose(0,1)
    Yi = -Xi.transpose(0,1)

    return Yr, Yi

def CMMhinv(Mr, Mi, cuda=True):
    # purpose of this function is avoid matrix inverse of complex number
    # pdb.set_trace()
    Yr = torch.mm(Mr.transpose(0, 1), Mr) - torch.mm(Mi.transpose(0, 1), Mi)
    #pdb.set_trace()
    Yr = torch.inverse(Yr)
    if(cuda):
        Yi = torch.zeros(Yr.size()).cuda()
    else:
        Yi = torch.zeros(Yr.size())

    return Yr, Yi


class ComplexDepthwiseConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1,  bias=True, init_method = 'real'):
        super(ComplexDepthwiseConv, self).__init__()
        self.in_channels = in_channels
        self.groups = in_channels
        self.out_channels = self.groups*out_channels
        self.kernel_size = _mktuple2d(kernel_size)
        self.stride = _mktuple2d(stride)
        self.padding = _mktuple2d(padding)
        self.dilation = _mktuple2d(dilation)

        self.Wreal = torch.nn.Parameter(torch.Tensor(self.out_channels,
                                                  self.in_channels // self.groups,
                                                  *self.kernel_size))
        self.Wimag = torch.nn.Parameter(torch.Tensor(self.out_channels,
                                                  self.in_channels // self.groups,
                                                  *self.kernel_size))
        if bias:
            self.Breal = torch.nn.Parameter(torch.Tensor(self.out_channels))
            self.Bimag = torch.nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter("Breal", None)
            self.register_parameter("Bimag", None)
        self.reset_parameters(init_method)

    def reset_parameters(self, init_method):
        if(init_method == 'complex'):
            fanin = self.in_channels // self.groups
            for s in self.kernel_size: fanin *= s
            complex_rayleigh_init(self.Wreal, self.Wimag, fanin)
        elif(init_method == 'real'):
            torch.nn.init.kaiming_normal_(self.Wreal, mode='fan_out', nonlinearity='relu')
            torch.nn.init.kaiming_normal_(self.Wimag, mode='fan_out', nonlinearity='relu')

        if self.Breal is not None and self.Bimag is not None:
            self.Breal.data.zero_()
            self.Bimag.data.zero_()

    def forward(self, xr, xi):
        #pdb.set_trace()
        yrr = torch.nn.functional.conv2d(xr, self.Wreal, self.Breal, self.stride, self.padding, self.dilation, self.groups)
        yri = torch.nn.functional.conv2d(xr, self.Wimag, self.Bimag, self.stride, self.padding, self.dilation, self.groups)
        yir = torch.nn.functional.conv2d(xi, self.Wreal, None, self.stride, self.padding, self.dilation, self.groups)
        yii = torch.nn.functional.conv2d(xi, self.Wimag, None, self.stride, self.padding, self.dilation, self.groups)

        Yr = yrr - yii
        Yi = yri + yir

        Yr = Yr.view(Yr.size(0), self.in_channels, -1, Yr.size(3))
        Yi = Yi.view(Yi.size(0), self.in_channels, -1, Yi.size(3))
        return Yr, Yi

class RealDepthwiseConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1,  bias=True):
        super(RealDepthwiseConv, self).__init__()
        self.in_channels = in_channels
        self.groups = in_channels
        self.out_channels = self.groups*out_channels
        self.kernel_size = _mktuple2d(kernel_size)
        self.stride = _mktuple2d(stride)
        self.padding = _mktuple2d(padding)
        self.dilation = _mktuple2d(dilation)

        self.W = torch.nn.Parameter(torch.Tensor(self.out_channels,
                                                  self.in_channels // self.groups,
                                                  *self.kernel_size))
        if bias:
            self.B = torch.nn.Parameter(torch.Tensor(self.out_channels))

        else:
            self.register_parameter("B", None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.W, mode='fan_out', nonlinearity='relu')

        if self.B is not None:
            self.B.data.zero_()

    def forward(self, x):
        y = torch.nn.functional.conv2d(x, self.W, self.B, self.stride, self.padding, self.dilation, self.groups)
        y = y.view(y.size(0), self.in_channels, -1, y.size(3))

        return y

class ComplexSeqWise(torch.nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim N*H*T to (N*T)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(ComplexSeqWise, self).__init__()
        self.module = module

    def forward(self, xr, xi):
        assert(xr.dim() == 3 and xi.dim() == 3)
        n, t = xr.size(0), xr.size(2)
        xr = xr.transpose(1, 2) # (NxHxT) --> (NxTxH)
        xr = xr.contiguous().view(n*t, -1)
        xi = xi.transpose(1, 2) # (NxHxT) --> (NxTxH)
        xi = xi.contiguous().view(n*t, -1)
        xr, xi = self.module(xr, xi)
        xr = xr.view(n, t, -1)
        xr = xr.transpose(1, 2)
        xi = xi.view(n, t, -1)
        xi = xi.transpose(1, 2)
        return xr, xi

class ComplexSeqDepthWise(torch.nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim N*F*H*T to (N*T)*F*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(ComplexSeqDepthWise, self).__init__()
        self.module = module

    def forward(self, xr, xi):
        assert(xr.dim() == 4 and xi.dim() == 4)
        n, f, h, t = xr.size(0), xr.size(1), xr.size(2), xr.size(3)
        xr = xr.transpose(2, 3).transpose(1, 2) # (NxFxHxT) --> (NxTxFxH)
        xr = xr.contiguous().view(n*t, f, h)
        xi = xi.transpose(2, 3).transpose(1, 2) # (NxFxHxT) --> (NxTxFxH)
        xi = xi.contiguous().view(n*t, f, h)
        xr, xi = self.module(xr, xi)
        xr = xr.view(n, t, f, h)
        xr = xr.transpose(1, 2).transpose(2, 3)
        xi = xi.view(n, t, f, h)
        xi = xi.transpose(1, 2).transpose(2, 3)
        return xr, xi


class RealSeqDepthWise(torch.nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim N*F*H*T to N*H*F*T, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(RealSeqDepthWise, self).__init__()
        self.module = module

    def forward(self, x):
        assert(x.dim() == 4)
        #n, f, h, t = x.size(0), x.size(1), x.size(2), x.size(3)
        x = x.transpose(1, 2) # (NxFxHxT) --> (NxHxFxT)
        x = self.module(x)
        x = x.transpose(1, 2)

        return x

class ComplexBatchNorm(torch.nn.Module):
    """Mostly copied/inspired from PyTorch torch/nn/modules/batchnorm.py"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(num_features))
            # WHERE IS Wir?
            self.Br = torch.nn.Parameter(torch.Tensor(num_features))
            self.Bi = torch.nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            # WHERE IS Wir?
            self.register_parameter('Wii', None)
            self.register_parameter('Br', None)
            self.register_parameter('Bi', None)
        if self.track_running_stats:
            self.register_buffer('RMr', torch.zeros(num_features))
            self.register_buffer('RMi', torch.zeros(num_features))
            self.register_buffer('RVrr', torch.ones(num_features))
            self.register_buffer('RVri', torch.zeros(num_features))
            self.register_buffer('RVii', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr', None)
            self.register_parameter('RMi', None)
            self.register_parameter('RVrr', None)
            self.register_parameter('RVri', None)
            self.register_parameter('RVii', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9)  # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert (xr.shape == xi.shape)
        assert (xr.size(1) == self.num_features)

    def forward(self, xr, xi):
        self._check_input_dim(xr, xi)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
                #exponential_average_factor = 1.0 / self.num_batches_tracked.data[0]
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i != 1]
        if not training:
            vdim = [1] * xr.dim()
            vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            Mr = xr
            Mi = xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr - Mr, xi - Mi

        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        #
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr = Vrr + self.eps
        Vri = Vri
        Vii = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        tau = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, -1, Vri, Vri)
        s = delta.sqrt()
        t = (tau + 2 * s).sqrt()
        rst = (s * t).reciprocal()

        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = (-Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Zrr = self.Wrr * Urr + self.Wri * Uri
            Zri = self.Wrr * Uri + self.Wri * Uii
            Zir = self.Wri * Urr + self.Wii * Uri
            Zii = self.Wri * Uri + self.Wii * Uii
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        #pdb.set_trace()
        yr, yi = Zrr * xr + Zri * xi, Zir * xr + Zii * xi

        if self.affine:
            yr = yr + self.Br
            yi = yi + self.Bi

        return yr, yi

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)
'''
    def _load_from_state_dict(self, state_dict, prefix, strict, missing_keys,
                              unexpected_keys, error_msgs):
        super(ComplexBatchNorm, self)._load_from_state_dict(state_dict,
                                                            prefix,
                                                            strict,
                                                            missing_keys,
                                                            unexpected_keys,
                                                            error_msgs)
'''


class ComplexDepthwiseBatchNorm(torch.nn.Module):
    """ComplexBatchNorm extension for DepthwiseConvolution layer"""

    def __init__(self, num_features, num_channels, eps=1e-6, eps_freq0=-10, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(ComplexDepthwiseBatchNorm, self).__init__()
        self.num_features = num_features
        self.num_channels = num_channels
        self.eps = eps
        self.eps_freq0 = eps_freq0
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(num_channels, num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(num_channels, num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(num_channels, num_features))
            self.Br = torch.nn.Parameter(torch.Tensor(num_channels, num_features))
            self.Bi = torch.nn.Parameter(torch.Tensor(num_channels, num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br', None)
            self.register_parameter('Bi', None)
        if self.track_running_stats:
            self.register_buffer('RMr', torch.zeros(num_channels, num_features))
            self.register_buffer('RMi', torch.zeros(num_channels, num_features))
            self.register_buffer('RVrr', torch.ones(num_channels, num_features))
            self.register_buffer('RVri', torch.zeros(num_channels, num_features))
            self.register_buffer('RVii', torch.ones(num_channels, num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr', None)
            self.register_parameter('RMi', None)
            self.register_parameter('RVrr', None)
            self.register_parameter('RVri', None)
            self.register_parameter('RVii', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9)  # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        #pdb.set_trace()
        assert (xr.shape == xi.shape)
        assert (xr.size(-1) == self.num_features)

    def forward(self, xr, xi):
        #pdb.set_trace()
        self._check_input_dim(xr, xi)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        #redux = [i for i in reversed(range(xr.dim())) if i != 1]
        redux = [0] # get  average statistic only along dim=0 (size=N*T)

        '''
        if not training:
            vdim = [1, self.num_channels, xr.size(-1)]
            #vdim[-1] = xr.size(-1)
        '''

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        #pdb.set_trace()
        if training:
            Mr = xr
            Mi = xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            #pdb.set_trace()
            #Mr = self.RMr.view(vdim)
            #Mi = self.RMi.view(vdim)
            Mr = self.RMr
            Mi = self.RMi
        xr, xi = xr - Mr, xi - Mi

        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.

        #pdb.set_trace()
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            #Vrr = self.RVrr.view(vdim)
            #Vri = self.RVri.view(vdim)
            #Vii = self.RVii.view(vdim)
            Vrr = self.RVrr
            Vri = self.RVri
            Vii = self.RVii

        #Vrr = Vrr + self.eps  # remove Tikhonov regularization for now
        #Vii = Vii + self.eps  # remove Tikhonov regularization for now

        if(self.eps_freq0 > 0):
            Vrr[:, 0] = Vrr[:, 0] + self.eps_freq0
            Vii[:, 0] = Vii[:, 0] + self.eps_freq0

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        tau = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, -1, Vri, Vri) # how to ensure positiveness of delta?
        '''
        if((delta<0).sum().item() > 0):
            idx = (delta<0).nonzero()
            print('delta < 0 detected. save input as .mat')
            print(idx)
            import scipy.io as sio
            sio.savemat('input_delta_negative.mat', {'xr':xr.data.cpu().numpy(), 'xi': xi.data.cpu().numpy()})
            exit(0)
        '''
        delta = torch.clamp(delta, min = self.eps, max = 100000000)

        #pdb.set_trace()
        s = delta.sqrt() # if delta has negative number, then s has nan
        t = (tau + 2 * s).sqrt()
        rst = (s * t).reciprocal() # original
        #rst = t.reciprocal() # ken
        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = (-Vri) * rst # original

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Zrr = self.Wrr * Urr + self.Wri * Uri
            Zri = self.Wrr * Uri + self.Wri * Uii
            Zir = self.Wri * Urr + self.Wii * Uri
            Zii = self.Wri * Uri + self.Wii * Uii
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        #pdb.set_trace()
        yr, yi = Zrr * xr + Zri * xi, Zir * xr + Zii * xi

        if self.affine:
            yr = yr + self.Br
            yi = yi + self.Bi

        return yr, yi

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

'''
    def _load_from_state_dict(self, state_dict, prefix, strict, missing_keys,
                              unexpected_keys, error_msgs):
        super(ComplexDepthwiseBatchNorm, self)._load_from_state_dict(state_dict,
                                                            prefix,
                                                            strict,
                                                            missing_keys,
                                                            unexpected_keys,
                                                            error_msgs)
'''

class ComplexConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, init_method='real'):
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _mktuple2d(kernel_size)
        self.stride = _mktuple2d(stride)
        self.padding = _mktuple2d(padding)
        self.dilation = _mktuple2d(dilation)
        self.groups = groups

        self.Wr = torch.nn.Parameter(torch.Tensor(self.out_channels,
                                                  self.in_channels // self.groups,
                                                  *self.kernel_size))
        self.Wi = torch.nn.Parameter(torch.Tensor(self.out_channels,
                                                  self.in_channels // self.groups,
                                                  *self.kernel_size))
        if bias:
            self.Br = torch.nn.Parameter(torch.Tensor(out_channels))
            self.Bi = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("Br", None)
            self.register_parameter("Bi", None)
        self.reset_parameters(init_method)

    def reset_parameters(self, init_method):
        if(init_method == 'complex'):
            fanin = self.in_channels // self.groups
            for s in self.kernel_size: fanin *= s
            complex_rayleigh_init(self.Wr, self.Wi, fanin)
        elif(init_method == 'real'):
            torch.nn.init.kaiming_normal_(self.Wr, mode='fan_out', nonlinearity='relu')
            torch.nn.init.kaiming_normal_(self.Wi, mode='fan_out', nonlinearity='relu')

        if self.Br is not None and self.Bi is not None:
            self.Br.data.zero_()
            self.Bi.data.zero_()

    def forward(self, xr, xi):
        yrr = torch.nn.functional.conv2d(xr, self.Wr, self.Br, self.stride,
                                         self.padding, self.dilation, self.groups)
        yri = torch.nn.functional.conv2d(xr, self.Wi, self.Bi, self.stride,
                                         self.padding, self.dilation, self.groups)
        yir = torch.nn.functional.conv2d(xi, self.Wr, None, self.stride,
                                         self.padding, self.dilation, self.groups)
        yii = torch.nn.functional.conv2d(xi, self.Wi, None, self.stride,
                                         self.padding, self.dilation, self.groups)
        return yrr - yii, yri + yir


class ComplexConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, init_method='real'):
        super(ComplexConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.groups = groups

        self.Wreal = torch.nn.Parameter(torch.Tensor(self.out_channels,
                                                  self.in_channels // self.groups,
                                                  *self.kernel_size))
        self.Wimag = torch.nn.Parameter(torch.Tensor(self.out_channels,
                                                  self.in_channels // self.groups,
                                                  *self.kernel_size))
        if bias:
            self.Breal = torch.nn.Parameter(torch.Tensor(out_channels))
            self.Bimag = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("Breal", None)
            self.register_parameter("Bimag", None)
        self.reset_parameters(init_method)

    def reset_parameters(self, init_method):
        if (init_method == 'complex'):
            fanin = self.in_channels // self.groups
            for s in self.kernel_size: fanin *= s
            complex_rayleigh_init(self.Wreal, self.Wimag, fanin)
        elif (init_method == 'real'):
            torch.nn.init.kaiming_normal_(self.Wreal, mode='fan_out', nonlinearity='relu')
            torch.nn.init.kaiming_normal_(self.Wimag, mode='fan_out', nonlinearity='relu')

        if self.Breal is not None and self.Bimag is not None:
            self.Breal.data.zero_()
            self.Bimag.data.zero_()

    def forward(self, xr, xi):
        yrr = torch.nn.functional.conv1d(xr, self.Wreal, self.Breal, self.stride, self.padding, self.dilation, self.groups)
        yri = torch.nn.functional.conv1d(xr, self.Wimag, self.Bimag, self.stride, self.padding, self.dilation, self.groups)
        yir = torch.nn.functional.conv1d(xi, self.Wreal, None, self.stride, self.padding, self.dilation, self.groups)
        yii = torch.nn.functional.conv1d(xi, self.Wimag, None, self.stride, self.padding, self.dilation, self.groups)
        return yrr - yii, yri + yir


class ComplexLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_method='real'):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.Wr = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.Wi = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.Br = torch.nn.Parameter(torch.Tensor(out_features))
            self.Bi = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('Br', None)
            self.register_parameter('Bi', None)
        self.reset_parameters(init_method)

    def reset_parameters(self, init_method):
        if(init_method == 'complex'):
            complex_rayleigh_init(self.Wr, self.Wi, self.in_features)
        elif(init_method == 'real'):
            torch.nn.init.kaiming_uniform_(self.Wr, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.Wi, a=math.sqrt(5))

        if self.Br is not None and self.Bi is not None:
            self.Br.data.zero_()
            self.Bi.data.zero_()

    def forward(self, xr, xi):
        yrr = torch.nn.functional.linear(xr, self.Wr, self.Br)
        yri = torch.nn.functional.linear(xr, self.Wi, self.Bi)
        yir = torch.nn.functional.linear(xi, self.Wr, None)
        yii = torch.nn.functional.linear(xi, self.Wi, None)
        return yrr - yii, yri + yir