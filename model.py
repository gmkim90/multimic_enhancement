import math
from common_layer import *
from collections import OrderedDict
import librosa
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import pdb

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())

def get_linearF_from_melF(mel_basis):
    nMel = mel_basis.shape[0]
    linearF_range = torch.IntTensor(nMel, 2)

    for m in range(nMel):
        linearFs = mel_basis[m].nonzero()[0]
        #pdb.set_trace()
        linearF_range[m, 0] = np.asscalar(linearFs[0])
        linearF_range[m, 1] = np.asscalar(linearFs[-1])

    return linearF_range


def spec_to_specCW(spec, duration_side=1, CW_side=4):
    # assume spec is single utterance
    # spec: FxT
    assert(spec.size(0) == 1)

    spec = spec[0]

    spec_r, spec_i = torch.split(spec, int(spec.size(0)/2), dim=0)
    CH = spec_r.size(0)
    center_frame = int(spec.size(-1)/2)
    start_frame = center_frame - duration_side
    end_frame = center_frame + duration_side

    specCW_r = torch.FloatTensor((duration_side * 2 + 1), CH * (2 * CW_side + 1)).fill_(0)
    specCW_i = torch.FloatTensor((duration_side * 2 + 1), CH * (2 * CW_side + 1)).fill_(0)

    for t in range(start_frame, end_frame+1):
        spec_r_delT = spec_r[:, t-CW_side:t+CW_side+1]
        #pdb.set_trace()
        specCW_r[t-start_frame] = spec_r_delT.contiguous().view(spec_r_delT.size(0)*spec_r_delT.size(1))

        spec_i_delT = spec_i[:, t-CW_side:t+CW_side+1]
        specCW_i[t-start_frame] = spec_i_delT.contiguous().view(spec_i_delT.size(0)*spec_i_delT.size(1))

    specCW = torch.cat((specCW_r, specCW_i), dim=1) # Tx(2*CH*CW) = 5x36 (first half: real, next half: imag)

    return specCW.cuda()

def spec_to_specfixedlen(spec, duration_side=2):
    # assume spec is single utterance
    # spec: FxT
    assert (spec.size(0) == 1)

    center_frame = int(spec.size(-1)/2)
    start_frame = center_frame - duration_side
    end_frame = center_frame + duration_side

    specfixedlen = spec[:, :, start_frame:end_frame+1].squeeze().transpose(0,1)


    return specfixedlen


class STFT_to_LMFB(nn.Module):
    def __init__(self, mel_basis, win_size=320, cuda=True, window_change=True):
        super(STFT_to_LMFB, self).__init__()
        if(cuda):
            self.mel_basis = Variable(
                torch.unsqueeze(torch.FloatTensor(mel_basis), -1).cuda())  # 40x(nFFT/2+1)x1
        else:
            self.mel_basis = Variable(
                torch.unsqueeze(torch.FloatTensor(mel_basis), -1))
        self.win_size = win_size

        self.window_change = window_change

    def forward(self, stft):
        stft_real, stft_imag = torch.split(stft, int(stft.size(1)/2), dim=1)
        #pdb.set_trace()
        if(self.window_change and not self.win_size == 320):
            # additional istft(custom window size) + stft(20ms) will be needed
            # this time, is should be done in librosa
            N = stft_real.size(0)
            F = stft_real.size(1)
            T = stft_real.size(2)
            stft_np = np.zeros((N, F, T), dtype=complex)
            stft_np.real = stft_real.data.cpu().numpy()
            stft_np.imag = stft_imag.data.cpu().numpy()

            for n in range(stft_np.shape[0]):
                #pdb.set_trace()
                stft_np_n = stft_np[n]
                wav = librosa.istft(stft_np_n, hop_length=int(self.win_size/2), win_length=self.win_size, window='hamming')
                stft_20ms = librosa.stft(wav, n_fft=320, hop_length=160, win_length=320, window='hamming')
                if(n == 0):
                    T20 = stft_20ms.shape[1]
                    F20 = 161
                    stft_real_20ms = torch.FloatTensor(N, F20, T20)
                    stft_imag_20ms = torch.FloatTensor(N, F20, T20)

                #pdb.set_trace()
                stft_real_20ms[n] = torch.FloatTensor(stft_20ms.real)
                stft_imag_20ms[n] = torch.FloatTensor(stft_20ms.imag)
            stft_real = stft_real_20ms
            stft_imag = stft_imag_20ms

        power = torch.pow(stft_real, 2) + torch.pow(stft_imag, 2)
        energy = torch.sqrt(power)
        #pdb.set_trace()
        #lmfb = F.conv1d(power, self.mel_basis)

        mfb = torch.nn.functional.conv1d(energy.cuda(), self.mel_basis)
        out = torch.log1p(mfb)

        return out


class Spec_to_LMFB(nn.Module):
    def __init__(self, mel_basis, nCH):
        super(Spec_to_LMFB, self).__init__()
        self.mel_basis = Variable(torch.unsqueeze(torch.FloatTensor(mel_basis).repeat(1, nCH), -1).cuda())  # 40x(nFFT/2+1)x1

    def forward(self, input):
        input = input.contiguous()
        input = input.view(input.size(0), 2, -1, input.size(-1))  # Nx2x(CH*F)xT
        enh_real = input[:, 0]
        enh_imag = input[:, 1]

        enh_power = torch.pow(enh_real, 2) + torch.pow(enh_imag, 2)
        enh_energy = torch.sqrt(enh_power)
        # enh_mel = F.conv1d(enh_power, self.mel_basis)
        enh_mel = F.conv1d(enh_energy, self.mel_basis)

        out = torch.log1p(enh_mel)
        
        return out

class Encoder(nn.Module):
    def __init__(self, c_in=40, c_h1=128, c_h2=512, c_h3=128, ns=0.2, do_GRL=False, dp=0.3, two_input=False):
        super(Encoder, self).__init__()
        self.ns = ns
        self.two_input = two_input
        self.conv1s = nn.ModuleList(
                [nn.Conv1d(c_in, c_h1, kernel_size=k) for k in range(1, 8)]
            )
        self.conv2 = nn.Conv1d(len(self.conv1s)*c_h1 + c_in, c_h2, kernel_size=1)
        self.conv3 = nn.Conv1d(c_h2, c_h2, kernel_size=5)
        self.conv4 = nn.Conv1d(c_h2, c_h2, kernel_size=5, stride=2)
        self.conv5 = nn.Conv1d(c_h2, c_h2, kernel_size=5)
        self.conv6 = nn.Conv1d(c_h2, c_h2, kernel_size=5, stride=2)
        self.conv7 = nn.Conv1d(c_h2, c_h2, kernel_size=5)
        self.conv8 = nn.Conv1d(c_h2, c_h2, kernel_size=5, stride=2)
        self.dense1 = nn.Linear(c_h2, c_h2)
        self.dense2 = nn.Linear(c_h2, c_h2)
        self.dense3 = nn.Linear(c_h2, c_h2)
        self.dense4 = nn.Linear(c_h2, c_h2)
        self.RNN = nn.GRU(input_size=c_h2, hidden_size=c_h3, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(c_h2 + 2*c_h3, c_h2)
        # normalization layer
        self.ins_norm1 = nn.InstanceNorm1d(c_h2)
        self.ins_norm2 = nn.InstanceNorm1d(c_h2)
        self.ins_norm3 = nn.InstanceNorm1d(c_h2)
        self.ins_norm4 = nn.InstanceNorm1d(c_h2)
        self.ins_norm5 = nn.InstanceNorm1d(c_h2)
        self.ins_norm6 = nn.InstanceNorm1d(c_h2)
        #self.drop1 = nn.Dropout(p=dp)
        #self.drop2 = nn.Dropout(p=dp)
        #self.drop3 = nn.Dropout(p=dp)
        #self.drop4 = nn.Dropout(p=dp)
        #self.drop5 = nn.Dropout(p=dp)
        #self.drop6 = nn.Dropout(p=dp)

        if(self.two_input):
            self.conv_reduce_dim_half = nn.Conv1d(c_in, int(c_in/2), kernel_size=1)

        self.do_GRL = do_GRL

    def conv_block(self, x, conv_layers, norm_layers, res=True):
        out = x
        for layer in conv_layers:
            out = pad_layer(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in norm_layers:
            out = layer(out)
        if res:
            x_pad = F.pad(x, pad=(0, x.size(2) % 2), mode='reflect')
            x_down = F.avg_pool1d(x_pad, kernel_size=2)
            out = x_down + out
        return out

    def dense_block(self, x, layers, norm_layers, res=True):
        out = x
        for layer in layers:
            out = linear(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in norm_layers:
            out = layer(out)
        if res:
            out = out + x
        return out

    def enable_GRL_once(self):
        self.do_GRL = True

    def forward(self, x):
        if(self.two_input):
            x2 = x[1]
            x = x[0]

        if(self.do_GRL):
            x = grad_reverse(x)
            self.do_GRL = False # make do_GRL option will be False by default

        if(self.two_input): # reduce dimension by half & concat
            #pdb.set_trace()
            x_half = self.conv_reduce_dim_half(x)
            x2_half = self.conv_reduce_dim_half(x2)
            #print(x_half.size())
            #print(x2_half.size())
            x = torch.cat((x_half, x2_half), dim=1)

        outs = []
        for l in self.conv1s:
            out = pad_layer(x, l)
            outs.append(out)
        #pdb.set_trace()
        out = torch.cat(outs + [x], dim=1)
        out = F.leaky_relu(out, negative_slope=self.ns)
        #out = self.conv_block(out, [self.conv2], [self.ins_norm1, self.drop1], res=False)
        out = self.conv_block(out, [self.conv2], [self.ins_norm1], res=False)
        #out = self.conv_block(out, [self.conv3, self.conv4], [self.ins_norm2, self.drop2])
        out = self.conv_block(out, [self.conv3, self.conv4], [self.ins_norm2])
        #out = self.conv_block(out, [self.conv5, self.conv6], [self.ins_norm3, self.drop3])
        out = self.conv_block(out, [self.conv5, self.conv6], [self.ins_norm3])
        #out = self.conv_block(out, [self.conv7, self.conv8], [self.ins_norm4, self.drop4])

        out = self.conv_block(out, [self.conv7, self.conv8], [self.ins_norm4])

        # dense layer
        #out = self.dense_block(out, [self.dense1, self.dense2], [self.ins_norm5, self.drop5], res=True)
        out = self.dense_block(out, [self.dense1, self.dense2], [self.ins_norm5], res=True)
        #out = self.dense_block(out, [self.dense3, self.dense4], [self.ins_norm6, self.drop6], res=True)
        out = self.dense_block(out, [self.dense3, self.dense4], [self.ins_norm6], res=True)
        out_rnn = RNN(out, self.RNN)
        out = torch.cat([out, out_rnn], dim=1)
        out = linear(out, self.linear)
        out = F.leaky_relu(out, negative_slope=self.ns)
        return out

class Decoder(nn.Module):
    def __init__(self, c_in=512, c_out=40, c_h=512, ns=0.2, output_by_mask=False):
        super(Decoder, self).__init__()

        self.output_by_mask=output_by_mask

        self.ns = ns
        self.conv1 = nn.Conv1d(c_in, 2*c_h, kernel_size=3)
        self.conv2 = nn.Conv1d(c_h, c_h, kernel_size=3)
        self.conv3 = nn.Conv1d(c_h, 2*c_h, kernel_size=3)
        self.conv4 = nn.Conv1d(c_h, c_h, kernel_size=3)
        self.conv5 = nn.Conv1d(c_h, 2*c_h, kernel_size=3)
        self.conv6 = nn.Conv1d(c_h, c_h, kernel_size=3)
        self.dense1 = nn.Linear(c_h, c_h)
        self.dense2 = nn.Linear(c_h, c_h)
        self.dense3 = nn.Linear(c_h, c_h)
        self.dense4 = nn.Linear(c_h, c_h)
        self.RNN = nn.GRU(input_size=c_h, hidden_size=c_h//2, num_layers=1, bidirectional=True)
        self.dense5 = nn.Linear(2*c_h, c_h)
        self.linear = nn.Linear(c_h, c_out)

        # normalization layer
        self.ins_norm1 = nn.InstanceNorm1d(c_h)
        self.ins_norm2 = nn.InstanceNorm1d(c_h)
        self.ins_norm3 = nn.InstanceNorm1d(c_h)
        self.ins_norm4 = nn.InstanceNorm1d(c_h)
        self.ins_norm5 = nn.InstanceNorm1d(c_h)

    def conv_block(self, x, conv_layers, norm_layer, res=True):
        # first layer
        out = pad_layer(x, conv_layers[0])
        out = F.leaky_relu(out, negative_slope=self.ns)

        # upsample by pixelshuffle
        out = pixel_shuffle_1d(out, upscale_factor=2)
        out = pad_layer(out, conv_layers[1])
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = norm_layer(out)
        if res:
            x_up = upsample(x, scale_factor=2)
            out = out + x_up
        return out

    def dense_block(self, x, layers, norm_layer, res=True):
        out = x
        for layer in layers:
            out = linear(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        out = norm_layer(out)
        if res:
            out = out + x
        return out

    def forward(self, x):
        # conv layer
        #pdb.set_trace()
        out = self.conv_block(x, [self.conv1, self.conv2], self.ins_norm1, res=True )
        out = self.conv_block(out, [self.conv3, self.conv4], self.ins_norm2, res=True)
        out = self.conv_block(out, [self.conv5, self.conv6], self.ins_norm3, res=True)

        # dense layer
        out = self.dense_block(out, [self.dense1, self.dense2], self.ins_norm4, res=True)
        out = self.dense_block(out, [self.dense3, self.dense4], self.ins_norm5, res=True)

        # rnn layer
        out_rnn = RNN(out, self.RNN)
        out = torch.cat([out, out_rnn], dim=1)
        out = linear(out, self.dense5)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = linear(out, self.linear)

        if(self.output_by_mask):
            out = torch.mul(out, x)

        return out

class SpeechClassifierRNN(nn.Module):
    def __init__(self, I, O, H, L=3, do_GRL=False, rnn_type=nn.LSTM):
        super(SpeechClassifierRNN, self).__init__()
        self.I = I
        self.H = H
        self.L = L

        self.do_GRL = do_GRL

        self.rnn1 = BRNN(input_size=H, hidden_size = H, rnn_type = rnn_type, bidirectional=True)
        self.rnn2 = BRNN(input_size=H, hidden_size=H, rnn_type=rnn_type, bidirectional=True)
        self.rnn3 = BRNN(input_size=H, hidden_size=H, rnn_type=rnn_type, bidirectional=True)

        if(I != H):
            self.first_linear = nn.Conv1d(I, H, kernel_size=1, stride=1, padding=0) # linear transform from dimension I to H, needed for residual connection
        else:
            self.first_linear = None

        self.final_linear = nn.Linear(H, O)
        self.criterion = nn.CrossEntropyLoss(size_average = False)   # nn.LogSoftmax() + nn.NLLLoss() in one single class


    def enable_GRL_once(self):
        self.do_GRL = True

    def forward(self, input, target):
        if(self.do_GRL): # this option can be controlled outside the model (trainer)
            input = grad_reverse(input)
            self.do_GRL = False # make do_GRL option False by default

        if(self.first_linear):
            input = self.first_linear(input)
        input = input.transpose(1,2).transpose(0,1) # Transpose: NxHxT --> TxNxH
        h1 = self.rnn1(input) + input
        h2 = self.rnn2(h1) + h1
        h3 = self.rnn3(h2) + h2

        # Ver1
        #h3 = h3.transpose(0,1).transpose(1,2) # Transpose back: TxNxH --> NxHxT

        # Ver2
        h3 = h3.sum(0) # TxNxH --> NxH

        output = self.final_linear(h3) # NxH --> NxO
        loss = self.criterion(output, target)
        acc = calc_acc(output, target)

        return loss, acc


class BRNNmultiCH(nn.Module):
    def __init__(self, I, H, L, O, nCH, mel_basis, rnn_type=nn.GRU, output_is_LMFB=True):
        super(BRNNmultiCH, self).__init__()
        self.I = I
        self.H = H
        #self.L = L # currently, fix L=2 (implementation issue)
        self.O = O
        self.nCH = nCH
        self.rnn_type = rnn_type
        self.L = L

        self.output_is_LMFB = output_is_LMFB

        self.rnn1 = BRNN(input_size=H, hidden_size = H, rnn_type = rnn_type, bidirectional=True)
        self.rnn2 = BRNN(input_size=H, hidden_size = H, rnn_type = rnn_type, bidirectional=True)
        if (self.L == 3):
            self.rnn3 = BRNN(input_size=H, hidden_size=H, rnn_type=rnn_type, bidirectional=True)
        #pdb.set_trace()

        self.first_linear = nn.Conv1d(I, H, kernel_size=1, stride=1, padding=0) # linear transform from dimension I to H, needed for residual connection
        self.final_linear = nn.Conv1d(H, O, kernel_size=1, stride=1, padding=0)
        if(self.output_is_LMFB):
            self.mel_basis = Variable(torch.unsqueeze(torch.FloatTensor(mel_basis).repeat(1,self.nCH),-1).cuda()) # 40x(nFFT/2+1)x1

    def forward(self, input):
        #input: (N, nCH*F*2,T) (2: real/img)

        input_linear = self.first_linear(input)
        input_linear = input_linear.transpose(1,2).transpose(0,1) # Transpose: NxHxT --> TxNxH

        h1 = self.rnn1(input_linear) + input_linear
        h2 = self.rnn2(h1) + h1
        if(self.L == 3):
            h3 = self.rnn3(h2) + h2
            h = h3.transpose(0,1).transpose(1,2) # Transpose back: TxNxH --> NxHxT
        else:
            h = h2.transpose(0,1).transpose(1,2) # Transpose back: TxNxH --> NxHxT

        mask = self.final_linear(h) # (Nx(2*CH*F)xT)
        enh = torch.mul(input, mask)

        if(self.output_is_LMFB):
            enh = enh.view(input.size(0), 2, -1, input.size(-1)) # Nx2x(CH*F)xT
            enh_real = enh[:,0]
            enh_imag = enh[:,1]

            enh_power = torch.pow(enh_real, 2) + torch.pow(enh_imag, 2)
            enh_mel = F.conv1d(enh_power, self.mel_basis)
            output = torch.log1p(enh_mel)
        else:
            output = enh # Nx(2*CH*F)xT

        return output

    def set_output_LMFB(self):
        self.output_is_LMFB = True

    def set_output_SPEC(self):
        self.output_is_LMFB = False


class stackedBRNN(nn.Module):
    def __init__(self, I, O, H, L, do_GRL=False, rnn_type=nn.LSTM):
        super(stackedBRNN, self).__init__()
        self.I = I
        self.H = H
        self.L = L
        self.rnn_type = rnn_type

        self.do_GRL = do_GRL

        self.rnn1 = BRNN(input_size=H, hidden_size = H, rnn_type = rnn_type, bidirectional=True)
        self.rnn2 = BRNN(input_size=H, hidden_size=H, rnn_type=rnn_type, bidirectional=True)
        self.rnn3 = BRNN(input_size=H, hidden_size=H, rnn_type=rnn_type, bidirectional=True)
        self.rnn4 = BRNN(input_size=H, hidden_size=H, rnn_type=rnn_type, bidirectional=True)

        self.first_linear = nn.Conv1d(I, H, kernel_size=1, stride=1, padding=0) # linear transform from dimension I to H, needed for residual connection
        self.final_linear = nn.Conv1d(H, O, kernel_size=1, stride=1, padding=0) # linear transform from dimension H to I, needed for final output as logMel spectrogram


    def enable_GRL_once(self):
        self.do_GRL = True

    def forward(self, input):
        if(self.do_GRL):
            input = grad_reverse(input)
            self.do_GRL=False # make do_GRL option false by default

        input_linear = self.first_linear(input)
        input_linear = input_linear.transpose(1,2).transpose(0,1) # Transpose: NxHxT --> TxNxH
        h1 = self.rnn1(input_linear) + input_linear
        h2 = self.rnn2(h1) + h1
        h3 = self.rnn3(h2) + h2
        h4 = self.rnn4(h3) + h3
        h4 = h4.transpose(0,1).transpose(1,2) # Transpose back: TxNxH --> NxHxT
        #pdb.set_trace()
        output = self.final_linear(h4)

        return output

    def forward_paired(self, input, paired):

        input = torch.cat((input, paired), dim=1)
        output = self.forward(input)

        return output

    def forward_with_intermediate_output(self, input):
        #pdb.set_trace()
        input_linear = self.first_linear(input)
        input_linear = input_linear.transpose(1,2).transpose(0,1) # Transpose: NxHxT --> TxNxH
        h1 = self.rnn1(input_linear) + input_linear
        h2 = self.rnn2(h1) + h1
        h3 = self.rnn3(h2) + h2
        h4 = self.rnn4(h3) + h3
        h4 = h4.transpose(0,1).transpose(1,2) # Transpose back: TxNxH --> NxHxT
        #pdb.set_trace()
        output = self.final_linear(h4)

        return [output, h4]



class DeepSpeech(nn.Module):
    def __init__(self, rnn_type=nn.LSTM, labels="abc", rnn_hidden_size=512, rnn_layers=2, bidirectional=True,
                 kernel_sz=11, stride=2, map=256, cnn_layers=2,
                 nFreq=40, nDownsample=1, audio_conf = None):
        super(DeepSpeech, self).__init__()

        # model metadata needed for serialization/deserialization
        self.nFreq = nFreq

        self._version = '0.0.1'

        self._audio_conf = audio_conf # not used

        # RNN
        self.rnn_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        # CNN
        self.cnn_stride = stride   # use stride for subsampling
        self.cnn_map = map
        self.cnn_kernel = kernel_sz
        self.nDownsample = nDownsample

        self.cnn_layers = cnn_layers

        self._labels = labels


        num_classes = len(self._labels)

        conv_list = []
        conv_list.append(nn.Conv1d(nFreq, map, kernel_size=kernel_sz, stride=stride))
        conv_list.append(nn.BatchNorm1d(map))
        conv_list.append(nn.LeakyReLU(map, inplace=True))

        if(self.nDownsample == 1):
            stride=1

        for x in range(self.cnn_layers - 1):
            conv_list.append(nn.Conv1d(map, map, kernel_size=kernel_sz, stride=stride))
            conv_list.append(nn.BatchNorm1d(map))
            conv_list.append(nn.LeakyReLU(map, inplace=True))

        self.conv = nn.Sequential(*conv_list)

        rnn_input_size = map

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                       bidirectional=bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(self.rnn_layers - 1):
            rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                           bidirectional=bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()


    def forward(self, x):
        x = self.conv(x)
        #x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
        x = x.transpose(1,2).transpose(0,1)
        x = self.rnns(x)
        x = self.fc(x)
        x = x.transpose(0, 1)

        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x

    @classmethod
    def load_model(cls, path, gpu=-1):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        #pdb.set_trace()
        model = cls(rnn_hidden_size=package['rnn_size'], rnn_layers=package['rnn_layers'], rnn_type=supported_rnns[package['rnn_type']],
                    map=package['cnn_map'], stride = package['cnn_stride'], kernel_sz=package['cnn_kernel'], cnn_layers=package['cnn_layers'],
                    labels=package['labels']
                    )
        # the blacklist parameters are params that were previous erroneously saved by the model
        # care should be taken in future versions that if batch_norm on the first rnn is required
        # that it be named something else
        blacklist = ['rnns.0.batch_norm.module.weight', 'rnns.0.batch_norm.module.bias',
                     'rnns.0.batch_norm.module.running_mean', 'rnns.0.batch_norm.module.running_var']
        for x in blacklist:
            if x in package['state_dict']:
                del package['state_dict'][x]
        model.load_state_dict(package['state_dict'])
        for x in model.rnns:
            x.flatten_parameters()

        if gpu>=0:
            model = model.cuda()
        #if cuda:
#            model = torch.nn.DataParallel(model).cuda()
        return model

    @classmethod
    def load_model_package(cls, package, gpu=-1):
        model = cls(rnn_hidden_size=package['rnn_size'], rnn_layers=package['rnn_layers'],rnn_type=supported_rnns[package['rnn_type']],
                    map=package['cnn_map'], stride = package['cnn_stride'], kernel_sz=package['cnn_kernel'], cnn_layers=package['cnn_layers'],
                    labels=package['labels'],
                    )
        model.load_state_dict(package['state_dict'])
        if(gpu>=0):
            model = model.cuda()
        #if cuda:
#            model = torch.nn.DataParallel(model).cuda()
        return model

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
                  cer_results=None, wer_results=None, avg_loss=None, meta=None):
        #model_is_cuda = next(model.parameters()).is_cuda
        #pdb.set_trace()
        #model = model.module if model_is_cuda else model
        #model = model._modules if model_is_cuda else model

        package = {
            'version': model._version,
            'rnn_size': model.rnn_size,
            'rnn_layers': model.rnn_layers,
            'cnn_map': model.cnn_map,
            'cnn_kernel': model.cnn_kernel,
            'cnn_stride': model.cnn_stride,
            'cnn_layers': model.cnn_layers,
            'rnn_type': supported_rnns_inv.get(model.rnn_type, model.rnn_type.__name__.lower()),
            'labels': model._labels,
            'state_dict': model.state_dict()
       }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1  # increment for readability
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
        if meta is not None:
            package['meta'] = meta
        return package

    @staticmethod
    def get_labels(model):
        """
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._labels if model_is_cuda else model._labels
        """
        return model._labels

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params


    @staticmethod
    def get_audio_conf(model):
        return model._audio_conf


    @staticmethod
    def get_meta(model):
        model_is_cuda = next(model.parameters()).is_cuda
        m = model.module if model_is_cuda else model
        meta = {
            "version": m._version,
            "rnn_size": m.rnn_size,
            "rnn_layers": m.rnn_layers,
            "cnn_map": m.cnn_map,
            "cnn_kernel": m.cnn_kernel,
            "cnn_stride": m.cnn_stride,
            "cnn_layers": m.cnn_layers,
            "rnn_type": supported_rnns_inv[m.rnn_type]
        }
        return meta

class var_mask(nn.Module):
    def __init__(self):
        super(var_mask, self).__init__()

    def forward(self, X, mask):
        mask_sum = mask.data.sum()
        if(mask.data[0][0][0] == 0): # data_as_0 = True
            nElement = mask.data.nelement() - mask_sum
            nElement = nElement.item()

        #pdb.set_trace()
        mask = mask.unsqueeze(2)
        X.masked_fill(mask, 0)
        var = torch.var(X, dim=3)
        loss = var.sum()/nElement



        return loss, nElement

class L1Loss_mask(nn.Module):
    def __init__(self):
        super(L1Loss_mask, self).__init__()

    def forward(self, input, target, mask):
        mask_sum = mask.data.sum()
        if(mask.data[0][0][0] == 0): # data_as_0 = True
            nElement = mask.data.nelement() - mask_sum
            nElement = nElement.item()

        #pdb.set_trace()
        err = torch.abs(input-target)
        if(err.dim() == 4):
            mask = mask.unsqueeze(2)
        elif(err.dim() == 2):
            mask = mask.squeeze()
        #pdb.set_trace()
        err.masked_fill(mask, 0)

        loss = err.sum()/nElement
        return loss, nElement

class get_SNRout(nn.Module):
    def __init__(self, log_domain=False):
        super(get_SNRout, self).__init__()
        self.log_domain=log_domain
        print('(get_SNRout) log_domain =')
        print(log_domain)

    def forward(self, input, target, mask):
        if(self.log_domain):
            input = input.exp()-1
            target = target.exp()-1

        err = torch.pow(input-target, 2)
        if(err.dim() == 4):
            mask = mask.unsqueeze(2)
        elif(err.dim() == 2):
            mask = mask.squeeze()

        err.masked_fill(mask, 0)

        #SNRout = 10*torch.log(torch.pow(target, 2).sum()/(err.sum() + 1e-8))/2.3026  # 2.3026 = log(10)  # WRONG !!!
        if(err.dim() == 3):
            SNRout = 10 * torch.log10(torch.pow(target, 2).sum(2).sum(1) / (err.sum(2).sum(1) + 1e-8))
        elif(err.dim() == 2):
            SNRout = 10 * torch.log10(torch.pow(target, 2).sum(1) / (err.sum(1) + 1e-8))

        return SNRout


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, input, target, mask=None):
        nElement = target.data.nelement()
        err = torch.abs(input-target)
        loss = err.sum()/nElement

        return loss, nElement

class L1Loss_perfreq_mask(nn.Module):
    def __init__(self):
        super(L1Loss_perfreq_mask, self).__init__()

    def forward(self, input, target, mask):
        mask_sum = mask.data.sum()
        if(mask.data[0][0][0] == 0): # data_as_0 = True
            nElement = mask.data.nelement() - mask_sum
            nElement = nElement.item()

        err = torch.abs(input-target)

        if(input.dim() == 4):
            mask = mask.unsqueeze(2)
        err.masked_fill(mask, 0)

        if(self.training):
            loss_per_freq = err.sum(-1).sum(0)/nElement
            loss_r, loss_i = torch.split(loss_per_freq, int(loss_per_freq.size(0)/2), dim=0)
            loss = loss_r + loss_i
        else:
            loss = err.sum()/nElement

        return loss, nElement


class L1Loss_mask_expand(nn.Module):
    def __init__(self):
        super(L1Loss_mask_expand, self).__init__()

    def forward(self, input, target, mask):
        # input: NxFxnCHxT
        # target: NxFxnCH

        N,F,CH,T = input.size()

        mask = mask.expand(N, F, T)
        mask = mask.unsqueeze(2)
        mask = mask.expand(N, F, CH, T)

        mask_sum = mask.data.sum()
        nElement = mask.data.nelement() - mask_sum
        nElement = nElement.item()

        #pdb.set_trace()
        target = target.unsqueeze(0)
        target = target.expand(N, F, CH)
        target = target.unsqueeze(3)
        target = target.expand(N, F, CH, T)
        err = torch.abs(input - target)
        #pdb.set_trace()
        err.masked_fill(mask, 0)
        #pdb.set_trace()
        loss = err.sum()/nElement
        return loss, nElement
