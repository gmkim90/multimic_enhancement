import math
from layer_complex import *

import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch.autograd import Variable
from torch.nn.modules.utils import _single

import pdb

class linearmag2mel(nn.Module):
    def __init__(self, mel_basis):
        super(linearmag2mel, self).__init__()

        self.mel_basis = torch.FloatTensor(mel_basis)
        self.mel_basis = self.mel_basis.unsqueeze(-1).unsqueeze(-1)
        self.mel_basis = Variable(self.mel_basis.cuda())

    def forward(self, magnitude):
        magnitude = magnitude.cuda()
        mfb = torch.nn.functional.conv2d(magnitude, self.mel_basis)

        return mfb


def CBatchMatTwoConst(M, C1, C2):
    # M: NxFxT
    # C1, C2: Constant (special case: single frequency)

    Mr, Mi = torch.split(M, int(M.size(1)/2), dim=1)
    C1r, C1i = torch.split(C1, int(C1.size(0)/2), dim=0)
    C2r, C2i = torch.split(C2, int(C2.size(0)/2), dim=0)

    Y1r = Mr*C1r - Mi*C1i
    Y1i = Mr*C1i + Mi*C1r
    Y2r = Mr*C2r - Mi*C2i
    Y2i = Mr*C2i + Mi*C2r

    Y = torch.cat((Y1r, Y2r, Y1i, Y2i), dim=1)  # real/imag first, CH second

    return Y

def M_to_Mp_extraction(M):
    # M (real/imag): Nx2FxCHxT
    # Mp (magnitude & phase differnece): NxFx(CH + CHc2)xT

    Mr, Mi = torch.split(M, int(M.size(1) / 2), dim=1)
    nCH = M.size(2)
    Mmag = torch.sqrt(torch.pow(Mr, 2) + torch.pow(Mi, 2))
    nComb = int(nCH*(nCH-1)/2)
    Mpdiff = torch.zeros(M.size(0), Mmag.size(0), nComb, M.size(-1))

    count = 0
    for c1 in range(len(nCH)):
        for c2 in range(c1+1, len(nCH)):
            Mpdiff()

class sigmoid_min_max(nn.Module):
    def __init__(self, minval, maxval):
        super(sigmoid_min_max, self).__init__()
        self.minval = minval
        self.maxval = maxval

    def forward(self, input):
        output = Func.sigmoid(input) * (self.maxval-self.minval) + self.minval

        return output

def get_demixW_from_MS(M, S):
    # M: NxFx2xT --> (NTxFx2) (tensor (not variable))
    # S: NxFxT --> (NTxFx1) (tensor (not variable))

    N, F, CH, T = M.size()
    #pdb.set_trace()
    M = M.transpose(2, 3).transpose(1, 2).contiguous().view(N*T, F, CH)
    S = S.transpose(1, 2).contiguous().view(N*T, F).unsqueeze(2)

    Mr, Mi = torch.split(M, int(F / 2), dim=1)
    Sr, Si = torch.split(S, int(F / 2), dim=1)

    Wr = torch.zeros(int(F / 2), CH)
    Wi = torch.zeros(int(F / 2), CH)

    for f in range(Wr.size(0)):
        Mr_f = Mr[:, f]
        Mi_f = Mi[:, f]
        Sr_f = Sr[:, f]
        Si_f = Sr[:, f]

        #W[f] += Sf*Mf.transpose(0, 1)*torch.inverse(CMat)
        MMtinvr, MMtinvi = CMMhinv(Mr_f, Mi_f)
        Mhr, Mhi = Hermitian(Mr_f, Mi_f)
        #MhMMtinvr, MhMMtinvi = CMatMul(Mhr, Mhi, MMtinvr, MMtinvi)
        MhMMtinvr, MhMMtinvi = CMatMul(MMtinvr, MMtinvi, Mhr, Mhi)
        #SMhMMtinvr, SMhMMtinvi = CMatMul(Sr_f, Si_f, MhMMtinvr, MhMMtinvi)
        SMhMMtinvr, SMhMMtinvi = CMatMul(MhMMtinvr, MhMMtinvi, Sr_f, Si_f)

        #pdb.set_trace()
        Wr[f] = SMhMMtinvr.squeeze()
        Wi[f] = SMhMMtinvi.squeeze()

    W = torch.cat((Wr, Wi), dim=0)
    #return Wr, Wi
    return W.cuda()

def get_demixW_from_MS_single_CPU(M, S):
    # M: 2xFxT --> (TxFx2) (tensor (not variable))
    # S: FxT --> (TxFx1) (tensor (not variable))

    CH, F, T = M.size()

    #pdb.set_trace()

    M = M.transpose(0, 2).contiguous().view(T, F, CH)
    S = S.transpose(0, 1).contiguous().view(T, F).unsqueeze(2)

    Mr, Mi = torch.split(M, int(F / 2), dim=1)
    Sr, Si = torch.split(S, int(F / 2), dim=1)

    Wr = torch.zeros(int(F / 2), CH)
    Wi = torch.zeros(int(F / 2), CH)

    for f in range(Wr.size(0)):
        Mr_f = Mr[:, f]
        Mi_f = Mi[:, f]
        Sr_f = Sr[:, f]
        Si_f = Sr[:, f]

        MMtinvr, MMtinvi = CMMhinv(Mr_f, Mi_f, cuda=False)
        Mhr, Mhi = Hermitian(Mr_f, Mi_f)
        MhMMtinvr, MhMMtinvi = CMatMul(MMtinvr, MMtinvi, Mhr, Mhi)
        SMhMMtinvr, SMhMMtinvi = CMatMul(MhMMtinvr, MhMMtinvi, Sr_f, Si_f)

        Wr[f] = SMhMMtinvr.squeeze()
        Wi[f] = SMhMMtinvi.squeeze()

    W = torch.cat((Wr, Wi), dim=0)

    return W

class ComplexDeepDepthwiseCNN(nn.Module):
    def __init__(self, F, nCH, w, H, L, use_pad = False, BN=False, non_linear='relu', eps_freq0=-10,
                 use_sigmoid_min_max=False, output_W_too=False):
        super(ComplexDeepDepthwiseCNN, self).__init__()
        self.F = F
        self.H = H
        self.nCH = nCH
        self.w = w

        self.L = L

        self.output_W_too = output_W_too

        #self.BN = BN


        if(use_sigmoid_min_max):
            self.sigmoid_min_max_R = sigmoid_min_max_R(minval_R, maxval_R)
            self.sigmoid_min_max_I = sigmoid_min_max_I(minval_I, maxval_I)
        else:
            self.sigmoid_min_max_R = None
            self.sigmoid_min_max_I = None

        if(non_linear == 'relu'):
            self.non_linear = Func.relu
        elif(non_linear == 'tanh'):
            self.non_linear = Func.tanh

        if(use_pad):
            nPad = int((w-1)/2)
        else:
            nPad = 0

        #pdb.set_trace()
        self.cnn1 = ComplexDepthwiseConv(F, H, (nCH, w), padding=(0, nPad))
        if(L == 3):
            self.cnn2 = ComplexDepthwiseConv(F, H, (H, w), padding=(0, nPad))
            self.cnn3 = ComplexDepthwiseConv(F, nCH, (H, w), padding=(0, nPad))
        elif(L == 2):
            self.cnn2 = ComplexDepthwiseConv(F, nCH, (H, w), padding=(0, nPad))



        if(BN):
            self.bn1 = ComplexSeqDepthWise(ComplexDepthwiseBatchNorm(H, F, eps_freq0=eps_freq0))
            self.bn2 = ComplexSeqDepthWise(ComplexDepthwiseBatchNorm(H, F, eps_freq0=eps_freq0))
            assert(L == 3)
        else:
            self.bn1 = None
            self.bn2 = None


    def forward(self, x):
    #    pdb.set_trace()
        xr, xi = torch.split(x, int(x.size(1)/2), dim=1)

        #pdb.set_trace()
        h1r, h1i = self.cnn1(xr, xi)
        if(self.bn1):
            h1r, h1i = self.bn1(h1r, h1i)
        h1r = self.non_linear(h1r)
        h1i = self.non_linear(h1i)

        h2r, h2i = self.cnn2(h1r, h1i)
        if(self.bn2):
            h2r, h2i = self.bn2(h2r, h2i)
        if(self.L == 3):
            h2r = self.non_linear(h2r)
            h2i = self.non_linear(h2i)

            h3r, h3i = self.cnn3(h2r, h2i)
            hr = h3r
            hi = h3i
        elif(self.L == 2):
            hr = h2r
            hi = h2i

        # Old version: include average across time
        #T_initial = xr.size(3)

        #demixWr = hr.mean(3, keepdim=True) # NxFxCHx1
        #demixWi = hi.mean(3, keepdim=True) # NxFxCHx1

        #demixWr_expand = demixWr.expand(demixWr.size(0), demixWr.size(1), demixWr.size(2), T_initial)
        #demixWi_expand = demixWi.expand(demixWi.size(0), demixWi.size(1), demixWi.size(2), T_initial)

        # output = demixing weight * input
        #yrr = (demixWr_expand*xr).sum(2)
        #yii = (demixWi_expand*xi).sum(2)
        #yri = (demixWr_expand*xi).sum(2)
        #yir = (demixWi_expand*xr).sum(2)

        # New version
        #pdb.set_trace()
        yrr = (xr*hr).sum(2)
        yri = (xr*hi).sum(2)
        yir = (xi*hr).sum(2)
        yii = (xi*hi).sum(2)

        Yr = yrr - yii
        Yi = yri + yir

        #return Yr, Yi

        out = torch.squeeze(torch.cat((Yr, Yi), dim=1))

        if(self.output_W_too):
            return out, hr, hi
        else:
            return out # singleCH spectrogram


class ComplexDeepDepthwiseCNN_BSS(nn.Module):
    def __init__(self, F, nCH, nSource, w, H, L, use_pad=False, BN=False, non_linear='relu', eps_freq0=-10,
                 use_sigmoid_min_max=False, output_W_too=False):
        super(ComplexDeepDepthwiseCNN_BSS, self).__init__()
        self.F = F
        self.H = H
        self.nCH = nCH
        self.nSource = nSource
        self.w = w

        self.L = L

        self.output_W_too = output_W_too

        # self.BN = BN

        if (use_sigmoid_min_max):
            self.sigmoid_min_max_R = sigmoid_min_max_R(minval_R, maxval_R)
            self.sigmoid_min_max_I = sigmoid_min_max_I(minval_I, maxval_I)

        if (non_linear == 'relu'):
            self.non_linear = Func.relu
        elif (non_linear == 'tanh'):
            self.non_linear = Func.tanh

        if (use_pad):
            nPad = int((w - 1) / 2)
        else:
            nPad = 0

        # pdb.set_trace()
        self.cnn1 = ComplexDepthwiseConv(F, H, (nCH, w), padding=(0, nPad))
        if (L == 3):
            self.cnn2 = ComplexDepthwiseConv(F, H, (H, w), padding=(0, nPad))
            self.cnn3 = ComplexDepthwiseConv(F, nCH*nSource, (H, w), padding=(0, nPad))
        elif (L == 2):
            self.cnn2 = ComplexDepthwiseConv(F, nCH*nSource, (H, w), padding=(0, nPad))

        if (BN):
            self.bn1 = ComplexSeqDepthWise(ComplexDepthwiseBatchNorm(H, F, eps_freq0=eps_freq0))
            self.bn2 = ComplexSeqDepthWise(ComplexDepthwiseBatchNorm(H, F, eps_freq0=eps_freq0))
            assert (L == 3)
        else:
            self.bn1 = None
            self.bn2 = None
        #pdb.set_trace()

    def forward(self, x):
        xr, xi = torch.split(x, int(x.size(1) / 2), dim=1)

        # pdb.set_trace()
        h1r, h1i = self.cnn1(xr, xi)
        if (self.bn1):
            h1r, h1i = self.bn1(h1r, h1i)
        h1r = self.non_linear(h1r)
        h1i = self.non_linear(h1i)

        h2r, h2i = self.cnn2(h1r, h1i)
        if (self.bn2):
            h2r, h2i = self.bn2(h2r, h2i)
        if (self.L == 3):
            h2r = self.non_linear(h2r)
            h2i = self.non_linear(h2i)

            h3r, h3i = self.cnn3(h2r, h2i)
            hr = h3r
            hi = h3i
        elif (self.L == 2):
            hr = h2r
            hi = h2i

        # Old version: include average across time
        # T_initial = xr.size(3)

        # demixWr = hr.mean(3, keepdim=True) # NxFxCHx1
        # demixWi = hi.mean(3, keepdim=True) # NxFxCHx1

        # demixWr_expand = demixWr.expand(demixWr.size(0), demixWr.size(1), demixWr.size(2), T_initial)
        # demixWi_expand = demixWi.expand(demixWi.size(0), demixWi.size(1), demixWi.size(2), T_initial)

        # output = demixing weight * input
        # yrr = (demixWr_expand*xr).sum(2)
        # yii = (demixWi_expand*xi).sum(2)
        # yri = (demixWr_expand*xi).sum(2)
        # yir = (demixWi_expand*xr).sum(2)

        # New version
        # pdb.set_trace()
        xr = xr.unsqueeze(3).expand(xr.size(0), xr.size(1), self.nCH, self.nSource, xr.size(-1))
        xi = xi.unsqueeze(3).expand(xi.size(0), xi.size(1), self.nCH, self.nSource, xi.size(-1))
        hr = hr.view(hr.size(0), hr.size(1), self.nCH, self.nSource, -1)
        hi = hi.view(hi.size(0), hi.size(1), self.nCH, self.nSource, -1)

        yrr = (xr * hr).sum(2)
        yri = (xr * hi).sum(2)
        yir = (xi * hr).sum(2)
        yii = (xi * hi).sum(2)

        Yr = yrr - yii
        Yi = yri + yir

        # return Yr, Yi

        out = torch.squeeze(torch.cat((Yr, Yi), dim=1))

        #pdb.set_trace()
        if (self.output_W_too):
            return out, hr, hi
        else:
            return out  # singleCH spectrogram

class LineartoMel_real_single(nn.Module):
    def __init__(self, nFreqs, nCH, w, H, L, use_pad=False, BN=False, non_linear='relu'):
        super(LineartoMel_real_single, self).__init__()
        self.nFreqs = nFreqs
        self.H = H
        self.nCH = nCH
        self.w = w

        self.L = L

        if(non_linear == 'relu'):
            self.non_linear = Func.relu
        elif(non_linear == 'tanh'):
            self.non_linear = Func.tanh

        if(use_pad):
            nPad = int((w-1)/2)
        else:
            nPad = 0

        self.nMag = nCH
        self.nPhasediff = int(nCH * (nCH - 1) / 2)
        self.nCombination = self.nMag + self.nPhasediff

        self.cnn1 = RealDepthwiseConv(in_channels=nFreqs, out_channels=H, kernel_size = (self.nCombination, w), padding = (0, nPad))
        if(L == 3):
            self.cnn2 = RealDepthwiseConv(in_channels=nFreqs, out_channels=H, kernel_size = (H, w), padding = (0, nPad))
            self.cnn3 = RealDepthwiseConv(in_channels=nFreqs, out_channels=nCH, kernel_size = (H, w), padding = (0, nPad))
        elif(L == 2):
            self.cnn2 = RealDepthwiseConv(in_channels=nFreqs, out_channels=nCH, kernel_size = (H, w), padding = (0, nPad))

        if(BN):
            assert (L == 3), 'L == 2 should be re-implemented'
            self.bn1 = RealSeqDepthWise(nn.BatchNorm2d(H))
            self.bn2 = RealSeqDepthWise(nn.BatchNorm2d(H))
        else:
            self.bn1 = None
            self.bn2 = None

    def forward(self, M):
        Mmagphsdiff = M[0]
        mfb = M[1]
        #Mmag = Mmagphsdiff[:, :, 0:self.nMag]

        #pdb.set_trace()
        h1 = self.cnn1(Mmagphsdiff)
        if(self.bn1):
            h1 = self.bn1(h1)
        h1 = self.non_linear(h1)

        h2 = self.cnn2(h1)
        if(self.bn2):
            h2 = self.bn2(h2)
        if(self.L == 3):
            h2 = self.non_linear(h2)
            h3 = self.cnn3(h2)

            W = h3

        elif(self.L == 2):
            W = h2

        W_mel = W.sum(1)

        lmfb = torch.log(1+mfb*W_mel).sum(1)

        return lmfb # singleCH spectrogram

class LineartoMel_real(nn.Module):
    def __init__(self, F,  melF_to_linearFs, nCH, w, H, L, use_pad=False, BN=False, non_linear='relu', output_W_too=False):
        super(LineartoMel_real, self).__init__()
        self.F = F
        self.melF_to_linearFs = melF_to_linearFs
        self.H = H
        self.nCH = nCH
        self.w = w
        self.output_W_too = output_W_too

        self.L = L

        if(non_linear == 'relu'):
            self.non_linear = Func.relu
        elif(non_linear == 'tanh'):
            self.non_linear = Func.tanh

        if(use_pad):
            nPad = int((w-1)/2)
        else:
            nPad = 0

        self.nMag = nCH
        self.nPhasediff = int(nCH * (nCH - 1) / 2)
        self.nCombination = self.nMag + self.nPhasediff

        nMel = melF_to_linearFs.size(0)
        self.mel2linear_binaryW = torch.FloatTensor(nMel, F, 1, 1).zero_()
        for m in range(nMel):
            fstart = melF_to_linearFs[m][0]
            fend = melF_to_linearFs[m][1]
            self.mel2linear_binaryW[m, fstart:fend+1] = 1
        self.mel2linear_binaryW = self.mel2linear_binaryW.cuda()

        self.cnn1 = RealDepthwiseConv(in_channels=F, out_channels=H, kernel_size = (self.nCombination, w), padding = (0, nPad))
        if(L == 3):
            self.cnn2 = RealDepthwiseConv(in_channels=F, out_channels=H, kernel_size = (H, w), padding = (0, nPad))
            self.cnn3 = RealDepthwiseConv(in_channels=F, out_channels=nCH, kernel_size = (H, w), padding = (0, nPad))
        elif(L == 2):
            self.cnn2 = RealDepthwiseConv(in_channels=F, out_channels=nCH, kernel_size = (H, w), padding = (0, nPad))

        if(BN):
            self.bn1 = RealSeqDepthWise(nn.BatchNorm2d(H))
            if(self.L == 3):
                self.bn2 = RealSeqDepthWise(nn.BatchNorm2d(H))
            else:
                self.bn2 = None
        else:
            self.bn1 = None
            self.bn2 = None


    def forward(self, M):
        Mmagphsdiff = M[0]
        mfb = M[1]

        h1 = self.cnn1(Mmagphsdiff)
        if(self.bn1):
            h1 = self.bn1(h1)
        h1 = self.non_linear(h1)

        h2 = self.cnn2(h1)
        if(self.bn2):
            h2 = self.bn2(h2)
        if(self.L == 3):
            h2 = self.non_linear(h2)
            h3 = self.cnn3(h2)

            W = h3

        elif(self.L == 2):
            W = h2

        W_mel = Func.conv2d(W, self.mel2linear_binaryW)

        lmfb = torch.log(1+(mfb*W_mel).sum(2))

        if(not self.output_W_too):
            return lmfb # singleCH spectrogram
        else:
            return [lmfb, W_mel, W]

class RealDeepDepthwiseCNN(nn.Module):
    def __init__(self, F, nCH, w, H, L, use_pad = False, BN=False, non_linear='relu'):
        super(RealDeepDepthwiseCNN, self).__init__()
        self.F = F
        self.H = H
        self.nCH = nCH
        self.w = w

        self.L = L

        if(non_linear == 'relu'):
            self.non_linear = Func.relu
        elif(non_linear == 'tanh'):
            self.non_linear = Func.tanh

        if(use_pad):
            nPad = int((w-1)/2)
        else:
            nPad = 0

        #pdb.set_trace()
        self.cnn1 = RealDepthwiseConv(in_channels=F, out_channels=H, kernel_size = (nCH*2, w), padding = (0, nPad))
        if(L == 3):
            self.cnn2 = RealDepthwiseConv(in_channels=F, out_channels=H, kernel_size = (H, w), padding = (0, nPad))
            self.cnn3 = RealDepthwiseConv(in_channels=F, out_channels=nCH*2, kernel_size = (H, w), padding = (0, nPad))
        elif(L == 2):
            self.cnn2 = RealDepthwiseConv(in_channels=F, out_channels=nCH*2, kernel_size = (H, w), padding = (0, nPad))

        if(BN):
            assert (L == 3), 'L == 2 should be re-implemented'
            self.bn1 = RealSeqDepthWise(nn.BatchNorm2d(H))
            self.bn2 = RealSeqDepthWise(nn.BatchNorm2d(H))
        else:
            self.bn1 = None
            self.bn2 = None


    def forward(self, x):
        xr, xi = torch.split(x, int(x.size(1)/2), dim=1)
        x0 = torch.cat((xr, xi), dim=2)

        #pdb.set_trace()
        h1 = self.cnn1(x0)
        if(self.bn1):
            h1 = self.bn1(h1)
        h1 = self.non_linear(h1)

        h2 = self.cnn2(h1)
        if(self.bn2):
            h2 = self.bn2(h2)
        if(self.L == 3):
            h2 = self.non_linear(h2)
            h3 = self.cnn3(h2)

            W = h3

        elif(self.L == 2):
            W = h2

        #pdb.set_trace()
        y = x0*W


        out = torch.cat((y[:, :, 0:2].sum(2), y[:, :, 2:4].sum(2)), dim=1)

        return out # singleCH spectrogram

class ComplexDeepConv1d(nn.Module):  # single frequency learning
    def __init__(self, nCH, w, H, L, output_is_demixW=False, use_pad = False, non_linear='relu', include_nonlinear=True, no_demixW=False, BN=False,
                 use_sigmoid_min_max=False, minval_R = 100.0, maxval_R = -100.0, minval_I = 100.0, maxval_I = -100.0, init_method='complex'):
        super(ComplexDeepConv1d, self).__init__()
        self.H = H
        self.nCH = nCH
        self.w = w

        self.L = L

        self.BN = BN
        self.include_nonlinear = include_nonlinear
        self.no_demixW = no_demixW
        if(non_linear == 'relu'):
            self.non_linear = Func.relu
        elif(non_linear == 'tanh'):
            self.non_linear = Func.tanh

        if(use_pad):
            nPad = int((w-1)/2)
        else:
            nPad = 0

        self.cnn1 = ComplexConv1d(nCH, H, w, padding=nPad, init_method=init_method)
        if(L == 3):
            self.cnn2 = ComplexConv1d(H, H, w, padding = nPad, init_method=init_method)
            #self.cnn3 = ComplexConv1d(H, 1, w, padding = nPad, init_method=init_method)
            self.cnn3 = ComplexConv1d(H, 2, w, padding=nPad, init_method=init_method)
        elif(L == 2):
            #self.cnn2 = ComplexConv1d(H, 1, w, padding=nPad, init_method=init_method)
            self.cnn2 = ComplexConv1d(H, 2, w, padding=nPad, init_method=init_method)

        self.output_is_demixW = output_is_demixW

        if(use_sigmoid_min_max):
            self.sigmoid_min_max_R = sigmoid_min_max(minval_R, maxval_R)
            self.sigmoid_min_max_I = sigmoid_min_max(minval_I, maxval_I)
        else:
            self.sigmoid_min_max_R = None
            self.sigmoid_min_max_I = None


        if(BN):
            assert (L == 3), 'L == 2 should be re-implemented'
            self.bn1 = ComplexSeqWise(ComplexBatchNorm(H))
            self.bn2 = ComplexSeqWise(ComplexBatchNorm(H))
        else:
            self.bn1 = None
            self.bn2 = None


    def forward(self, x):
    #    pdb.set_trace()
        xr, xi = torch.split(x, int(x.size(1)/2), dim=1)

        #pdb.set_trace()
        h1r, h1i = self.cnn1(xr, xi)
        if(self.bn1):
            h1r, h1i = self.bn1(h1r, h1i)

        if(self.include_nonlinear):
            h1r = self.non_linear(h1r)
            h1i = self.non_linear(h1i)

        h2r, h2i = self.cnn2(h1r, h1i)

        if (self.bn2):
            h2r, h2i = self.bn2(h2r, h2i)

        if(self.L == 3):
            if(self.include_nonlinear):
                h2r = self.non_linear(h2r)
                h2i = self.non_linear(h2i)

            h3r, h3i = self.cnn3(h2r, h2i)
            hr = h3r
            hi = h3i
        elif(self.L == 2):
            hr = h2r
            hi = h2i

        #pdb.set_trace()

        if(self.no_demixW):
            out_r = hr.sum(1).unsqueeze(1)
            out_i = hi.sum(1).unsqueeze(1)

            if (self.sigmoid_min_max_R):
                out_r = self.sigmoid_min_max_R(out_r)
                out_i = self.sigmoid_min_max_I(out_i)

            out = torch.cat((out_r, out_i), dim=1)
        else:
            if(not self.output_is_demixW):
                # old version: average over time
                '''
                T_initial = xr.size(-1)
                
                demixWr = hr.mean(2, keepdim=True) # NxCHx1
                demixWi = hi.mean(2, keepdim=True) # NxCHx1

                demixWr_expand = demixWr.expand(demixWr.size(0), demixWr.size(1), T_initial)
                demixWi_expand = demixWi.expand(demixWi.size(0), demixWi.size(1), T_initial)

                # output = demixing weight * input
                yrr = (demixWr_expand*xr).sum(1)
                yii = (demixWi_expand*xi).sum(1)
                yri = (demixWr_expand*xi).sum(1)
                yir = (demixWi_expand*xr).sum(1)
                '''

                # new version
                yrr = (xr*hr).sum(1)
                yri = (xr*hi).sum(1)
                yir = (xi*hr).sum(1)
                yii = (xi*hi).sum(1)

                Yr = yrr - yii
                Yi = yri + yir

                out = torch.squeeze(torch.cat((Yr.unsqueeze(1), Yi.unsqueeze(1)), dim=1))

            else:
                out = torch.cat((hr, hi), dim=1) # (NxFxCHxTfinal), (NxFxCHxTfinal) --> (Nx2FxCHxTfinal)

        return out # singleCH spectrogram



class ComplexMLP(nn.Module):  # single frequency learning
    def __init__(self, nCH, H, CW=9, init_method='real'):
        super(ComplexMLP, self).__init__()
        self.H = H
        self.nCH = nCH

        self.linear1 = ComplexLinear(nCH*CW, H, init_method=init_method)
        self.linear2 = ComplexLinear(H, 1, init_method=init_method)

    def forward(self, x):
        #pdb.set_trace()
        xr, xi = torch.split(x, int(x.size(1)/2), dim=1)

        h1r, h1i = self.linear1(xr, xi)
        h1r = torch.tanh(h1r)
        h1i = torch.tanh(h1i)

        out_r, out_i= self.linear2(h1r, h1i)
        out = torch.cat((out_r, out_i), dim=1)

        return out # singleCH spectrogram

class ComplexConv1d_1layer(torch.nn.Module): # temporary, for debug
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, init_method='real', bias=True):
        super(ComplexConv1d_1layer, self).__init__()
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
            print('initialize Wr, Wi by complex rayleigh method')
            fanin = self.in_channels // self.groups
            for s in self.kernel_size: fanin *= s
            complex_rayleigh_init(self.Wreal, self.Wimag, fanin)
        elif(init_method == 'real'):
            print('initialize Wr, Wi by real kaiming normal method')
            torch.nn.init.kaiming_normal_(self.Wreal, mode='fan_out', nonlinearity='relu')
            torch.nn.init.kaiming_normal_(self.Wimag, mode='fan_out', nonlinearity='relu')

        if self.Breal is not None and self.Bimag is not None:
            self.Breal.data.zero_()
            self.Bimag.data.zero_()

    def init_weight_by_transfer(self, h1, h2):

        H = torch.FloatTensor([[h1[0], h1[1]], [h2[0], h2[1]], [-h1[1], h1[0]], [-h2[1], h2[0]]])
        W_tmp = torch.mm(torch.inverse(torch.mm(H.transpose(0, 1), H)), H.transpose(0, 1))
        Wr = torch.FloatTensor(W_tmp.size(0), 1)
        Wi = torch.FloatTensor(W_tmp.size(0), 1)

        Wr[0] = W_tmp[0][0]
        Wr[1] = W_tmp[0][1]
        Wi[0] = W_tmp[1][0]
        Wi[0] = W_tmp[1][1]

        self.Wreal.data.copy_(Wr)
        self.Wimag.data.copy_(Wi)
    def forward(self, x):
        #pdb.set_trace()
        xr, xi = torch.split(x, int(x.size(1)/2), dim=1)

        yrr = torch.nn.functional.conv1d(xr, self.Wreal, self.Breal, self.stride, self.padding, self.dilation, self.groups)
        yri = torch.nn.functional.conv1d(xr, self.Wimag, self.Bimag, self.stride, self.padding, self.dilation, self.groups)
        yir = torch.nn.functional.conv1d(xi, self.Wreal, None, self.stride, self.padding, self.dilation, self.groups)
        yii = torch.nn.functional.conv1d(xi, self.Wimag, None, self.stride, self.padding, self.dilation, self.groups)

        yr = yrr - yii
        yi = yri + yir

        y = torch.cat((yr, yi), dim=1)
        #return yrr - yii, yri + yir
        return y


