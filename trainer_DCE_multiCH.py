import os
from glob import glob
from shutil import copyfile
import librosa

from tqdm import trange
import random

from utils import _get_variable, AverageMeter, count_parameters
from decoder import GreedyDecoder
from model import *
from model_complex import *
import scipy.io as sio


class Trainer(object): # the most basic model
    def __init__(self, config, data_loader=None):
        if(config.w_minWvar > 0):
            config.minimize_W_var = True
            self.varLoss = var_mask()

        self.config = config
        self.data_loader = data_loader  # needed for VAE

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.diffLoss = L1Loss_mask() # custom module


        log_domain=False
        if(self.config.linear_to_mel):
            log_domain = True
        self.get_SNRout = get_SNRout(log_domain=log_domain)

        self.valmin_iter = 0
        self.model_dir = 'models/' + str(config.expnum)
        self.log_dir = 'logs_only/' + str(config.expnum)
        self.savename_G = ''
        self.decoder = GreedyDecoder(data_loader.labels)

        self.kt = 0  # used for Proportional Control Theory in BEGAN, initialized as 0
        self.lb = 0.001
        self.conv_measure = 0 # convergence measure

        self.dce_tr = AverageMeter()
        self.dce_val = AverageMeter()

        self.snrout_tr = AverageMeter()
        self.snrout_val = AverageMeter()
        self.snrimpv_tr = AverageMeter()
        self.snrimpv_val = AverageMeter()

        if(config.linear_to_mel):
            self.mel_basis = librosa.filters.mel(self.config.fs, self.config.nFFT, self.config.nMel)
            self.melF_to_linearFs = get_linearF_from_melF(self.mel_basis)
            self.STFT_to_LMFB = STFT_to_LMFB(self.mel_basis, window_change=False)
            self.mag2mfb = linearmag2mel(self.mel_basis)
            
            
        mel_basis_20ms = librosa.filters.mel(self.config.fs, 320, self.config.nMel) # mel_basis will be used only for 20ms window spectrogram
        self.STFT_to_LMFB_20ms = STFT_to_LMFB(mel_basis_20ms, win_size=self.config.nFFT)

        self.F = int(self.config.nFFT/2 + 1)

        self.build_model()
        self.G.loss_stop = 100000
        #self.get_weight_statistic()

        if self.config.gpu >= 0:
            self.G.cuda()

        if len(self.config.load_path) > 0:
            self.load_model()

        if config.mode == 'train':
            self.logFile = open(self.log_dir + '/log.txt', 'w')

    def zero_grad_all(self):
        self.G.zero_grad()

    def build_model(self):
        self.G = LineartoMel_real(F=self.F, melF_to_linearFs=self.melF_to_linearFs, nCH=self.config.nCH, w=self.config.convW,
                                            H=self.config.nMap_per_F, L=self.config.L_CNN,
                                            non_linear=self.config.non_linear, BN=self.config.complex_BN) # 현재 사용중인 모델
        G_name = 'LineartoMel_real'
        
        print('initialized enhancement model as ' + G_name)
        nParam = count_parameters(self.G)
        print('# trainable parameters = ' + str(nParam))

    def load_model(self):
        print("[*] Load models from {}...".format(self.config.load_path))
        postfix = '_valmin'
        paths = glob(os.path.join(self.config.load_path, 'G{}*.pth'.format(postfix)))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.load_path))
            assert(0), 'checkpoint not avilable'

        idxes = [int(os.path.basename(path.split('.')[0].split('_')[-1])) for path in paths]
        if self.config.start_iter <=0 :
            self.config.start_iter = max(idxes)
            if(self.config.start_iter <=0): # if still 0, then raise error
                raise Exception("start iter is still less than 0 --> probably try to load initial random model")

        if self.config.gpu < 0:  #CPU
            map_location = lambda storage, loc: storage
        else: # GPU
            map_location = None

        # Ver2
        print('Load models from ' + self.config.load_path + ', ITERATION = ' + str(self.config.start_iter))
        self.G.load_state_dict(torch.load('{}/G{}_{}.pth'.format(self.config.load_path, postfix, self.config.start_iter), map_location=map_location))

        print("[*] Model loaded")

    def train(self):
        # Setting
        optimizer_g = torch.optim.Adam(self.G.parameters(), lr=self.config.lr, betas=(self.beta1, self.beta2), amsgrad = True)
        
        for iter in trange(self.config.start_iter, self.config.max_iter):
            # Train
            data_list = self.data_loader.next(cl_ny = 'ny', type = 'train')
            inputs, cleans, mask =  data_list[0], data_list[1], data_list[2]  # cleans: NxFxT, mask: Nx1xT

            if(len(data_list) >= 9):
                mixture_magnitude = data_list[7]
                mixture_phsdiff = data_list[8]
                inputs_augmented = torch.cat((torch.log(1 + mixture_magnitude), mixture_phsdiff), dim=2)
                mfb = self.mag2mfb(mixture_magnitude)
                cleans = self.STFT_to_LMFB(cleans)
                    
            if(self.config.linear_to_mel):
                inputs = [_get_variable(inputs_augmented), _get_variable(mfb)]
            else:
                inputs = _get_variable(inputs)
            cleans = _get_variable(cleans)
            mask = _get_variable(mask)

            # forward
            outputs = self.G(inputs) # forward(입력(=[log(magnitude) phase difference]-->출력(=log-mel-filterbank output))

            dce, nElement = self.diffLoss(outputs, cleans, mask) # already normalized inside function
            if(self.config.loss_per_freq):
                if (iter + 1) % self.config.log_iter == 0:
                    for f in range(dce.size(0)):
                        str_loss = "[{}/{}] (train) DCE_{}: {:.7f}".format(iter, self.config.max_iter, f, dce[f].sum().item())
                        self.logFile.write(str_loss + '\n')

                dce = dce.sum() # sum up all the loss

            total_loss = dce

            # backward
            self.zero_grad_all()
            total_loss.backward()

            optimizer_g.step()

            # log
            #pdb.set_trace()
            if (iter+1) % self.config.log_iter == 0:
                #pdb.set_trace()
                str_loss= "[{}/{}] (train) DCE: {:.7f}".format(iter, self.config.max_iter, dce.item())
                print(str_loss)
                self.logFile.write(str_loss + '\n')

                SNRout = self.get_SNRout(outputs, cleans, mask)
                SNRout = SNRout.sum()/cleans.size(0)

                str_loss = "[{}/{}] (train) SNRout: {:.7f}".format(iter, self.config.max_iter, SNRout.item())
                print(str_loss)
                self.logFile.write(str_loss + '\n')

                self.logFile.flush()


            if (iter+1) % self.config.save_iter == 0:
                with torch.no_grad():
                    self.G.eval()
                    self.diffLoss.eval()
                    # Measure performance on training subset
                    self.dce_tr.reset()
                    self.snrout_tr.reset()
                    self.snrimpv_tr.reset()

                    for _ in trange(0, len(self.data_loader.trsub_dl)):
                        data_list = self.data_loader.next(cl_ny='ny', type='trsub')
                        inputs, cleans, mask = data_list[0], data_list[1], data_list[2]
                        if(len(data_list)>=6):
                            targets, input_percentages, target_sizes = data_list[3], data_list[4], data_list[5]
                            if (len(data_list) >= 7):
                                SNRin_1s = _get_variable(data_list[6])
                                if (len(data_list) >= 9):
                                    mixture_magnitude = data_list[7]
                                    mixture_phsdiff = data_list[8]
                                    inputs_augmented = torch.cat((torch.log(1 + mixture_magnitude), mixture_phsdiff), dim=2)
                                    mfb = self.mag2mfb(mixture_magnitude)
                                    cleans = self.STFT_to_LMFB(cleans)

                        cleans, mask = _get_variable(cleans), _get_variable(mask)

                        if(self.config.linear_to_mel):
                            inputs = [_get_variable(inputs_augmented), _get_variable(mfb)]
                        else:
                            inputs = _get_variable(inputs)

                        # Forward (of training subset)
                        outputs = self.G(inputs)

                        dce, nElement = self.diffLoss(outputs, cleans, mask)  # already normalized inside function
                        self.dce_tr.update(dce.item(), nElement)

                        SNRout = self.get_SNRout(outputs, cleans, mask)
                        SNRimprovement = SNRout - SNRin_1s
                        SNRout = SNRout.sum()/cleans.size(0)
                        SNRimprovement = SNRimprovement.sum()/cleans.size(0)

                        self.snrout_tr.update(SNRout.item(), cleans.size(0))
                        self.snrimpv_tr.update(SNRimprovement.item(), cleans.size(0))


                    str_loss= "[{}/{}] (training subset) DCE: {:.7f}".format(iter, self.config.max_iter, self.dce_tr.avg)
                    print(str_loss)
                    self.logFile.write(str_loss + '\n')

                    str_loss = "[{}/{}] (training subset) SNRout: {:.7f}".format(iter, self.config.max_iter, self.snrout_tr.avg)
                    print(str_loss)
                    self.logFile.write(str_loss + '\n')

                    str_loss = "[{}/{}] (training subset) SNRimprovement: {:.7f}".format(iter, self.config.max_iter, self.snrimpv_tr.avg)
                    print(str_loss)
                    self.logFile.write(str_loss + '\n')

                    # Measure performance on validation data
                    self.dce_val.reset()
                    self.wer_val.reset()
                    self.cer_val.reset()
                    self.snrout_tr.reset()
                    self.snrimpv_tr.reset()

                    for _ in trange(0, len(self.data_loader.val_dl)):
                        data_list = self.data_loader.next(cl_ny='ny', type='val')
                        inputs, cleans, mask = data_list[0], data_list[1], data_list[2]
                        if(len(data_list)>=6):
                            targets, input_percentages, target_sizes = data_list[3], data_list[4], data_list[5]
                            if (len(data_list) >= 7):
                                SNRin_1s = _get_variable(data_list[6])
                                if (len(data_list) >= 9):
                                    mixture_magnitude = data_list[7]
                                    mixture_phsdiff = data_list[8]
                                    mfb = self.mag2mfb(mixture_magnitude)
                                    inputs_augmented = torch.cat((torch.log(1 + mixture_magnitude), mixture_phsdiff), dim=2)
                                    cleans = self.STFT_to_LMFB(cleans)

                        cleans, mask = _get_variable(cleans), _get_variable(mask)

                        if(self.config.linear_to_mel):
                            inputs = [_get_variable(inputs_augmented), _get_variable(mfb)]
                        else:
                            inputs = _get_variable(inputs)

                        # Forward (of validation)
                        outputs = self.G(inputs)

                        dce, nElement = self.diffLoss(outputs, cleans, mask)  # already normalized inside function

                        self.dce_val.update(dce.item(), nElement)

                        SNRout = self.get_SNRout(outputs, cleans, mask)
                        SNRimprovement = SNRout - SNRin_1s
                        SNRout = SNRout.sum()/cleans.size(0)
                        SNRimprovement = SNRimprovement.sum()/cleans.size(0)

                        self.snrout_val.update(SNRout.item(), cleans.size(0))
                        self.snrimpv_val.update(SNRimprovement.item(), cleans.size(0))

                    str_loss= "[{}/{}] (validation) DCE: {:.7f}".format(iter, self.config.max_iter, self.dce_val.avg)
                    print(str_loss)
                    self.logFile.write(str_loss + '\n')

                    str_loss = "[{}/{}] (validation) SNRout: {:.7f}".format(iter, self.config.max_iter, self.snrout_val.avg)
                    print(str_loss)
                    self.logFile.write(str_loss + '\n')

                    str_loss = "[{}/{}] (validation) SNRimprovement: {:.7f}".format(iter, self.config.max_iter, self.snrimpv_val.avg)
                    print(str_loss)
                    self.logFile.write(str_loss + '\n')

                    self.G.train() # end of validation
                    self.diffLoss.train()
                    self.logFile.flush()

                    # Save model
                    if (len(self.savename_G) > 0): # do not remove here
                        if os.path.exists(self.savename_G):
                            os.remove(self.savename_G) # remove previous model
                    self.savename_G = '{}/G_{}.pth'.format(self.model_dir, iter)
                    torch.save(self.G.state_dict(), self.savename_G)

                    if(self.G.loss_stop > self.wer_val.avg):
                        self.G.loss_stop = self.wer_val.avg
                        savename_G_valmin_prev = '{}/G_valmin_{}.pth'.format(self.model_dir, self.valmin_iter)
                        if os.path.exists(savename_G_valmin_prev):
                            os.remove(savename_G_valmin_prev) # remove previous model

                        print('save model for this checkpoint')
                        savename_G_valmin = '{}/G_valmin_{}.pth'.format(self.model_dir, iter)
                        copyfile(self.savename_G, savename_G_valmin)
                        self.valmin_iter = iter