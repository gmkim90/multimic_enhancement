#-*- coding: utf-8 -*-
import argparse
import os
import pdb

def str2bool(v):
    return v.lower() in ('true', '1')

def decide_expnum(logs_dir):
    expnum = 1
    while os.path.exists(logs_dir + '/' + str(expnum)):
        expnum = expnum + 1
    print('this EXPNUM = ' + str(expnum))
    return expnum

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Basic options
parser.add_argument('--trainer', type=str, default = 'AAS')
parser.add_argument('--mode', type=str, default = 'train', help = 'train | test | visualize')
parser.add_argument('--simul_real', type=str, default = 'real', help = 'simul | real | simulreal')
parser.add_argument('--DB_name', type=str, default='librispeech', help = 'librispeech | chime')
parser.add_argument('--expnum', type=int, default=-1)
parser.add_argument('--gpu', default=-1, type=int)
parser.add_argument('--print_every', type = int, default=100)
parser.add_argument('--load_path', type = str, default='')
parser.add_argument('--ASR_path', type = str, default='')

# Noise label
#parser.add_argument('--include_false_noise_label', type = str2bool, default=False)
#parser.add_argument('--false_label_detach', type = str2bool, default=False)
parser.add_argument('--noise_clf', type = str2bool, default=False)
parser.add_argument('--include_zd_to_classifier', type=str2bool, default=False)
parser.add_argument('--noise_clf_adv_coeff', type = float, default=1)

# Speaker label
parser.add_argument('--spk_clf', type = str2bool, default=False)
parser.add_argument('--spk_clf_adv_coeff', type = float, default=1)
parser.add_argument('--zd_detach', type=str2bool, default=False)

# General
parser.add_argument('--do_recon', type=str2bool, default=False)
parser.add_argument('--to_dec_zi_only', type=str2bool, default=False)

# Disentangle FoV
parser.add_argument('--disentangle_FoV', type=str2bool, default=False)

# Decoding option (used in test())
parser.add_argument('--use_enhancement', type = str2bool, default=False) # used in test() mode decide whether apply enhancement module or not.
parser.add_argument('--output_path', type = str, default='') # used in test() mode

# Manifest list
parser.add_argument('--tr_cl_manifest', default='')
parser.add_argument('--tr_ny_manifest', default='')
parser.add_argument('--trsub_manifest', default='')
parser.add_argument('--val_manifest', default='')
parser.add_argument('--val2_manifest', default='')

parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
parser.add_argument('--labels_path', default='labels.json', help='Contains all characters for transcription')

# Speech enhancement architecture
parser.add_argument('--nFeat_in', default=40, type=int) # this may vary depending on augmented features
parser.add_argument('--nFeat_out', default=40, type=int)
parser.add_argument('--rnn_size', default=500, type=int, help='Hidden size of RNNs')
parser.add_argument('--rnn_layers', default=4, type=int, help='Number of RNN layers')
parser.add_argument('--rnn_type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')

# Multi-channel option
parser.add_argument('--multiCH', default=False, type=str2bool, help = 'use for multiCH setting')
parser.add_argument('--nCH', default=5, type=int, help = 'use for multiCH setting')
parser.add_argument('--divide_real_and_imag', default=False, type=str2bool, help = 'use for multiCH setting')
parser.add_argument('--fix_G_init', default=False, type=str2bool, help = 'use for multiCH setting')
parser.add_argument('--nIter_multiCH', default=-1, type=int)
parser.add_argument('--G_init_path', default='', type=str)
parser.add_argument('--final_iter_input_gt_at_train', default=False, type=str2bool)
parser.add_argument('--output_is_multiCH', default=False, type=str2bool)
parser.add_argument('--iterative_estimate', default=False, type=str2bool)
parser.add_argument('--complexNN', default=False, type=str2bool)
parser.add_argument('--update_by_conjugate_gradient', default=False, type=str2bool)
parser.add_argument('--L_CNN', default=3, type=int)
parser.add_argument('--output_is_demixW', default=False, type=str2bool)
parser.add_argument('--use_pad', default=False, type=str2bool)
parser.add_argument('--single_freq', default=False, type=str2bool)
parser.add_argument('--non_linear', default='relu', type=str)
parser.add_argument('--include_nonlinear', default=True, type=str2bool)
parser.add_argument('--no_demixW', default=False, type=str2bool)
parser.add_argument('--output_bounded', default=False, type=str2bool)
parser.add_argument('--minval_R', default=100.0, type=float)
parser.add_argument('--maxval_R', default=-100.0, type=float)
parser.add_argument('--minval_I', default=100.0, type=float)
parser.add_argument('--maxval_I', default=-100.0, type=float)
#parser.add_argument('--long_window', default=False, type=str2bool) # DEPRECATED
parser.add_argument('--fixed_len', default=False, type=str2bool)
parser.add_argument('--CW', default=9, type=int)
parser.add_argument('--duration', default=5, type=int)
parser.add_argument('--single_minibatch', default=False, type=str2bool)
parser.add_argument('--mini', default=False, type=str2bool)
parser.add_argument('--power_normalize', default=False, type=str2bool)
parser.add_argument('--multiply_RIRfreq', default=False, type=str2bool)
parser.add_argument('--save_histogram', default=False, type=str2bool)
parser.add_argument('--init_method', default='real', type=str, help='complex (complex rayleigh init)|real')
parser.add_argument('--f', default=0, type=int, help='selected frequency (when single_freq=True)')
parser.add_argument('--Winit_by_H', default=False, type=str2bool)
parser.add_argument('--complex_BN', default=False, type=str2bool)
#parser.add_argument('--complex_BN1', default=False, type=str2bool) # first layer
#parser.add_argument('--complex_BN2', default=False, type=str2bool) # second layer
parser.add_argument('--eps_freq0', default=-10, type=float) # second layer
parser.add_argument('--loss_per_freq', default=False, type=str2bool)
parser.add_argument('--real_counterpart', default=False, type=str2bool)
parser.add_argument('--minimize_W_var', default=False, type=str2bool) # equivalent when w_minWvar > 0
parser.add_argument('--w_minWvar', default=0, type=float)
parser.add_argument('--add_noise', default=False, type=str2bool)
parser.add_argument('--BSS', default=False, type=str2bool)
parser.add_argument('--BSE', default=False, type=str2bool)
parser.add_argument('--nSource', default=4, type=int)
parser.add_argument('--variance_normalization', default=False, type=str2bool)
parser.add_argument('--linear_to_mel', default=False, type=str2bool)


# DepthwiseConvolution
parser.add_argument('--nMap_per_F', default=16, type=int)
parser.add_argument('--convW', default=5, type=int)


# encdec archiecture
parser.add_argument('--encdec', default=False, type=str2bool, help = 'use for encoder-decoder architecture given from multispeaker VC')
parser.add_argument('--e_h1', default=128, type=int, help='encoder h1')
parser.add_argument('--e_h2', default=512, type=int, help='encoder h2')
parser.add_argument('--e_h3', default=128, type=int, help='encoder h3')
parser.add_argument('--d_h', default=512, type=int, help='decoder h')

parser.add_argument('--nFFT', default=320, type=int, help = 'use for multiCH setting')
parser.add_argument('--nMel', default=40, type=int, help = 'use for multiCH setting')
parser.add_argument('--fs', default=16000, type=int, help = 'use for multiCH setting')



# Optimization
parser.add_argument('--epochs', default=300, type=int, help='Number of training epochs')
parser.add_argument('--start_iter', default=0, type=int)
parser.add_argument('--max_iter', default = 30000000, type=int)
parser.add_argument('--log_iter', default = 100, type=int)
parser.add_argument('--save_iter', default = 2000, type=int)
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, help='initial learning rate')
parser.add_argument('--w_acoustic', default=1, type=float)
parser.add_argument('--w_adversarial', default=1, type=float)
parser.add_argument('--allow_ASR_update_iter', type = int, default=0)

parser.add_argument('--gamma', type = float, default = 0.5, help = 'began parameter')
parser.add_argument('--lambda_k', type = float, default = 0.001, help = 'began parameter')


parser.add_argument('--optimizer', default='adam', help='adam|sgd')
parser.add_argument('--random_seed', type=int, default=123)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)

def get_config():
    config, unparsed = parser.parse_known_args()
    if(len(unparsed) > 0):
        print(unparsed)
        assert(len(unparsed) == 0), 'length of unparsed option should be 0'
    return config, unparsed
