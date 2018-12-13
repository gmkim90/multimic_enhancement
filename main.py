import torch
from config import get_config, decide_expnum
from data_loader import DataLoader
import os
import json
import sys
import pdb

def main(config):
    #prepare_dirs_and_logger(config)

    from data_loader import DataLoader
    if(config.trainer == 'DCE_multiCH'):
        paired = True
        from trainer_DCE_multiCH import Trainer  # 학습 초기화

    if config.gpu >= 0:
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.set_device(config.gpu)

    # Check manually assigned manifest path exists
    if (len(config.tr_cl_manifest) > 0):
        print('manually assinged manifest exists for clean training set. Use it.')
    if (len(config.tr_ny_manifest) > 0):
        print('manually assinged manifest exists for noisy training set. Use it.')
    if (len(config.trsub_manifest) > 0):
        print('manually assinged manifest exists for noisy training SUBset. Use it.')
    if (len(config.val_manifest) > 0):
        print('manually assinged manifest exists for noisy validation set. Use it.')
    if (len(config.val2_manifest) > 0):
        print('manually assinged manifest exists for noisy validation2 set. Use it.')

    # Assign manifest
    if(config.DB_name == 'dereverb_4IR'):
        if(config.multiCH):
            if(paired):
                config.tr_ny_manifest = 'data_sorted/Dereverb_4IR_train_inputSNR.csv'
                config.trsub_manifest = 'data_sorted/Dereverb_4IR_trsub_inputSNR.csv'
                config.val_manifest = 'data_sorted/Dereverb_4IR_val_inputSNR.csv'
            else:
                assert (0), 'NOT IMPLEMENTED YET'
            noise_info = {}  # dummy
            spk_info = {}  # dummy

        print('tr_ny_manifest = ' + config.tr_ny_manifest)
        print('trsub_manifest = ' + config.trsub_manifest)
        print('val_manifest = ' + config.val_manifest)
        print('val2_manifest = ' + config.val2_manifest)

        noise_info = {
            'BUS': 0,
            'PED': 1,
            'CAF': 2,
            'STR': 3
        }

    config.noise_info = noise_info
    config.spk_info = spk_info
    with open(config.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))

    data_loader = DataLoader(batch_size = config.batch_size, paired=paired,
                             tr_cl_manifest=config.tr_cl_manifest, tr_ny_manifest=config.tr_ny_manifest, trsub_manifest=config.trsub_manifest,
                             val_manifest=config.val_manifest, val2_manifest=config.val2_manifest, labels=labels,
                             include_noise=config.noise_clf, noise_info=noise_info,
                             include_spk=config.spk_clf, spk_info=spk_info,
                             multiCH=config.multiCH, BSS=config.BSS, BSE=config.BSE, linear_to_mel=config.linear_to_mel,
                             db=config.DB_name)

    if(config.mode == 'train'):
        assert(config.expnum == -1), 'for training, do not specify expnum'
        config.expnum = decide_expnum('logs_only')
    else:
        assert(not config.expnum == -1), 'for test or analysis, please specify expnum'

    if not os.path.exists('logs_only/' + str(config.expnum)):
        os.makedirs('logs_only/' + str(config.expnum))

    if not os.path.exists('models/' + str(config.expnum)):
        os.makedirs('models/' + str(config.expnum))

    trainer = Trainer(config, data_loader)

    torch.manual_seed(config.random_seed)

    print('tr_cl_manifest = ' + config.tr_cl_manifest)
    print('tr_ny_manifest = ' + config.tr_ny_manifest)
    print('trsub_manifest = ' + config.trsub_manifest)
    print('val_manifest = ' + config.val_manifest)
    print('val2_manifest = ' + config.val2_manifest)

    if (config.mode == 'train'):
        trainer.train() # 학습 시작
    elif(config.mode == 'visualize'):
        trainer.visualize()


if __name__ == "__main__":
    config, unparsed = get_config()

    main(config)
