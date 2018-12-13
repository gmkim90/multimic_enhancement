#from torch.utils.data import Dataset # in loader_functions.py
#from torch.utils.data import DataLoader # in loader_functions.py
#from torch.utils.data.sampler import Sampler # in loader_functions.py

from loader_functions import FeatDataset, FeatSampler, FeatLoader, FeatLoader_paired, FeatLoader_multiCH_paired, FeatLoader_BSS_paired, FeatLoader_BSE_paired, FeatLoader_linear2mel_paired
import pdb

class DataLoader():
    def __init__(self, batch_size, paired = False, tr_cl_manifest="", tr_ny_manifest="", trsub_manifest="", val_manifest="", val2_manifest="", labels=None,
                 include_noise=False, noise_info=None, include_spk=False, spk_info=None, multiCH=False, db= '', BSS = False, BSE = False, linear_to_mel=False):
        self.batch_size = batch_size
        self.labels = labels
        if(paired):
            if(BSS):
                self.Loader = FeatLoader_BSS_paired
            elif(linear_to_mel):
                self.Loader = FeatLoader_linear2mel_paired
            elif(BSE):
                self.Loader = FeatLoader_BSE_paired
            elif(multiCH):
                self.Loader = FeatLoader_multiCH_paired
            else:
                self.Loader = FeatLoader_paired
        else:
            self.Loader = FeatLoader

        if(multiCH):
            CH_sampling = False
        else:
            CH_sampling = True

        # Define dataset, sampler (only for training), and loader
        # iter() in loader allows using next()
        if(len(tr_cl_manifest) > 0):
            self.tr_cl_ds = FeatDataset(manifest=tr_cl_manifest, labels=labels)
            self.tr_cl_sp = FeatSampler(self.tr_cl_ds, batch_size=batch_size)
            self.tr_cl_dl = iter(self.Loader(self.tr_cl_ds, num_workers=0, batch_sampler=self.tr_cl_sp))

        if(len(tr_ny_manifest) > 0):
            self.tr_ny_ds = FeatDataset(manifest=tr_ny_manifest, labels=labels, include_noise=include_noise, noise_info=noise_info, db=db,
                                        include_spk=include_spk, spk_info=spk_info, CH_sampling=CH_sampling)
            self.tr_ny_sp = FeatSampler(self.tr_ny_ds, batch_size=batch_size)
            self.tr_ny_dl = iter(self.Loader(self.tr_ny_ds, num_workers=0, batch_sampler=self.tr_ny_sp))

        if(len(trsub_manifest) > 0):
            self.trsub_ds = FeatDataset(manifest=trsub_manifest, labels=labels, include_noise=include_noise, noise_info=noise_info, db=db,
                                        include_spk=include_spk, spk_info=spk_info, CH_sampling=CH_sampling)
            self.trsub_dl = iter(self.Loader(self.trsub_ds, batch_size=batch_size))

        if(len(val_manifest) > 0):
            self.val_ds = FeatDataset(manifest=val_manifest, labels=labels, include_noise=include_noise, noise_info=noise_info, db=db, CH_sampling=CH_sampling)
            self.val_dl = iter(self.Loader(self.val_ds, batch_size=batch_size))

        if(len(val2_manifest) > 0):
            self.val2_ds = FeatDataset(manifest=val2_manifest, labels=labels, include_noise=include_noise, noise_info=noise_info, db=db, CH_sampling=CH_sampling)
            self.val2_dl = iter(self.Loader(self.val2_ds, batch_size=batch_size))


    def next(self, cl_ny='', type=''):
        if(cl_ny == 'ny'):
            if(type == 'train'):
                loader = self.tr_ny_dl
            elif(type == 'trsub'):
                loader = self.trsub_dl
            elif(type == 'val'):
                loader = self.val_dl
            elif(type == 'val2'):
                loader = self.val2_dl
        elif(cl_ny == 'cl'):
            if(type == 'train'):
                loader = self.tr_cl_dl

        #pdb.set_trace()
        try:
            data_list = loader.next()
        except StopIteration:
            if (cl_ny == 'ny'):
                if (type == 'train'):
                    self.tr_ny_sp.shuffle()
                    self.tr_ny_dl = iter(self.Loader(self.tr_ny_ds, num_workers=0, batch_sampler=self.tr_ny_sp))
                    loader = self.tr_ny_dl
                elif (type == 'trsub'):
                    self.trsub_dl = iter(self.Loader(self.trsub_ds, batch_size=self.batch_size, num_workers=0))
                    loader = self.trsub_dl
                elif (type == 'val'):
                    self.val_dl = iter(self.Loader(self.val_ds, batch_size=self.batch_size, num_workers=0))
                    loader = self.val_dl
                elif (type == 'val2'):
                    self.val2_dl = iter(self.Loader(self.val2_ds, batch_size=self.batch_size, num_workers=0))
                    loader = self.val2_dl
                    loader = self.te_dl
            elif (cl_ny == 'cl'):
                if (type == 'train'):
                    self.tr_cl_sp.shuffle()
                    self.tr_cl_dl = iter(self.Loader(self.tr_cl_ds, num_workers=0, batch_sampler=self.tr_cl_sp))
                    loader = self.tr_cl_dl

            data_list = loader.next()

        return data_list