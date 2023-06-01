import yaml
import os
import numpy as np
import torch
from sklearn.model_selection import KFold
from monai import transforms
from mri_dataset import MRIDataset


def read_config(name_config, exp_folder="./exp_configs"):
    with open(os.path.join(exp_folder, name_config), "r") as stream:
        try:
            config_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None
    return config_params


def get_dataloder(dataset_dict, config_params, transformation=None, mode='kfold'):

    def make_tensors(set_subj):
        X = []
        y = []
        for subj in set_subj:
            for tensor, label, _, _ in dataset_dict[subj]:
                X.append(tensor)
                y.append(label)

        return X, y
    
    all_keys_subj = np.sort(list(dataset_dict.keys()))
    
    mri_dataloader_train = None
    mri_dataloader_test = None
    
    if mode == 'kfold':
        kfold = KFold(n_splits=config_params['training_params']['count_folds_splits'],
                      random_state=config_params['global_params']['random_state'],
                      shuffle=True)
        for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_subj)):

            if config_params['training_params']['verbose_train']:
                print(f"Number fold: {i+1}")

            X_train, y_train = make_tensors(all_keys_subj[train_idx])
            X_test, y_test = make_tensors(all_keys_subj[test_idx])

            if config_params['training_params']['transormation']:
                mri_dataset_train = MRIDataset(X_train, y_train, transformation)
            else:
                mri_dataset_train = MRIDataset(X_train,
                                               y_train,
                                               transforms.Compose([
                                                   transforms.NormalizeIntensity(nonzero=True, channel_wise=True),
                                               ]))

            mri_dataset_test = MRIDataset(X_test, 
                                          y_test, 
                                          transforms.Compose([
                                                   transforms.NormalizeIntensity(nonzero=True, channel_wise=True),
                                          ]))

            mri_dataloader_train = torch.utils.data.DataLoader(mri_dataset_train,
                                                               batch_size=config_params['training_params']['batch_size'],
                                                               shuffle=True)
            mri_dataloader_test = torch.utils.data.DataLoader(mri_dataset_test, batch_size=1, shuffle=False)
            
            yield mri_dataloader_train, mri_dataloader_test
    else:
        X_train, y_train = make_tensors(all_keys_subj)

        if config_params['training_params']['transormation']:
            mri_dataset_train = MRIDataset(X_train, y_train, transformation)
        else:
            mri_dataset_train = MRIDataset(X_train,
                                           y_train,
                                           transforms.Compose([
                                               transforms.NormalizeIntensity(nonzero=True, channel_wise=True),
                                           ]))

        mri_dataloader_train = torch.utils.data.DataLoader(mri_dataset_train,
                                                           batch_size=config_params['training_params']['batch_size'],
                                                           shuffle=True)

        yield mri_dataloader_train, mri_dataloader_test