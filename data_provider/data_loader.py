import os
import json
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
try:
    from sktime.datasets import load_from_tsfile_to_dataframe
except ImportError:
    load_from_tsfile_to_dataframe = None
import warnings
from utils.augmentation import run_augmentation_single
from datasets import load_dataset
from huggingface_hub import hf_hub_download
warnings.filterwarnings('ignore')

HUGGINGFACE_REPO = "thuml/Time-Series-Library"

class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            df_raw = ds["train"].to_pandas()
            
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            df_raw = ds["train"].to_pandas()

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            split_name = "train" if "train" in ds else list(ds.keys())[0]
            df_raw = ds[split_name].to_pandas()

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           max(0, cut_point - self.label_len):min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        train_path = os.path.join(root_path, "train.csv")
        test_path = os.path.join(root_path, "test.csv")
        label_path = os.path.join(root_path, "test_label.csv")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_df      = pd.read_csv(train_path)
            test_df       = pd.read_csv(test_path)
            test_label_df = pd.read_csv(label_path)
        else:
            ds_data  = load_dataset(HUGGINGFACE_REPO, name="PSM-data")
            ds_label = load_dataset(HUGGINGFACE_REPO, name="PSM-label")
            train_df      = ds_data["train"].to_pandas()
            test_df       = ds_data["test"].to_pandas()
            test_label_df = ds_label[next(iter(ds_label))].to_pandas()

        data = train_df.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        test_data = test_df.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_label_df.values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        train_path = os.path.join(root_path, "MSL_train.npy")
        test_path  = os.path.join(root_path, "MSL_test.npy")
        label_path = os.path.join(root_path, "MSL_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data  = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_train.npy",repo_type="dataset")
            test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_test.npy",repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_test_label.npy",repo_type="dataset")

            train_data  = np.load(train_path)
            test_data   = np.load(test_path)
            test_label  = np.load(label_path)

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data  = self.scaler.transform(test_data)

        self.train = train_data
        self.test  = test_data
        self.test_labels = test_label

        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        train_path = os.path.join(root_path, "SMAP_train.npy")
        test_path  = os.path.join(root_path, "SMAP_test.npy")
        label_path = os.path.join(root_path, "SMAP_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data  = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_train.npy",repo_type="dataset")
            test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_test.npy",repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_test_label.npy",repo_type="dataset")

            train_data  = np.load(train_path)
            test_data   = np.load(test_path)
            test_label = np.load(label_path)

        # 标准化
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data  = self.scaler.transform(test_data)

        self.train = train_data
        self.test  = test_data
        self.test_labels = test_label

        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        train_path = os.path.join(root_path, "SMD_train.npy")
        test_path  = os.path.join(root_path, "SMD_test.npy")
        label_path = os.path.join(root_path, "SMD_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data  = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_train.npy",repo_type="dataset")
            test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_test.npy",repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_test_label.npy",repo_type="dataset")

            train_data  = np.load(train_path)
            test_data   = np.load(test_path)
            test_label = np.load(label_path)
            
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_label
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train2_path = os.path.join(root_path, "swat_train2.csv")
        test_path   = os.path.join(root_path, "swat2.csv")
        if all(os.path.exists(p) for p in [train2_path, test_path]):
            train_data = pd.read_csv(train2_path)
            test_data   = pd.read_csv(test_path)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name="SWaT")
            train_data = ds["train"].to_pandas()
            test_data  = ds["test"].to_pandas()
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def _resolve_ts_path(self, root_path, dataset_name, flag):
        split = "TRAIN" if "train" in str(flag).lower() else "TEST"
        fname = f"{dataset_name}_{split}.ts"
        local = os.path.join(root_path, fname)
        if os.path.exists(local):
            return local
        return hf_hub_download(HUGGINGFACE_REPO, filename=f"{dataset_name}/{fname}", repo_type="dataset")

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        dataset_name = self.args.model_id
        ts_path = self._resolve_ts_path(root_path, dataset_name, flag or "train")

        all_df, labels_df = self.load_single(ts_path)
        return all_df, labels_df

    def load_single(self, filepath):
        if load_from_tsfile_to_dataframe is None:
            raise ImportError(
                'sktime is required for UEA .ts datasets but is not needed for QAR datasets')
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)


class QARFlightDataset(Dataset):
    """
    二分类 QAR 航班时序数据集（按文件夹组织类别: root_path/{0,1}/*.csv）。

    每个样本为一次航班。利用 FLIGHT_PHASE 列定位两个飞行阶段切换点，
    截取固定两段并拼接成原始长度 2000 的序列：
        - 段A: FLIGHT_PHASE 从 2->3 的切换点(阶段3首行 t), 前300 + 后700 = 1000
        - 段B: FLIGHT_PHASE 从 12->13 的切换点(阶段13首行 j), 前1000       = 1000
        原始拼接长度 RAW_SEQ_LEN = 2000
    越界或缺失的片段用 0 填充, 同时生成 (2000,) 的 mask(1=真实数据, 0=填充)。

    随后在 mask>0 的有效行上做逐样本标准化(归一化), 再对 2000 行做
    分桶随机保序降采样到 SEQ_LEN(=200)：2000 分成 200 个桶(每桶 10 行),
    每桶在 mask>0 行内随机选 1 行, 按桶顺序拼接。train/vali/test 全部
    随机降采样(同一文件每个 epoch 看到不同采样点, 起数据增强作用)。

    特征列为除 Time / FLIGHT_PHASE 之外的所有列。

    为兼容 exp/exp_classification.py::_build_model, 暴露:
        max_seq_len  -> args.seq_len  (现为降采样后长度 SEQ_LEN = 200)
        feature_df   -> args.enc_in (取 .shape[1])
        class_names  -> args.num_class (取 len)
    """

    PHASE_COL = 'FLIGHT_PHASE'
    # 段A / 段B 原始窗口（恢复注释值）
    SEG_A_PRE   = 30          # 2->3 前取
    SEG_A_POST  = 70          # 2->3 后取
    SEG_A_LEN   = SEG_A_PRE + SEG_A_POST   # 1000
    SEG_B_PRE   = 0         # 12->13 前取 取零的话是放弃了降落阶段的选择
    SEG_B_LEN   = SEG_B_PRE                  # 1000
    RAW_SEQ_LEN = SEG_A_LEN + SEG_B_LEN     # 2000  原始拼接长度
    # 降采样
    SEQ_LEN        = RAW_SEQ_LEN                        # 降采样后长度（= max_seq_len，传给模型）
    BUCKET_WIDTH   = RAW_SEQ_LEN // SEQ_LEN     # 10    每桶行数
    PHASE_A = (2, 3)
    PHASE_B = (12, 13)
    SPLIT_RATIO = 0.8
    DROP_COLS = ('Time', 'FLIGHT_PHASE')

    def __init__(self, args, root_path, flag='TRAIN'):
        self.args = args
        self.root_path = root_path
        self.flag = str(flag).upper()

        # 扫描类别目录（目录名即类别标签，按整型排序）
        self.class_names = sorted(
            [d for d in os.listdir(root_path)
             if os.path.isdir(os.path.join(root_path, d))],
            key=lambda x: int(x)
        )

        # 收集每个类别下的 CSV 文件并排序
        files_per_class = {}
        for cls in self.class_names:
            files = sorted(glob.glob(os.path.join(root_path, cls, '*.csv')))
            files_per_class[cls] = files
        self.files_per_class = files_per_class

        # 由首个 CSV 表头推断特征列（排除 Time / FLIGHT_PHASE）
        first_file = files_per_class[self.class_names[0]][0]
        header = pd.read_csv(first_file, nrows=0).columns.tolist()
        self.feature_cols = [c for c in header if c not in self.DROP_COLS]

        # 按类别内排序做 8:2 切分，保持类别平衡且无时间穿越
        self.samples = []  # list of (filepath, label_int)
        for cls in self.class_names:
            files = files_per_class[cls]
            k = int(len(files) * self.SPLIT_RATIO)
            chosen = files[:k] if self.flag == 'TRAIN' else files[k:]
            label = int(cls)
            for f in chosen:
                self.samples.append((f, label))

        # 兼容 _build_model 所需属性
        self.max_seq_len = self.SEQ_LEN
        self.feature_df = pd.DataFrame(columns=self.feature_cols)

        print('{} samples: {}'.format(self.flag, len(self.samples)))

    @staticmethod
    def _find_transition(phase, fr, to):
        """返回 phase 中首次出现 "前一行==fr 且 当前行==to" 的索引(即 'to' 首行), 否则 None。"""
        if phase.shape[0] < 2:
            return None
        hits = np.flatnonzero((phase[:-1] == fr) & (phase[1:] == to))
        if hits.size == 0:
            return None
        return int(hits[0] + 1)

    @staticmethod
    def _window(feat, start, end, L):
        """从 feat(T,C) 取 [start,end) 共 L 行; 越界部分补 0, 返回 (out[L,C], mask[L])。"""
        T = feat.shape[0]
        out = np.zeros((L, feat.shape[1]), dtype=np.float32)
        m = np.zeros(L, dtype=np.float32)
        s = max(start, 0)
        e = min(end, T)
        if e > s:
            ds = s - start
            de = ds + (e - s)
            out[ds:de] = feat[s:e]
            m[ds:de] = 1.0
        return out, m

    @staticmethod
    def _instance_norm(x, mask):
        """逐样本标准化: 仅在 mask>0 的有效行上计算均值方差, 填充行保持 0。"""
        valid = mask > 0
        x_norm = np.zeros_like(x)
        if valid.sum() > 1:
            mu = x[valid].mean(axis=0)
            sigma = x[valid].std(axis=0) + 1e-5
            x_norm[valid] = (x[valid] - mu) / sigma
        return x_norm

    @staticmethod
    def _bucket_random_downsample(x, mask, target_len):
        """
        分桶随机保序降采样: x(L, C) -> out(target_len, C), out_mask(target_len)
        L 行分成 target_len 个连续桶(每桶 L//target_len 行), 每桶在 mask>0 的有效行里随机选 1 行;
        整桶 padding 时取 0 向量并 mask=0。保留时间顺序。
        """
        L = x.shape[0]
        bw = L // target_len
        out = np.zeros((target_len, x.shape[1]), dtype=np.float32)
        out_mask = np.zeros(target_len, dtype=np.float32)
        for i in range(target_len):
            s, e = i * bw, (i + 1) * bw
            bucket_m = mask[s:e]
            valid = np.flatnonzero(bucket_m > 0)
            if valid.size > 0:
                pick = valid[np.random.randint(valid.size)]
                out[i] = x[s + pick]
                out_mask[i] = 1.0
        return out, out_mask

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        df = pd.read_csv(filepath)
        # 补空值：CSV 中可能存在 NaN，统一填 0，避免 NaN 进入特征数组污染归一化统计量。
        # FLIGHT_PHASE 的 NaN 也会变 0，不会匹配任何 fr->to 切换点(_find_transition 安全)。
        df = df.fillna(0.0)
        feat = df[self.feature_cols].to_numpy(dtype=np.float32)   # (T, C)
        phase = df[self.PHASE_COL].to_numpy(dtype=np.int64)       # (T,)

        # 段A: 2->3
        t = self._find_transition(phase, *self.PHASE_A)
        if t is not None:
            segA, mA = self._window(feat, t - self.SEG_A_PRE, t + self.SEG_A_POST, self.SEG_A_LEN)
        else:
            segA = np.zeros((self.SEG_A_LEN, feat.shape[1]), dtype=np.float32)
            mA = np.zeros(self.SEG_A_LEN, dtype=np.float32)

        # 段B: 12->13
        j = self._find_transition(phase, *self.PHASE_B)
        if j is not None:
            segB, mB = self._window(feat, j - self.SEG_B_PRE, j, self.SEG_B_LEN)
        else:
            segB = np.zeros((self.SEG_B_LEN, feat.shape[1]), dtype=np.float32)
            mB = np.zeros(self.SEG_B_LEN, dtype=np.float32)

        # 拼接 -> 归一化(在 2000 行 mask>0 上算统计量) -> 分桶随机降采样到 200
        x = np.concatenate([segA, segB], axis=0)       # (2000, C)
        mask = np.concatenate([mA, mB], axis=0)         # (2000,)
        x = self._instance_norm(x, mask)
        # x, mask = self._bucket_random_downsample(x, mask, self.SEQ_LEN)  # (200, C),(200,)

        return (torch.from_numpy(x),
                torch.tensor([label], dtype=torch.long),
                torch.from_numpy(mask))

    def __len__(self):
        return len(self.samples)


class QARFlightDatasetShift(QARFlightDataset):
    """
    在 QARFlightDataset 基础上, 对 FLIGHT_PHASE 2->3 切换点(段A)施加可配置偏移量。

    用途: 数据增强 / 切换点位置敏感性分析。例如:
        shift_a = -100 -> 切换点向前移 100 行(段A 窗口整体前移)
        shift_a = +100 -> 切换点向后移 100 行(段A 窗口整体后移)
        shift_a = [-100, 100] -> 每个 epoch 在 [-100, 100] 内随机采样偏移(数据增强)

    偏移语义: 实际窗口中心 = 原切换点 t + shift; 越界部分仍由 _window 补 0 且 mask=0,
    无越界风险。段B(12->13) 不做偏移, 行为与父类一致。

    参数来源(优先级从高到低):
        1) 构造参数 shift_a
        2) args.phase_a_shift (int 或 [min, max])
        3) 默认 0 (等同父类行为)
    """

    def __init__(self, args, root_path, flag='TRAIN', shift_a=None):
        if shift_a is None:
            shift_a = getattr(args, 'phase_a_shift', 0)
        self.shift_a = self._parse_shift(shift_a)
        self._using_compact_cache = False

        if isinstance(self.shift_a, int):
            shift_tag = 'N{}'.format(abs(self.shift_a)) if self.shift_a < 0 else 'P{}'.format(self.shift_a)
            cache_path = os.path.join(root_path, 'qar_compact_shift{}.npz'.format(shift_tag))
            if os.path.isfile(cache_path):
                cache = np.load(cache_path, allow_pickle=False)
                cached_shift = int(cache['phase_a_shift'][0])
                if cached_shift != self.shift_a:
                    raise ValueError(
                        'Compact QAR cache shift mismatch: requested {}, cache has {}'.format(
                            self.shift_a, cached_shift))

                self.args = args
                self.root_path = root_path
                self.flag = str(flag).upper()
                self._compact_x = cache['x']
                self._compact_mask = cache['mask']
                self._compact_labels = cache['labels'].astype(np.int64, copy=False)
                self.class_names = cache['class_names'].astype(str).tolist()
                self.feature_cols = cache['feature_cols'].astype(str).tolist()
                self.max_seq_len = int(self._compact_x.shape[1])
                self.feature_df = pd.DataFrame(columns=self.feature_cols)

                selected = []
                for cls in self.class_names:
                    class_indices = np.flatnonzero(self._compact_labels == int(cls))
                    split_at = int(len(class_indices) * self.SPLIT_RATIO)
                    chosen = class_indices[:split_at] if self.flag == 'TRAIN' else class_indices[split_at:]
                    selected.extend(chosen.tolist())
                self.samples = selected
                self._using_compact_cache = True
                print('{} samples: {} (compact cache: {})'.format(
                    self.flag, len(self.samples), cache_path))
                return

        super().__init__(args, root_path, flag=flag)

    @staticmethod
    def _parse_shift(sa):
        """规范偏移量配置为 int 或 [min, max]。
        int -> int; "n" -> int; "min,max" -> [min, max]; list/tuple -> 原样。
        """
        if isinstance(sa, str):
            sa = sa.strip()
            if ',' in sa:
                parts = [float(x) for x in sa.split(',')]
                return [int(parts[0]), int(parts[1])]
            return int(float(sa))
        return sa

    def _sample_shift_a(self):
        """返回本次段A偏移量。int -> 固定; list/tuple -> 区间内随机。"""
        sa = self.shift_a
        if isinstance(sa, (list, tuple, np.ndarray)):
            if len(sa) == 1:
                return int(sa[0])
            return int(np.random.randint(int(sa[0]), int(sa[1]) + 1))
        return int(sa)

    def __getitem__(self, idx):
        if self._using_compact_cache:
            cache_idx = self.samples[idx]
            return (
                torch.from_numpy(self._compact_x[cache_idx]),
                torch.tensor([self._compact_labels[cache_idx]], dtype=torch.long),
                torch.from_numpy(self._compact_mask[cache_idx]),
            )

        filepath, label = self.samples[idx]
        df = pd.read_csv(filepath)
        def linear_fill_single_zero(series):
            """
            填充pandas Series中孤立的零值，使用前后值的线性插值。
            假设零值单个出现，并且不在序列边界。
            """
            values = series.to_numpy(copy=True)
            for i in range(1, len(values) - 1):
                if values[i] == 0 and values[i - 1] != 0 and values[i + 1] != 0:
                    values[i] = (values[i - 1] + values[i + 1]) / 2
            return pd.Series(values, index=series.index, name=series.name)
        df['N21'] = linear_fill_single_zero(df['N21'])
        df['N22'] = linear_fill_single_zero(df['N22'])
        df = df.fillna(0.0)
        feat = df[self.feature_cols].to_numpy(dtype=np.float32)   # (T, C)
        phase = df[self.PHASE_COL].to_numpy(dtype=np.int64)       # (T,)

        # 段A: 2->3 (施加偏移)
        t = self._find_transition(phase, *self.PHASE_A)
        if t is not None:
            t = t + self._sample_shift_a()
            segA, mA = self._window(feat, t - self.SEG_A_PRE, t + self.SEG_A_POST, self.SEG_A_LEN)
        else:
            segA = np.zeros((self.SEG_A_LEN, feat.shape[1]), dtype=np.float32)
            mA = np.zeros(self.SEG_A_LEN, dtype=np.float32)

        # 段B: 12->13 (不偏移)
        j = self._find_transition(phase, *self.PHASE_B)
        if j is not None:
            segB, mB = self._window(feat, j - self.SEG_B_PRE, j, self.SEG_B_LEN)
        else:
            segB = np.zeros((self.SEG_B_LEN, feat.shape[1]), dtype=np.float32)
            mB = np.zeros(self.SEG_B_LEN, dtype=np.float32)

        # 拼接 -> 归一化
        x = np.concatenate([segA, segB], axis=0)       # (2000, C)
        mask = np.concatenate([mA, mB], axis=0)         # (2000,)
        x = self._instance_norm(x, mask)

        return (torch.from_numpy(x),
                torch.tensor([label], dtype=torch.long),
                torch.from_numpy(mask))

    def __len__(self):
        return len(self.samples)


class QARCompactForecastDataset(Dataset):
    """
    Forecasting adapter for QAR compact caches.

    The classification pipeline stores each flight/window as one compact tensor:
        x:    (N, T, C)
        mask: (N, T)

    For long-term forecasting we keep the split at the flight/window level instead
    of concatenating all flights into one artificial time series.  Each selected
    flight/window is then expanded into one or more forecasting sub-windows.
    When ``tsfile_conversion_meta.json`` contains anchor definitions, the default
    ``segment`` mode slides inside each anchor segment and avoids crossing the
    artificial concatenation boundaries between flight-condition snippets.
    ``seq_y`` follows the Time-Series-Library convention: it contains
    ``label_len`` known decoder points plus ``pred_len`` future points.
    """

    SPLIT_RATIOS = (0.7, 0.1, 0.2)

    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='qar_compact_shiftN80.npz',
                 target='OT', scale=False, timeenc=1, freq='h', seasonal_patterns=None):
        if size is None:
            self.seq_len = 60
            self.label_len = 20
            self.pred_len = 20
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.args = args
        self.root_path = root_path
        self.flag = str(flag).lower()
        self.features = features
        self.target = target
        self.scale = False
        self.timeenc = timeenc
        self.freq = freq

        if self.flag not in ('train', 'val', 'test'):
            raise ValueError('QARCompactForecastDataset flag must be train/val/test, got {}'.format(flag))

        cache_path = os.path.join(root_path, data_path)
        if not os.path.isfile(cache_path):
            cache_path = os.path.join(root_path, 'qar_compact_shiftN80.npz')
        if not os.path.isfile(cache_path):
            raise FileNotFoundError('QAR compact forecast cache not found under {}'.format(root_path))

        cache = np.load(cache_path, allow_pickle=False)
        x = np.asarray(cache['x'], dtype=np.float32)
        if x.ndim != 3:
            raise ValueError('Expected compact x with shape (N,T,C), got {}'.format(x.shape))
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        labels = cache['labels'].astype(np.int64, copy=False) if 'labels' in cache.files else np.zeros(x.shape[0], dtype=np.int64)
        if 'mask' in cache.files:
            mask = np.asarray(cache['mask'], dtype=np.float32)
        else:
            mask = np.ones(x.shape[:2], dtype=np.float32)

        if 'feature_cols' in cache.files:
            self.feature_cols = cache['feature_cols'].astype(str).tolist()
        else:
            self.feature_cols = ['var_{}'.format(i) for i in range(x.shape[2])]

        required_len = self.seq_len + self.pred_len
        self.forecast_stride = max(1, int(getattr(args, 'forecast_stride', required_len)))
        self.forecast_window_mode = str(getattr(args, 'forecast_window_mode', 'segment')).lower()
        if self.forecast_window_mode not in ('first', 'full', 'segment'):
            raise ValueError('Unsupported QAR forecast window mode: {}'.format(self.forecast_window_mode))

        if x.shape[1] < required_len:
            raise ValueError(
                'Forecast window too short: T={}, need seq_len + pred_len = {}'.format(
                    x.shape[1], required_len))

        self._all_x = x
        self._all_labels = labels
        self._all_mask = mask
        self.cache_path = cache_path
        self.max_seq_len = int(x.shape[1])
        self.feature_df = pd.DataFrame(columns=self.feature_cols)
        self.class_names = sorted([str(v) for v in np.unique(labels).tolist()], key=lambda z: int(z) if str(z).isdigit() else str(z))

        selected = []
        for label in np.unique(labels):
            idx = np.flatnonzero(labels == label)
            n = len(idx)
            train_end = int(n * self.SPLIT_RATIOS[0])
            val_end = int(n * (self.SPLIT_RATIOS[0] + self.SPLIT_RATIOS[1]))
            if self.flag == 'train':
                selected.extend(idx[:train_end].tolist())
            elif self.flag == 'val':
                selected.extend(idx[train_end:val_end].tolist())
            else:
                selected.extend(idx[val_end:].tolist())

        self.source_samples = np.asarray(selected, dtype=np.int64)
        self.segments = self._resolve_forecast_segments(root_path, self.max_seq_len, required_len)
        self.windows = self._build_forecast_windows(self.source_samples, mask, required_len)
        if len(self.windows) == 0:
            raise ValueError(
                'No valid QAR forecast samples for flag={} in {}; check mask/window length.'.format(
                    self.flag, cache_path))

        # Four zero-valued temporal covariates match the default timeF+h embedding.
        # Relative ordering is still available through the model's positional embedding.
        self._stamp = np.zeros((self.max_seq_len, 4), dtype=np.float32)

        zero_count = 0
        inspect_count = min(len(self.windows), 1000)
        for cache_idx, start in self.windows[:inspect_count]:
            window = x[int(cache_idx), int(start):int(start) + required_len]
            if np.abs(window).sum() == 0:
                zero_count += 1
        zero_rate = float(zero_count / inspect_count) if inspect_count else 0.0
        print('{} forecast windows: {} from {} compact samples '
              '(mode={}, stride={}, segments={}, cache={}, x_shape={}, inspected_zero_window_rate={:.6f})'.format(
                  self.flag, len(self.windows), len(self.source_samples), self.forecast_window_mode,
                  self.forecast_stride, len(self.segments), cache_path, tuple(x.shape), zero_rate))

    def _resolve_forecast_segments(self, root_path, total_len, required_len):
        if self.forecast_window_mode == 'first':
            return [(0, min(total_len, required_len))]
        if self.forecast_window_mode == 'full':
            return [(0, total_len)]

        meta_path = os.path.join(root_path, 'tsfile_conversion_meta.json')
        if not os.path.isfile(meta_path):
            return [(0, total_len)]

        try:
            with open(meta_path, 'r', encoding='utf-8') as handle:
                meta = json.load(handle)
        except Exception as exc:
            print('Warning: failed to read {}; falling back to full-window forecast segments: {}'.format(
                meta_path, exc))
            return [(0, total_len)]

        anchors = meta.get('anchors') or []
        segments = []
        cursor = 0
        for anchor in anchors:
            try:
                length = int(anchor.get('pre', 0)) + int(anchor.get('post', 0))
            except Exception:
                length = 0
            if length <= 0:
                continue
            start = cursor
            end = min(cursor + length, total_len)
            if end - start >= required_len:
                segments.append((start, end))
            cursor += length

        if not segments:
            return [(0, total_len)]
        if cursor != total_len:
            print('Warning: anchor segment length {} != compact length {}; using bounded anchor segments only.'.format(
                cursor, total_len))
        return segments

    def _segment_starts(self, start, end, required_len):
        last_start = end - required_len
        if last_start < start:
            return []
        starts = list(range(start, last_start + 1, self.forecast_stride))
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def _build_forecast_windows(self, source_samples, mask, required_len):
        windows = []
        for cache_idx in source_samples:
            row_mask = mask[int(cache_idx)]
            for segment_start, segment_end in self.segments:
                for start in self._segment_starts(segment_start, segment_end, required_len):
                    end = start + required_len
                    if row_mask[start:end].sum() >= required_len:
                        windows.append((int(cache_idx), int(start)))
        return np.asarray(windows, dtype=np.int64)

    def __getitem__(self, index):
        cache_idx = int(self.windows[index, 0])
        s_begin = int(self.windows[index, 1])
        seq = self._all_x[cache_idx]

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = seq[s_begin:s_end]
        seq_y = seq[r_begin:r_end]
        seq_x_mark = self._stamp[s_begin:s_end]
        seq_y_mark = self._stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.windows)

    def inverse_transform(self, data):
        return data
