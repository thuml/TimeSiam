import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
from utils.tools import lineage_search
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.utils import load_data
import warnings
from sklearn.preprocessing import StandardScaler
import random
from statsmodels.tsa.stattools import adfuller
import math
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100, seasonal_patterns=None, window_gap=None, sampling_range=None, lineage_tokens=None, neighours=False):
        # size [seq_len, label_len, pred_len]
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
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.window_gap = window_gap

        self.sampling_range = sampling_range
        self.lineage_tokens = lineage_tokens

        self.__read_data__()

        self.neighours = neighours
        self.time_series = self.data_x
        self.window_size = 5
        self.epsilon = 3
        self.delta = 5 * self.window_size * self.epsilon
        self.mc_sample_size = 20
        self.T = self.seq_len

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

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
        self.data_stamp = data_stamp

        print('data size:', self.data_x.shape)

    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len

        if self.sampling_range == 0:
            r_begin = s_begin
            r_end = s_end
            segment = 0
        elif self.sampling_range and self.sampling_range != 0:

            r_limit = s_begin + self.sampling_range*self.seq_len
            if r_limit > len(self.data_x) - self.seq_len:
                r_limit = len(self.data_x) - self.seq_len

            r_begin = random.randint(s_begin, r_limit)
            r_end = r_begin + self.seq_len
            segment = lineage_search(r_limit-s_begin, self.lineage_tokens, r_begin-s_begin)

        else:
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.neighours:
            self.time_series = seq_x
            ind = index % len(self.time_series)
            t = np.random.randint(2 * self.window_size, self.T - 2 * self.window_size)
            x_t = self.time_series[t - self.window_size // 2:t + self.window_size // 2]

            self.time_series = np.transpose(self.time_series)
            X_close = self._find_neighours(self.time_series, t)
            X_distant = self._find_non_neighours(self.time_series, t)

            return x_t, X_close, X_distant

        if self.sampling_range is not None:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, segment
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def _find_neighours(self, x, t):
        T = self.time_series.shape[-2]

        ## Random from a Gaussian
        t_p = [int(t+np.random.randn()*self.epsilon*self.window_size) for _ in range(self.mc_sample_size)]
        t_p = [max(self.window_size//2+1,min(t_pp,T-self.window_size//2)) for t_pp in t_p]

        x_p = torch.stack(
            [torch.from_numpy(x[:, t_ind - self.window_size // 2:t_ind + self.window_size // 2]) for t_ind in t_p])
        return x_p

    def _find_non_neighours(self, x, t):
        T = self.time_series.shape[-2]
        if t>T/2:
            t_n = np.random.randint(self.window_size//2, max((t - self.delta + 1), self.window_size//2+1), self.mc_sample_size)
        else:
            t_n = np.random.randint(min((t + self.delta), (T - self.window_size-1)), (T - self.window_size//2), self.mc_sample_size)

        x_n = torch.stack([torch.from_numpy(x[:, t_ind-self.window_size//2:t_ind+self.window_size//2]) for t_ind in t_n])

        if len(x_n)==0:
            rand_t = np.random.randint(0,self.window_size//5)
            if t > T / 2:
                x_n = x[:,rand_t:rand_t+self.window_size].unsqueeze(0)
            else:
                x_n = x[:, T - rand_t - self.window_size:T - rand_t].unsqueeze(0)
        return x_n


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', percent=100, seasonal_patterns=None, window_gap=None, sampling_range=None, lineage_tokens=None, neighours=False):
        # size [seq_len, label_len, pred_len]
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
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.window_gap = window_gap
        self.sampling_range = sampling_range
        self.lineage_tokens = lineage_tokens
        self.__read_data__()

        self.neighours = neighours
        self.time_series = self.data_x
        self.window_size = 5
        self.epsilon = 3
        self.delta = 5 * self.window_size * self.epsilon
        self.mc_sample_size = 20
        self.T = self.seq_len

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

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
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if self.sampling_range == 0:
            r_begin = s_begin
            r_end = s_end
            segment = 0
        elif self.sampling_range and self.sampling_range != 0:

            r_limit = s_begin + self.sampling_range*self.seq_len
            if r_limit > len(self.data_x) - self.seq_len:
                r_limit = len(self.data_x) - self.seq_len

            r_begin = random.randint(s_begin, r_limit)
            r_end = r_begin + self.seq_len
            segment = lineage_search(r_limit-s_begin, self.lineage_tokens, r_begin-s_begin)

        else:
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.neighours:
            self.time_series = seq_x
            ind = index % len(self.time_series)
            t = np.random.randint(2 * self.window_size, self.T - 2 * self.window_size)
            x_t = self.time_series[t - self.window_size // 2:t + self.window_size // 2]

            self.time_series = np.transpose(self.time_series)
            X_close = self._find_neighours(self.time_series, t)
            X_distant = self._find_non_neighours(self.time_series, t)

            return x_t, X_close, X_distant

        if self.sampling_range is not None:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, segment
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def _find_neighours(self, x, t):
        T = self.time_series.shape[-2]

        ## Random from a Gaussian
        t_p = [int(t+np.random.randn()*self.epsilon*self.window_size) for _ in range(self.mc_sample_size)]
        t_p = [max(self.window_size//2+1,min(t_pp,T-self.window_size//2)) for t_pp in t_p]

        x_p = torch.stack(
            [torch.from_numpy(x[:, t_ind - self.window_size // 2:t_ind + self.window_size // 2]) for t_ind in t_p])
        return x_p

    def _find_non_neighours(self, x, t):
        T = self.time_series.shape[-2]
        if t>T/2:
            t_n = np.random.randint(self.window_size//2, max((t - self.delta + 1), self.window_size//2+1), self.mc_sample_size)
        else:
            t_n = np.random.randint(min((t + self.delta), (T - self.window_size-1)), (T - self.window_size//2), self.mc_sample_size)

        x_n = torch.stack([torch.from_numpy(x[:, t_ind-self.window_size//2:t_ind+self.window_size//2]) for t_ind in t_n])

        if len(x_n)==0:
            rand_t = np.random.randint(0,self.window_size//5)
            if t > T / 2:
                x_n = x[:,rand_t:rand_t+self.window_size].unsqueeze(0)
            else:
                x_n = x[:, T - rand_t - self.window_size:T - rand_t].unsqueeze(0)
        return x_n


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100, seasonal_patterns=None, window_gap=None, sampling_range=None, lineage_tokens=None, neighours=False):
        # size [seq_len, label_len, pred_len]
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
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.window_gap = window_gap
        self.sampling_range = sampling_range
        self.lineage_tokens = lineage_tokens
        self.__read_data__()

        self.neighours = neighours
        self.time_series = self.data_x
        self.window_size = 5
        self.epsilon = 3
        self.delta = 5 * self.window_size * self.epsilon
        self.mc_sample_size = 20
        self.T = self.seq_len

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

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

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

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
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if self.sampling_range == 0:
            r_begin = s_begin
            r_end = s_end
            segment = 0
        elif self.sampling_range and self.sampling_range > 0:

            r_limit = s_begin + self.sampling_range*self.seq_len
            if r_limit > len(self.data_x) - self.seq_len:
                r_limit = len(self.data_x) - self.seq_len

            r_begin = random.randint(s_begin, r_limit)
            r_end = r_begin + self.seq_len
            segment = lineage_search(r_limit-s_begin, self.lineage_tokens, r_begin-s_begin)

        else:
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.neighours:
            self.time_series = seq_x
            ind = index % len(self.time_series)
            t = np.random.randint(2 * self.window_size, self.T - 2 * self.window_size)
            x_t = self.time_series[t - self.window_size // 2:t + self.window_size // 2]

            self.time_series = np.transpose(self.time_series)
            X_close = self._find_neighours(self.time_series, t)
            X_distant = self._find_non_neighours(self.time_series, t)

            return x_t, X_close, X_distant

        if self.sampling_range is not None:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, segment
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def _find_neighours(self, x, t):
        T = self.time_series.shape[-2]

        ## Random from a Gaussian
        t_p = [int(t+np.random.randn()*self.epsilon*self.window_size) for _ in range(self.mc_sample_size)]
        t_p = [max(self.window_size//2+1,min(t_pp,T-self.window_size//2)) for t_pp in t_p]

        x_p = torch.stack(
            [torch.from_numpy(x[:, t_ind - self.window_size // 2:t_ind + self.window_size // 2]) for t_ind in t_p])
        return x_p

    def _find_non_neighours(self, x, t):
        T = self.time_series.shape[-2]
        if t>T/2:
            t_n = np.random.randint(self.window_size//2, max((t - self.delta + 1), self.window_size//2+1), self.mc_sample_size)
        else:
            t_n = np.random.randint(min((t + self.delta), (T - self.window_size-1)), (T - self.window_size//2), self.mc_sample_size)

        x_n = torch.stack([torch.from_numpy(x[:, t_ind-self.window_size//2:t_ind+self.window_size//2]) for t_ind in t_n])

        if len(x_n)==0:
            rand_t = np.random.randint(0,self.window_size//5)
            if t > T / 2:
                x_n = x[:,rand_t:rand_t+self.window_size].unsqueeze(0)
            else:
                x_n = x[:, T - rand_t - self.window_size:T - rand_t].unsqueeze(0)
        return x_n