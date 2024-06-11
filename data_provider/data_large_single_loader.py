import os
import random

import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import torch.distributed as dist
from utils.tools import lineage_search
warnings.filterwarnings('ignore')


class DatasetLargeSingle(Dataset):
    def __init__(self, root_path,
                 flag='train',
                 seq_len=96,
                 label_len=48,
                 pred_len=96,
                 features='M',
                 normalize=True,
                 subset=1.0,
                 timeenc=1,
                 freq='h',
                 target='OT',
                 batch_size=32,
                 use_multi_gpu=False,
                 split=0.9,
                 sampling_range=None,
                 lineage_tokens=None
                 ):
        # info
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len

        # init
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}

        self.split = split

        self.set_type = type_map[flag]
        self.use_multi_gpu = use_multi_gpu

        self.features = features
        self.target = target
        self.scale = normalize
        self.timeenc = timeenc
        self.freq = freq
        self.batch_size = batch_size

        self.dataset_list = []
        self.dataset_stamp_list = []
        self.dataset_len_list = []

        self.root_path = root_path
        self.sampling_range = sampling_range
        self.lineage_tokens = lineage_tokens
        self.__confirm_data__()

    def __confirm_data__(self):

        max_cols = 0
        for root, dirs, files in os.walk(self.root_path):
            for f in files:
                if f.endswith('.csv') or f.endswith('.txt') or f.endswith('.npz'):

                    if 'hhh' in f:
                        continue
                    dataset_path = os.path.join(root, f)

                    data, data_stamp = self.__read_data__(dataset_path)
                    max_cols = max(max_cols, data.shape[1])
                    if self.use_multi_gpu:
                        if data.shape[0] < (self.seq_len + self.pred_len + self.batch_size) * dist.get_world_size():
                            continue
                    else:
                        if data.shape[0] < self.seq_len + self.pred_len + self.batch_size:
                            continue
                    self.dataset_list.append(data)
                    self.dataset_stamp_list.append(data_stamp)
                    # dataset_len_list[i] is the sum of the length of all the datasets before dataset_list[i]
                    dataset_len = len(data) - self.seq_len - self.pred_len + 1
                    self.dataset_len_list.append(dataset_len if len(self.dataset_len_list) == 0 else self.dataset_len_list[-1] + dataset_len)
                    print(self.flag, "dataset name: ", dataset_path, " data shape: ", data.shape,
                          " data_stamp shape: ", data_stamp.shape, " all_len: ", self.dataset_len_list[-1])

        print("max_cols: ", max_cols)

    def __read_data__(self, data_path):
        self.scaler = StandardScaler()
        if data_path.endswith('.csv'):
            df_raw = pd.read_csv(data_path)
        elif data_path.endswith('.txt'): # For SolarEnergy

            df_raw = []
            with open(data_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)
        elif data_path.endswith('.npz'):  # For PEMS
            data = np.load(data_path, allow_pickle=True)
            data = data['data']
            if data.ndim > 2:
                data = data[:, :, 0]
            df_raw = pd.DataFrame(data)
        df_raw = df_raw.fillna(method='ffill', limit=len(df_raw)).fillna(method='bfill', limit=len(df_raw))

        if isinstance(df_raw[df_raw.columns[0]][0], str): # is datetime
            data_stamp = time_features(pd.to_datetime(pd.to_datetime(df_raw[df_raw.columns[0]]).values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            data = df_raw[df_raw.columns[1:]].values
        else:
            data_stamp = np.zeros((len(df_raw), 4))
            data = df_raw.values

        num_train = int(len(data) * self.split)
        border1s = [0, num_train - self.seq_len]
        border2s = [num_train, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)
        else:
            data = data

        return data[border1:border2], data_stamp[border1:border2]

    def __getitem__(self, index):

        dataset_index = 0
        while index >= self.dataset_len_list[dataset_index]:
            dataset_index += 1

        index = index - self.dataset_len_list[dataset_index - 1] if dataset_index > 0 else index

        n_timepoint = len(self.dataset_list[dataset_index]) - self.seq_len - self.pred_len + 1

        c_begin = index // n_timepoint  # select variable
        s_begin = index % n_timepoint  # select start timestamp
        s_end = s_begin + self.seq_len


        self.data_x = self.dataset_list[dataset_index]
        self.data_y = self.dataset_list[dataset_index]
        self.data_stamp = self.dataset_stamp_list[dataset_index]
        self.data_stamp = self.dataset_stamp_list[dataset_index]


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
            r_begin = s_end
            r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end, c_begin:c_begin + 1]
        seq_y = self.data_y[r_begin:r_end, c_begin:c_begin + 1]
        assert seq_x.shape[0] == self.seq_len
        assert seq_y.shape[0] == self.pred_len
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.sampling_range is not None:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, segment
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        print(len(self.dataset_len_list))
        return self.dataset_len_list[-1]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class LargeSampler(Sampler):
    def __init__(self, data_source: DatasetLargeSingle, shuffle=False, batch_size=32, subset_ratio=1.0):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle
        self.dataset_len_list = data_source.dataset_len_list
        self.num_samples = len(self.data_source)
        self.batch_size = batch_size
        self.subset_ratio = subset_ratio
    def __iter__(self):
        final_list = []

        total_len = self.dataset_len_list[-1]
        if self.shuffle:
            final_list = torch.randperm(total_len)[:total_len // self.batch_size * self.batch_size].reshape(-1, self.batch_size).tolist()
        else:
            final_list = torch.arange(total_len)[:total_len // self.batch_size * self.batch_size].reshape(-1, self.batch_size).tolist()

        # 对final_list进行shuffle
        print("Batch Count: ", len(final_list), "Batch Size: ", self.batch_size)
        if self.shuffle:
            random.shuffle(final_list)
            print("shuffle done")
        final_list = final_list[:int(len(final_list) * self.subset_ratio)]
        print("Sampled Batch Count: ", len(final_list), "Batch Size: ", self.batch_size)
        yield from final_list

    def __len__(self):
        return self.num_samples


class LargeDistributedSampler(DistributedSampler):
    def __init__(self, data_source: DatasetLargeSingle, shuffle=False, batch_size=32, subset_ratio=1.0):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle
        self.dataset_len_list = data_source.dataset_len_list
        self.num_samples = len(self.data_source)
        self.batch_size = batch_size
        self.subset_ratio = subset_ratio
    def __iter__(self):
        final_list = []

        total_len = self.dataset_len_list[-1]
        if self.shuffle:
            final_list = torch.randperm(total_len)[:total_len // self.batch_size * self.batch_size].reshape(-1, self.batch_size).tolist()
        else:
            final_list = torch.arange(total_len)[:total_len // self.batch_size * self.batch_size].reshape(-1, self.batch_size).tolist()

        # 对final_list进行shuffle
        print("Total Batch Count: ", len(final_list), "Batch Size: ", self.batch_size)

        rank_num_samples = math.ceil(len(final_list) / self.num_replicas)
        total_size = rank_num_samples * self.num_replicas
        final_list = final_list[:total_size][self.rank:total_size:self.num_replicas][:rank_num_samples - 1]
        print("Single Batch Count: ", len(final_list), "Batch Size: ", self.batch_size)
        if self.shuffle:
            random.shuffle(final_list)
            print("shuffle done")
        final_list = final_list[:int(len(final_list) * self.subset_ratio)]
        print("Sampled Single Batch Count: ", len(final_list), "Batch Size: ", self.batch_size)
        yield from final_list

    def __len__(self):
        return self.num_samples



class DataloaderLargeSingle(DataLoader):
    def __init__(self, root_path,
                 flag='train',
                 seq_len=96,
                 label_len=48,
                 pred_len=96,
                 features='M',
                 normalize=True,
                 subset=1.0,
                 timeenc=1,
                 freq='h',
                 target='OT',
                 batch_size=32,
                 shuffle=True,
                 use_multi_gpu=False,
                 subset_ratio=1.0,
                 sampling_range=None,
                 lineage_tokens=None
                 ):
        dataset = DatasetLargeSingle(root_path,
                                    flag=flag,
                                    seq_len=seq_len,
                                    label_len=label_len,
                                    pred_len=pred_len,
                                    features=features,
                                    normalize=normalize,
                                    subset=subset,
                                    timeenc=timeenc,
                                    freq=freq,
                                    target=target,
                                    batch_size=batch_size,
                                    use_multi_gpu=use_multi_gpu,
                                    sampling_range=sampling_range,
                                    lineage_tokens=lineage_tokens
                                     )


        if use_multi_gpu:
            sampler = LargeDistributedSampler(dataset, shuffle=shuffle, batch_size=batch_size, subset_ratio=subset_ratio)
            super(DataloaderLargeSingle, self).__init__(dataset=dataset, batch_sampler=sampler)

        else:
            sampler = LargeSampler(dataset, shuffle=shuffle, batch_size=batch_size, subset_ratio=subset_ratio)
            super(DataloaderLargeSingle, self).__init__(dataset=dataset, batch_sampler=sampler)
