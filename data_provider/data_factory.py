from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from data_provider.data_large_loader import DataloaderLarge
from data_provider.data_large_single_loader import DataloaderLargeSingle

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Traffic': Dataset_Custom,
    'Exchange': Dataset_Custom,
    'Weather': Dataset_Custom,
    'ECL': Dataset_Custom,
    'TSLD-0.5G': [DataloaderLarge, DataloaderLargeSingle],
    'TSLD-1G': [DataloaderLarge, DataloaderLargeSingle]
}


def data_provider(args, flag):

    Data = data_dict[args.data]
    root_path = args.root_path
    percent = args.percent

    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=root_path,
            flag=flag,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False

        if 'TSLD' in args.data:

            if 'PatchTST' in args.model:
                Data = Data[1]
            elif 'iTransformer' in args.model:
                Data = Data[0]

            data_loader = Data(
                root_path=root_path,
                flag=flag,
                seq_len=args.seq_len,
                label_len=args.label_len,
                pred_len=args.pred_len,
                features=args.features,
                timeenc=timeenc,
                freq=freq,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                sampling_range=args.sampling_range,
                lineage_tokens=args.lineage_tokens,
            )

            return data_loader.dataset, data_loader
        else:

            data_set = Data(
                root_path=root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                percent=percent,
                seasonal_patterns=args.seasonal_patterns,
                sampling_range=args.sampling_range,
                lineage_tokens=args.lineage_tokens,
                neighours=False,
            )

            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)

            return data_set, data_loader