import argparse
import torch
from exp.pretrain import Exp_Train, Exp_Pretrain_TimeSiam, Exp_Pretrain_PatchTST
import random
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

parser = argparse.ArgumentParser(description='TimeSiam')

parser.add_argument('--seed', type=int, default=2023, help='seed')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='in_domain', help='task name, options:[in_domain, cross_domain]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='PatchTST', help='model name, options: [PatchTST, iTransformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./datasets', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./outputs/checkpoints/', help='location of model checkpoints')
parser.add_argument('--pretrain_checkpoints', type=str, default='./outputs/pretrain_checkpoints/', help='location of model pre-training checkpoints')
parser.add_argument('--load_checkpoints', type=str, default=None, help='location of model checkpoints')
parser.add_argument('--select_channels', type=float, default=1, help='select the rate of channels to train')
parser.add_argument('--percent', type=int, default=100)

# forecasting task
parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.40, help='prior anomaly ratio (%)')

# model define
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=3, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')

# optimization
parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

# Pre-train
parser.add_argument('--lm', type=int, default=3, help='average masking length')
parser.add_argument('--positive_nums', type=int, default=3, help='masking nums')
parser.add_argument('--rbtp', type=int, default=1, help='0: rebuild the embedding of oral data; 1: rebuild oral data')
parser.add_argument('--temperature', type=float, default=0.02, help='temperature')
parser.add_argument('--masked_rule', type=str, default='channel_continuous', help='binomial, channel_binomial, continuous, channel_continuous, mask_last, mask_patch')
parser.add_argument('--window_gap', type=str, default=None, help='0 [0.5*seq, 1*seq], 1 [1*seq, 2*seq], 2 [2*seq, 4*seq], 3 [4*seq, 8*seq], default')
parser.add_argument('--finetune_ratio', type=float, default=None, help='finetune_ratio')

# PatchTST
parser.add_argument('--patch_len', type=int, default=12, help='path length')
parser.add_argument('--stride', type=int, default=12, help='stride')

# Siamese
parser.add_argument('--sampling_range', type=int, default=None, help='Siamese subseries sampling range: [0, sampling_range*seq_len]')
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
parser.add_argument('--lineage_tokens', type=int, default=2, help='the type numbers of lineage embeddings')
parser.add_argument('--tokens_using', type=str, default='single', help='single, flatten along time dimension')
parser.add_argument('--representation_using', type=str, default='avg', help='concat, avg')
parser.add_argument('--current_token', action='store_true', default=False, help='whether use current token')

# visual
parser.add_argument('--visual', type=int, default=0)
parser.add_argument('--visual_path', type=str, default='./outputs/showcases')


args = parser.parse_args()
print(torch.cuda.is_available())
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


if args.task_name == 'timesiam':
    Exp = Exp_Pretrain_TimeSiam
elif args.task_name == 'patchtst':
    Exp = Exp_Pretrain_PatchTST
else:
    Exp = Exp_Train

print(Exp)

if args.is_training == 0:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_df{}_nh{}_el{}_dl{}_fc{}_dp{}_hdp{}_ep{}_bs{}_lr{}_lm{}_pn{}_mr{}_tp{}'.format(
            args.task_name,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.d_ff,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.factor,
            args.dropout,
            args.head_dropout,
            args.train_epochs,
            args.batch_size,
            args.learning_rate,
            args.lm,
            args.positive_nums,
            args.mask_rate,
            args.temperature
        )

        exp = Exp(args)  # set experiments
        print('>>>>>>>start pre_training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.pretrain()

elif args.is_training == 1:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_df{}_nh{}_el{}_dl{}_fc{}_dp{}_hdp{}_ep{}_bs{}_lr{}_ru{}'.format(
            args.task_name,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.d_ff,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.factor,
            args.dropout,
            args.head_dropout,
            args.train_epochs,
            args.batch_size,
            args.learning_rate,
            args.representation_using
        )

        exp = Exp(args)  # set experiments
        exp.fine_tune(setting)
else:
    ii = 0
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
