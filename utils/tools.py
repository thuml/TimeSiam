import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
import math
plt.switch_backend('agg')

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 1 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def compare_tensors(tensor1, tensor2):
    """
    Compares two PyTorch tensors element-wise and returns a tensor with 1 where tensor1 is greater than or
    equal to tensor2, and 0 where tensor1 is less than tensor2.

    Args:
        tensor1 (torch.Tensor): A PyTorch tensor.
        tensor2 (torch.Tensor): A PyTorch tensor with the same shape as tensor1.

    Returns:
        A PyTorch tensor with the same shape as tensor1, containing 1s where tensor1 is greater than or equal to
        tensor2, and 0s where tensor1 is less than tensor2.
    """
    # Use PyTorch's element-wise comparison function to create a tensor of 1s and 0s
    comparison = torch.ge(tensor1, tensor2)

    # Convert the boolean tensor to a tensor of 1s and 0s
    result = comparison.int()

    return result.type_as(torch.LongTensor())

def show_pretrain_case(past_window_true, future_window_true, future_window_mask, future_prediction, segment):

    segment_window = torch.zeros(past_window_true.shape[0]).repeat(segment)

    # 将张量转换为NumPy数组
    past_window_true = past_window_true.numpy()
    future_window_true = future_window_true.numpy()
    future_window_mask = future_window_mask.numpy()
    segment_window = segment_window.numpy()

    # 创建一个指定大小的图像，并将图像分为两个子图
    fig, axs = plt.subplots(1, 1, figsize=(20, 10))

    # 绘制第一个子图（past_window_true）
    past_and_future_window = np.concatenate([past_window_true,  segment_window, future_window_true], axis=0)
    past_and_future_window_mask = np.concatenate([past_window_true,  segment_window, future_window_mask], axis=0)
    past_and_future_prediction = np.concatenate([past_window_true,  segment_window, future_prediction], axis=0)

    x = list(range(past_and_future_window.shape[0]))

    for i in range(len(past_and_future_window) - 1):
        if i < len(past_window_true):
            axs.plot([x[i], x[i + 1]], [past_and_future_window[i], past_and_future_window[i + 1]], '-', color='black', label='ture past window')
        elif i >= len(past_window_true) and i < (len(past_window_true) + len(segment_window)):
            continue
        else:
            if past_and_future_window_mask[i] == 0:
                axs.plot([x[i], x[i + 1]], [past_and_future_window[i], past_and_future_window[i + 1]], ':', color='grey', alpha=0.5, label='masked future window')
            else:
                axs.plot([x[i], x[i + 1]], [past_and_future_window[i], past_and_future_window[i + 1]], '-', color='blue', label='unmasked future window')

            axs.plot([x[i], x[i + 1]], [past_and_future_prediction[i], past_and_future_prediction[i + 1]], '-', color='orange', label='predition future window')

    axs.set_title('Past vs Future')
    axs.set_xlabel('X - time points')
    axs.set_ylabel('Y - time values')

    return fig

def show_pretrain_case(past, current, masked_current, predict_current, segment, mask, noise=None, mask_rate=None, t=None):

    # 将张量转换为NumPy数组
    past_window_true = past[:, -1].numpy() # [seq_len]
    future_window_true = current[:, -1].numpy() # [seq_len]
    future_window_mask = masked_current[:, -1].numpy() # [seq_len]
    future_prediction = predict_current[:, -1]
    mask = mask[:, -1] # [seq_len]

    fig, axs = plt.subplots(1, 3, figsize=(24, 6))
    x = list(range(past_window_true.shape[0]))

    axs[0].plot(x, past_window_true, color='black', label=f'Past window ({segment.item()}-th segment)')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Values')
    axs[0].set_title('Past window')
    axs[0].legend()

    if noise is not None:
        noise = noise[:, -1].numpy()
        axs[1].plot(x, noise, ':', color='grey', label='Gaussian noise', alpha=0.5)

    axs[1].plot(x, future_prediction, '-', color='orange', label='Forecasting current window')

    masked_idx = np.where(mask)[0]

    for i in range(len(x) - 1):
        if i not in masked_idx:
            axs[1].plot([x[i], x[i + 1]], [future_window_true[i], future_window_true[i + 1]], '-', color='blue', label='Current window')
        else:
            # 在 masked_idx 中的索引绘制实心灰色圆点
            axs[1].scatter(x[i], future_window_mask[i], facecolors='none', edgecolors='red', linewidths=2, s=30, label='{:.2f}% masked Gaussian noise, S/R ({:.2f}/{:.2f})'.format(mask_rate, t[0], t[1]) if i == masked_idx[0] else '')
            #axs[1].plot([x[i], x[i + 1]], [future_window_mask[i], future_window_mask[i + 1]], '-', color='grey', label='Masked timestamps')

            # if i != 0:
            #     axs[1].plot([x[i-1], x[i]], [future_window_mask[i-1], future_window_mask[i]], '-', color='red', label='{:.2f}% masked Gaussian noise, S/R ({:.2f}/{:.2f})'.format(mask_rate, t, 1-t), linewidth=2)
            # axs[1].plot([x[i], x[i + 1]], [future_window_mask[i], future_window_mask[i + 1]], '-', color='red', label='{:.2f}% masked Gaussian noise, S/R ({:.2f}/{:.2f})'.format(mask_rate, t, 1-t), linewidth=2)

            axs[1].scatter(x[i], future_window_true[i], facecolors='none', edgecolors='blue', linewidths=1, s=30, label='Original data')
            axs[1].scatter(x[i], noise[i], facecolors='none', edgecolors='grey', linewidths=1, s=30, label='Gaussian noise')

            #axs[1].plot([x[i], x[i + 1]], [future_window_true[i], future_window_true[i + 1]], ':', color='blue', label='Replaced data', alpha=0.5)

    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Values')
    axs[1].set_title('Current window')


    masked_future_prediction = future_prediction[masked_idx]
    masked_future_window_true = future_window_true[masked_idx]

    x = list(range(masked_future_prediction.shape[0]))

    axs[2].plot(x, masked_future_prediction, '-', color='orange', label='Masked Forecasting')
    axs[2].plot(x, masked_future_window_true, ':', color='grey', label='Ground truth', linewidth=2, alpha=0.7)
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Values')
    axs[2].set_title('Masked current window')
    axs[2].legend()


    handles, labels = axs[1].get_legend_handles_labels()
    unique_labels = list(set(labels))  # Get unique labels
    unique_handles = [handles[labels.index(label)] for label in unique_labels]  # Get handles for unique labels
    axs[1].legend(unique_handles, unique_labels)

    return fig


def show_pretrain_case(past, current, masked_current, predict_current, segment, mask):

    # 将张量转换为NumPy数组
    past_window_true = past[:, 0].numpy() # [seq_len]
    future_window_true = current[:, 0].numpy() # [seq_len]
    future_window_mask = masked_current[:, 0].numpy() # [seq_len]
    future_prediction = predict_current[:, 0]
    mask = mask[:, -1] # [seq_len]

    fig, axs = plt.subplots(1, 3, figsize=(24, 6))
    x = list(range(past_window_true.shape[0]))

    axs[0].plot(x, past_window_true, color='black', label=f'Past window ({segment.item()}-th segment)')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Values')
    axs[0].set_title('Past window')
    axs[0].legend()

    axs[1].plot(x, future_prediction, '-', color='orange', label='Forecasting current window')

    masked_idx = np.where(mask)[0]

    for i in range(len(x) - 1):
        if i not in masked_idx:
            axs[1].plot([x[i], x[i + 1]], [future_window_true[i], future_window_true[i + 1]], '-', color='blue', label='Current window')
        else:
            # 在 masked_idx 中的索引绘制实心灰色圆点
            axs[1].scatter(x[i], future_window_true[i], facecolors='none', edgecolors='grey', linewidths=2, s=30, label='Masked timestamps' if i == masked_idx[0] else '')

    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Values')
    axs[1].set_title('Current window')


    masked_future_prediction = future_prediction[masked_idx]
    masked_future_window_true = future_window_true[masked_idx]

    x = list(range(masked_future_prediction.shape[0]))

    axs[2].plot(x, masked_future_prediction, '-', color='orange', label='Masked Forecasting')
    axs[2].plot(x, masked_future_window_true, ':', color='grey', label='Ground truth', linewidth=2, alpha=0.7)
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Values')
    axs[2].set_title('Masked current window')
    axs[2].legend()


    handles, labels = axs[1].get_legend_handles_labels()
    unique_labels = list(set(labels))  # Get unique labels
    unique_handles = [handles[labels.index(label)] for label in unique_labels]  # Get handles for unique labels
    axs[1].legend(unique_handles, unique_labels)

    return fig

def show_pretrain_tst(pred, target, mask, mask_rate, name):

    fig, axs = plt.subplots(1, 2, figsize=(24, 6))
    x = list(range(target.shape[0]))

    # Part 1
    #axs[0].plot(x, noise_x_enc, '-', color='black', label='Mixing data', linewidth=2)
    #axs[0].plot(x, noise, ':', color='grey', label='Gaussian noise', alpha=0.5)

    masked_idx = np.where(~mask)[0]

    for i in range(len(x) - 1):
        if i not in masked_idx:
            axs[0].plot([x[i], x[i+1]], [target[i], target[i+1]], '-', color='blue', label='Remain time points', linewidth=2)
        else:
            axs[0].scatter(x[i], target[i], facecolors='none', edgecolors='grey', linewidths=2, s=30, label='{:.2f}% time points masking'.format(mask_rate))

    axs[0].plot(x, pred, '-', color='green', label='Prediction', linewidth=2)

    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Values')
    axs[0].set_title('Time points masking')
    axs[0].legend()


    handles, labels = axs[0].get_legend_handles_labels()
    unique_labels = list(set(labels))  # Get unique labels
    unique_handles = [handles[labels.index(label)] for label in unique_labels]  # Get handles for unique labels
    axs[0].legend(unique_handles, unique_labels)

    target = target[masked_idx]
    pred = pred[masked_idx]
    x = list(range(target.shape[0]))

    # Part 2
    axs[1].plot(x, target, ':', color='grey', label='Ground Truth')
    axs[1].plot(x, pred, '-', color='green', label='Prediction', linewidth=2)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Values')
    axs[1].set_title('Masking time points reconstruction')

    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def show_pretrain_patchtst(pred, target, mask, mask_rate, name):

    fig, axs = plt.subplots(1, 2, figsize=(24, 6))
    x = list(range(target.shape[0]))

    # Part 1
    axs[0].plot(x, pred, '-', color='green', linewidth=2, label='Prediction')

    masked_idx = np.where(mask)[0]
    for i in range(len(x) - 1):
        if i in masked_idx:
            color = 'grey'
            line_sty = ':'
        else:
            color = 'blue'
            line_sty = '-'
        axs[0].plot([x[i], x[i + 1]], [target[i], target[i + 1]], line_sty, color=color, label='Ground Truth', linewidth=2)

    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Values')
    axs[0].set_title('Patch Masking ({:.2f}%)'.format(mask_rate))
    axs[0].legend()


    handles, labels = axs[0].get_legend_handles_labels()
    unique_labels = list(set(labels))  # Get unique labels
    unique_handles = [handles[labels.index(label)] for label in unique_labels]  # Get handles for unique labels
    axs[0].legend(unique_handles, unique_labels)

    target = target[masked_idx]
    pred = pred[masked_idx]
    x = list(range(target.shape[0]))

    # Part 2
    axs[1].plot(x, target, ':', color='grey', label='Ground Truth')
    axs[1].plot(x, pred, '-', color='green', label='Prediction', linewidth=2)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Values')
    axs[1].set_title('Patch Masking Reconstruction')

    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def show_mask(mask, title):

    true_count = torch.sum(mask).item()
    false_count = mask.numel() - true_count

    print('Masked: {0}\tUnmasked: {1}\tMasked ratio: {2:.2f}'.format(true_count, false_count, true_count/mask.numel()))

    mask = mask.permute(1, 0)

    if mask.shape[0] > 50:
        mask = mask[:50, :]
    if mask.shape[1] > 50:
        mask = mask[:, :50]

    fig = plt.figure(figsize=(20, 20))
    colors = [(0, 0.749, 1), (1, 1, 1)]  # 蓝色到白色
    cmap = LinearSegmentedColormap.from_list('white_to_blue', colors)

    sns.heatmap(mask, cmap=cmap, cbar=False, square=True, annot=False, xticklabels=False, yticklabels=False)

    plt.xlabel("Sequence")
    plt.ylabel("Channels")
    plt.title(title)

    # plt.savefig(f'./masking/{title}_bar.png', dpi=1000, format='png')
    plt.show()
    return fig

def transfer_weights(weights_path, model, exclude_head=True, device='cpu'):
    new_state_dict = torch.load(weights_path,  map_location=device)['model_state_dict']

    matched_layers = 0
    unmatched_layers = []
    for name, param in model.state_dict().items():
        if exclude_head and 'head' in name: continue
        if name in new_state_dict or 'module.' + name in new_state_dict:
            if 'module.' + name in new_state_dict:
                match_name = 'module.' + name
            else:
                match_name =  name

            matched_layers += 1
            input_param = new_state_dict[match_name]
            if input_param.shape == param.shape:
                param.copy_(input_param)
            else:
                unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
            pass # these are weights that weren't in the original model, such as a new head

    if matched_layers == 0:
        raise Exception("No shared weight names were found between the models")
    else:
        if len(unmatched_layers) > 0:
            print(f'check unmatched_layers: {unmatched_layers}')
        else:
            print(f"weights from {weights_path} successfully transferred!\n")
    model = model.to(device)
    return model

def show_series(batch_x, batch_x_m, pred_batch_x, idx, time_points=336):

    batch_x = batch_x.permute(0, 2, 1).reshape(batch_x.shape[0], -1)
    batch_x_m = batch_x_m.permute(0, 2, 1).reshape(batch_x_m.shape[0], -1)
    pred_batch_x = pred_batch_x.permute(0, 2, 1).reshape(batch_x_m.shape[0], -1)

    bs = batch_x.shape[0]

    if time_points is None:
        time_points = batch_x.shape[1]

    positive_numbers = batch_x_m.shape[0] // bs

    batch_x = batch_x.numpy()
    batch_x_m = batch_x_m.numpy()

    x = list(range(time_points))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b']

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    for t_i in range(time_points - 1):
        for pn in range(positive_numbers):
            s_i = pn * bs + idx
            if batch_x_m[s_i][t_i] == 0:
                axs[0].plot([x[t_i], x[t_i + 1]], [batch_x[idx][t_i], batch_x[idx][t_i + 1]], ':', color='grey', alpha=0.5, label='masked')
            else:
                axs[0].plot([x[t_i], x[t_i + 1]], [batch_x[idx][t_i], batch_x[idx][t_i + 1]], '-', color=colors[pn], label='unmasked')

        axs[1].plot([x[t_i], x[t_i + 1]], [batch_x[idx][t_i], batch_x[idx][t_i + 1]], '-', color='blue', label='original')
        axs[1].plot([x[t_i], x[t_i + 1]], [pred_batch_x[idx][t_i], pred_batch_x[idx][t_i + 1]], '-', color='orange', label='prediction')

    axs[0].set_title('Multi-masked time series')
    axs[0].set_xlabel('X - time points')
    axs[0].set_ylabel('Y - time values')

    axs[1].set_title('Original vs Reconstruction')
    axs[1].set_xlabel('X - time points')
    axs[1].set_ylabel('Y - time values')

    return fig

def show_token(token):

    token = token.data.cpu().numpy()
    fig_tokens = plt.figure(figsize=(12, 10))
    sns.heatmap(token, cmap='coolwarm', vmin=np.min(token), vmax=np.max(token), annot=False, square=False)

    return fig_tokens

def show_matrix(logits, positive_matrix, rebuild_weight_matrix):

    logits = logits.cpu().numpy()
    fig_logits = plt.figure(figsize=(80, 80))
    sns.heatmap(logits, cmap='coolwarm', vmin=np.min(logits), vmax=np.max(logits), annot=False, fmt='.2f', square=False)

    positive_matrix = positive_matrix.cpu().numpy()
    fig_positive_matrix = plt.figure(figsize=(80, 80))
    sns.heatmap(positive_matrix, cmap='coolwarm', vmin=0, vmax=1, annot=False, fmt='.1f', square=False)

    rebuild_weight_matrix = rebuild_weight_matrix.cpu().numpy()
    fig_rebuild_weight_matrix = plt.figure(figsize=(100, 100))
    sns.heatmap(rebuild_weight_matrix, cmap='coolwarm', vmin=0, vmax=1, annot=False, fmt='.3f', square=False)

    return fig_logits, fig_positive_matrix, fig_rebuild_weight_matrix


class ContrastiveWeight(nn.Module):

    def __init__(self, args):
        super(ContrastiveWeight, self).__init__()
        self.temperature = args.temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.positive_nums = args.positive_nums

    def get_positive_and_negative_mask(self, similarity_matrix, cur_batch_size):
        diag = np.eye(cur_batch_size)
        mask = torch.from_numpy(diag)
        mask = mask.type(torch.bool)

        oral_batch_size = cur_batch_size // (self.positive_nums + 1)

        positives_mask = np.zeros(similarity_matrix.size())
        for i in range(self.positive_nums + 1):
            ll = np.eye(cur_batch_size, cur_batch_size, k=oral_batch_size * i)
            lr = np.eye(cur_batch_size, cur_batch_size, k=-oral_batch_size * i)
            positives_mask += ll
            positives_mask += lr

        positives_mask = torch.from_numpy(positives_mask).to(similarity_matrix.device)
        positives_mask[mask] = 0

        negatives_mask = 1 - positives_mask
        negatives_mask[mask] = 0

        return positives_mask.type(torch.bool), negatives_mask.type(torch.bool)

    def forward(self, batch_emb_om):
        cur_batch_shape = batch_emb_om.shape

        # get similarity matrix among mask samples
        norm_emb = F.normalize(batch_emb_om, dim=1)
        similarity_matrix = torch.matmul(norm_emb, norm_emb.transpose(0, 1))

        # get positives and negatives similarity
        positives_mask, negatives_mask = self.get_positive_and_negative_mask(similarity_matrix, cur_batch_shape[0])

        positives = similarity_matrix[positives_mask].view(cur_batch_shape[0], -1)
        negatives = similarity_matrix[negatives_mask].view(cur_batch_shape[0], -1)

        # generate predict and target probability distributions matrix
        logits = torch.cat((positives, negatives), dim=-1)
        y_true = torch.cat((torch.ones(cur_batch_shape[0], positives.shape[-1]), torch.zeros(cur_batch_shape[0], negatives.shape[-1])), dim=-1).to(batch_emb_om.device).float()

        # multiple positives - KL divergence
        predict = self.log_softmax(logits / self.temperature)
        loss = self.kl(predict, y_true)

        return loss, similarity_matrix, logits, positives_mask


class AggregationRebuild(torch.nn.Module):

    def __init__(self, args):
        super(AggregationRebuild, self).__init__()
        self.args = args
        self.temperature = args.temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.positive_nums = args.positive_nums

    def forward(self, similarity_matrix, batch_emb_om):

        cur_batch_shape = batch_emb_om.shape

        # get the weight among (oral, oral's masks, others, others' masks)
        similarity_matrix /= self.temperature

        similarity_matrix = similarity_matrix - torch.eye(cur_batch_shape[0]).to(similarity_matrix.device).float() * 1e12
        rebuild_weight_matrix = self.softmax(similarity_matrix)

        batch_emb_om = batch_emb_om.reshape(cur_batch_shape[0], -1)

        # generate the rebuilt batch embedding (oral, others, oral's masks, others' masks)
        rebuild_batch_emb = torch.matmul(rebuild_weight_matrix, batch_emb_om)

        # get oral' rebuilt batch embedding
        rebuild_oral_batch_emb = rebuild_batch_emb.reshape(cur_batch_shape[0], cur_batch_shape[1], -1)

        return rebuild_weight_matrix, rebuild_oral_batch_emb

def lineage_search(a, b, x):
    if b == 0 or b is None:
        return 0

    segment_length = a / b
    for i in range(b):
        if i * segment_length <= x < (i + 1) * segment_length:
            return i
    return b - 1

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    #pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]

        Returns:
            output Tensor of shape [batch_size, seq_len, d_model]
        """

        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)


def patch_random_masking(xb, mask_ratio=0.4):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, L, nvars, device=xb.device)  # noise in [0, 1], bs x L x nvars

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]  # ids_keep: [bs x len_keep x nvars]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))  # x_kept: [bs x len_keep x nvars  x patch_len]

    # removed x
    x_removed = torch.zeros(bs, L - len_keep, nvars, D,
                            device=xb.device)  # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, D))  # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch x nvars]
    return x_masked, x_kept, mask, ids_restore


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def jitter(x, loc=0, sigma=1):
    # return np.random.normal(loc=loc, scale=sigma, size=x.shape)

    noise = torch.normal(mean=loc, std=sigma, size=x.shape).to(x.device)
    return noise

def get_mean_std(x):
    mean = x.mean().detach()  # 计算整体均值
    stdev = torch.sqrt(x.var(unbiased=False))

    return mean, stdev  # 返回整体均值和标准差

def permute_batch(x):
    batch_size = x.shape[0]
    index = torch.randperm(batch_size)
    noise = x[index, :, :]
    return noise