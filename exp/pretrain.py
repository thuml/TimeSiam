from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, transfer_weights
from utils.augmentations import masked_data
from utils.metrics import metric
from utils.losses import MaskedMSELoss
from utils.masking import mask_function
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import shutil
from tensorboardX import SummaryWriter
import random
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter
warnings.filterwarnings('ignore')


class Exp_Pretrain_PatchTST(Exp_Basic):

    def __init__(self, args):
        super(Exp_Pretrain_PatchTST, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))
            model = transfer_weights(self.args.load_checkpoints, model)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # print out the model size
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

        return model

    def _get_data(self, flag):

        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def pretrain(self):

        print("{}>\t mask_rule: patch_masking\tmask_rate: {}".format('-'*50, self.args.mask_rate))

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim, T_max=self.args.train_epochs)
        min_vali_loss = None

        for epoch in range(self.args.train_epochs):
            start_time = time.time()

            train_loss = self.pretrain_one_epoch(train_loader, model_optim, model_scheduler, epoch)
            vali_loss = self.valid_one_epoch(vali_loader, epoch)

            end_time = time.time()
            print("Epoch: {0}, Lr: {1:.7f}, Time: {2:.2f}s | Train Loss: {3:.4f} Val Loss: {4:.4f}"
                  .format(epoch, model_scheduler.get_lr()[0], end_time-start_time, train_loss, vali_loss))

            loss_scalar_dict = {
                'train_loss': train_loss,
                'vali_loss': vali_loss,
            }

            self.writer.add_scalars(f"/pretrain_loss", loss_scalar_dict, epoch)

            # checkpoint saving
            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print("Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...".format(min_vali_loss, vali_loss, epoch))

                min_vali_loss = vali_loss
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.model.state_dict()}
                torch.save(encoder_ckpt, os.path.join(path, "ckpt_best.pth"))

            if (epoch + 1) % 5 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.model.state_dict()}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))

    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler, epoch):

        train_loss = []

        self.model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()

            # To device
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)

            # model
            loss, _, _, _ = self.model(batch_x, batch_x_mark) # pred/target: [bs x num_patch x n_vars x patch_len] mask: [bs x num_patch x n_vars]

            # Backward
            loss.backward()
            model_optim.step()
            train_loss.append(loss.item())

        model_scheduler.step()
        train_loss = np.average(train_loss)

        return train_loss

    def valid_one_epoch(self, vali_loader, epoch):
        valid_loss = []

        self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

            # To device
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)

            # model
            loss, _, _, _ = self.model(batch_x, batch_x_mark) # pred/target: [bs x num_patch x n_vars x patch_len] mask: [bs x num_patch x n_vars]
            valid_loss.append(loss.item())

        vali_loss = np.average(valid_loss)

        self.model.train()
        return vali_loss


class Exp_Pretrain_TimeSiam(Exp_Basic):

    def __init__(self, args):
        super(Exp_Pretrain_TimeSiam, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")
        self.loss_fn = self._select_criterion()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))
            model = transfer_weights(self.args.load_checkpoints, model, device=self.device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # print model size
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return MaskedMSELoss()

    def pretrain(self):

        print("{}> pretrain seq len: {} \t sampling_range: {}*{} \t lineage_tokens: {} \t mask_rule: {} \t mask_rate: {} \t tokens_using: {} \t representation_using: {}<{}"
              .format('-'*50, self.args.seq_len, self.args.sampling_range, self.args.seq_len, self.args.lineage_tokens, self.args.masked_rule, self.args.mask_rate, self.args.tokens_using, self.args.representation_using, '-'*50))

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim, T_max=self.args.train_epochs)
        min_vali_loss = None

        for epoch in range(self.args.train_epochs):
            start_time = time.time()

            train_loss = self.pretrain_one_epoch(train_loader, model_optim, model_scheduler)
            vali_loss = self.valid_one_epoch(vali_loader)

            end_time = time.time()
            print("Epoch: {0}, Lr: {1:.7f}, Time: {2:.2f}s | Train Loss: {3:.4f} Val Loss: {4:.4f}"
                  .format(epoch, model_scheduler.get_lr()[0], end_time-start_time, train_loss, vali_loss))

            loss_scalar_dict = {
                'train_loss': train_loss,
                'vali_loss': vali_loss,
            }

            self.writer.add_scalars(f"/pretrain_loss", loss_scalar_dict, epoch)

            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print("Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...".format(min_vali_loss, vali_loss, epoch))
                min_vali_loss = vali_loss
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.model.state_dict()}
                torch.save(encoder_ckpt, os.path.join(path, "ckpt_best.pth"))

            if (epoch + 1) % 5 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.model.state_dict()}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))

    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):

        train_loss = []

        self.model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, segment) in enumerate(train_loader):
            model_optim.zero_grad()

            past_x, cur_x = batch_x, batch_y

            _, cur_x_, mask = mask_function(cur_x, self.args)

            # to device
            past_x = past_x.float().to(self.device)
            cur_x = cur_x.float().to(self.device)
            cur_x_ = cur_x_.float().to(self.device)
            mask = mask.to(self.device)

            # Encoder
            pred_cur = self.model(past_x, None, cur_x_, None, segment=segment, mask=mask)

            if self.args.mask_rate == 0:
                loss = self.loss_fn(pred_cur, cur_x)
            else:
                loss = self.loss_fn(pred_cur, cur_x, ~mask)

            # Backward
            loss.backward()
            model_optim.step()
            train_loss.append(loss.item())

        model_scheduler.step()
        train_loss = np.average(train_loss)

        return train_loss

    def valid_one_epoch(self, vali_loader):
        valid_loss = []

        self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, segment) in enumerate(vali_loader):

            past_x, cur_x = batch_x, batch_y

            _, cur_x_, mask = mask_function(cur_x, self.args)

            # to device
            past_x = past_x.float().to(self.device)
            cur_x = cur_x.float().to(self.device)
            cur_x_ = cur_x_.float().to(self.device)
            mask = mask.to(self.device)

            # Encoder
            pred_cur = self.model(past_x, None, cur_x_, None, segment=segment, mask=mask)

            if self.args.mask_rate == 0:
                loss = self.loss_fn(pred_cur, cur_x)
            else:
                loss = self.loss_fn(pred_cur, cur_x, ~mask)

            valid_loss.append(loss.item())

        vali_loss = np.average(valid_loss)

        self.model.train()
        return vali_loss


class Exp_Train(Exp_Basic):
    def __init__(self, args):
        super(Exp_Train, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")
        self.iters = 0

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))

            model = transfer_weights(self.args.load_checkpoints, model)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # print out the model size
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

        return model

    def _get_data(self, flag):

        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            start_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                if self.args.select_channels < 1:
                    # Random
                    B, S, C = batch_x.shape
                    random_C = int(C * self.args.select_channels)

                    if random_C < 1:
                        random_C = 1

                    index = torch.LongTensor(random.sample(range(C), random_C))
                    batch_x = torch.index_select(batch_x, 2, index)
                    batch_y = torch.index_select(batch_y, 2, index)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                
                self.iters += 1

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            end_time = time.time()
            print("Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                epoch + 1, train_steps, end_time-start_time, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)

            loss_scalar_dict = {
                'train_loss': train_loss,
                'valid_loss': vali_loss,
                'test_loss': test_loss,
            }
            self.writer.add_scalars(f"/epochs_loss", loss_scalar_dict, epoch + 1)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.lr = model_optim.param_groups[0]['lr']

        return self.model

    def vali(self, vali_data, vali_loader, criterion, index=None):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                if index is not None:
                    batch_x = torch.index_select(batch_x, 2, index)
                    batch_y = torch.index_select(batch_y, 2, index)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting=None, test=0, log=1, iters=None):
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        folder_path = './outputs/test_results/{}'.format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        if log:
            if iters is not None:
                log = 'iter, {0}, {1}->{2}, {3:.3f}, {4:.3f}'.format(iters, self.args.seq_len, self.args.pred_len, mse, mae)
            else:
                log = 'ep, {0}, {1}->{2}, {3:.3f}, {4:.3f}'.format(self.args.train_epochs, self.args.seq_len, self.args.pred_len, mse, mae)

            print(log)
            f = open(f"{folder_path}/{self.args.task_name}_results.txt", 'a')
            f.write(log + '\n')
            f.close()

    def freeze(self):
        """
        freeze the model head
        require the model to have head attribute
        """

        if hasattr(get_model(self.model), 'head'):
            for name, param in get_model(self.model).named_parameters():
                param.requires_grad = False
            for name, param in get_model(self.model).named_parameters():
                if 'head' in name:
                    param.requires_grad = True
                    # print('unfreeze:', name)
            print('model is frozen except the head!')

    def freeze_part(self):
        """
        freeze the model head
        require the model to have head attribute
        """

        if hasattr(get_model(self.model), 'head'):
            for name, param in get_model(self.model).named_parameters():
                param.requires_grad = False
            for name, param in get_model(self.model).named_parameters():
                if 'enc_embedding' in name or 'norm' in name or 'head' in name:
                    param.requires_grad = True
                    print('unfreeze:', name)
                else:
                    print('freeze:', name)

    def unfreeze(self):
        for name, param in get_model(self.model).named_parameters():
            if 'token' in name:
                param.requires_grad = False
                # print('freeze:', name)
                continue
            param.requires_grad = True
            # print('unfreeze:', name)

    def fine_tune(self, setting):
        """
        Finetune the entire network
        """
        if self.args.task_name == 'long_term_forecast':
            print('Training the entire network!')
            self.unfreeze()
        elif self.args.task_name == 'fine_tune':
            print('Fine-tuning the entire network!')
            self.unfreeze()
        elif self.args.task_name == 'fine_tune_part':
            print('Fine-tuning part network!')
            self.freeze_part()
        elif self.args.task_name == 'linear_probe':
            print('Fine-tuning the head!')
            self.freeze()
        else:
            raise ValueError("Wrong task_name {}!".format(self.args.task_name))

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        self.train(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        self.test(setting)


def get_model(model):
    "Return the model maybe wrapped inside `model`."
    return model.module if isinstance(model, (DistributedDataParallel, nn.DataParallel)) else model