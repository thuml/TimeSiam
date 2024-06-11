import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, Siamese_DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from utils.tools import ContrastiveWeight, AggregationRebuild
from utils.losses import AutomaticWeightedLoss
from utils.tsne import visualization, visualization_PCA, visualization_token_PCA


class Flatten_Head(nn.Module):
    def __init__(self, nf, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-1)
        self.linear = nn.Linear(nf, pred_len, bias=True)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x   # x: [bs x nvars x seq_len]

class Pooler_Head(nn.Module):
    def __init__(self, n_vars, d_model, head_dropout=0):
        super().__init__()

        pn = n_vars * d_model
        dimension = 128
        self.pooler = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(pn, pn // 2),
            nn.BatchNorm1d(pn // 2),
            nn.ReLU(),
            nn.Linear(pn // 2, dimension),
            nn.Dropout(head_dropout),
        )

    def forward(self, x):  # [bs x n_vars x d_model]
        x = self.pooler(x) # [bs x dimension]
        return x

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.lineage_tokens = configs.lineage_tokens
        self.tokens_using = configs.tokens_using
        self.configs = configs

        self.d_model = configs.d_model

        self.visual = 0
        self.visual_path = None

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Temporal shift token
        if self.lineage_tokens:
            print("init {} lineage tokens!".format(self.lineage_tokens + 1))

            if self.configs.current_token:
                self.token_0 = nn.Parameter(torch.zeros(1, 1, configs.d_model), requires_grad=True) # current lineage token

            for i in range(self.lineage_tokens):
                setattr(self, f'token_{i + 1}', nn.Parameter(torch.zeros(1, 1, configs.d_model), requires_grad=True)) # past lineage token
            self.initialize_weights()

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        if self.task_name == 'simmtm':

            # for series-wise representation
            self.pooler = Pooler_Head(configs.enc_in, configs.d_model, head_dropout=configs.head_dropout)

            # for reconstruction
            self.projection = Flatten_Head(configs.d_model, configs.seq_len, head_dropout=configs.head_dropout)

            self.awl = AutomaticWeightedLoss(2)
            self.contrastive = ContrastiveWeight(self.configs)
            self.aggregation = AggregationRebuild(self.configs)
            self.mse = torch.nn.MSELoss()

        elif self.task_name == 'timesiam':

            self.decoder = Decoder(
                [
                    Siamese_DecoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.seq_len, bias=True)
            )

        if self.task_name in ['linear_probe', 'fine_tune', 'fine_tune_part']:
            self.representation_using = configs.representation_using

            if self.lineage_tokens and self.representation_using == 'concat':
                print("{}>  Representation ({}), head dimension {}*{}".format('-'*50, self.representation_using, configs.d_model, self.lineage_tokens))
                self.head = Flatten_Head(configs.d_model*self.lineage_tokens, configs.pred_len, head_dropout=configs.head_dropout)
            else:
                print("{}> Representation ({}), head dimension {}".format('-'*50, self.representation_using, configs.d_model))
                self.head = Flatten_Head(configs.d_model, configs.pred_len, head_dropout=configs.head_dropout)
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.head = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.head = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.head = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.head = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def initialize_weights(self):
        for i in range(self.lineage_tokens):
            token = getattr(self, f'token_{i + 1}')
            torch.nn.init.normal_(token, std=.02)

    def norm(self, x, mask=None):
        if mask is not None:
            means = torch.sum(x, dim=1) / torch.sum(mask == 1, dim=1)
            means = means.unsqueeze(1).detach()
            x = x - means
            x = x.masked_fill(mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x * x, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
            stdev = stdev.unsqueeze(1).detach()
            x /= stdev
        else:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        return x, means, stdev

    def denorm(self, x, means, stdev, L):
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return x

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape   # x_enc: [Batch Time Variate]

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.head(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def temperal_shift_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape   # x_enc: [Batch Time Variate]

        # Embedding
        enc_emb = self.enc_embedding(x_enc) # [bs x n_vars x d_model]

        # add temperal shift token
        if self.lineage_tokens and self.lineage_tokens != 0:

            if self.representation_using == 'concat':
                outputs = []
                for i in range(self.lineage_tokens):

                    # get and add temperal shift token
                    token = getattr(self, f'token_{i + 1}')

                    if self.tokens_using == 'single':
                        enc_input = enc_emb + token.repeat(enc_emb.shape[0], enc_emb.shape[1], 1).to(enc_emb.device)
                    else:
                        enc_input = enc_emb + token.repeat(enc_emb.shape[0], 1, 1).to(enc_emb.device)

                    # encoder
                    enc_out, attns = self.encoder(enc_input, attn_mask=None) # [bs x n_vars x d_model]

                    outputs.append(enc_out)
                enc_out = torch.cat(outputs, dim=-1)
            elif self.representation_using == 'avg':
                
                outputs = []

                # current token
                if self.configs.current_token:
                    # future token
                    enc_input = enc_emb + self.token_0.repeat(enc_emb.shape[0], enc_emb.shape[1], 1).to(enc_emb.device)
                    enc_out, attns = self.encoder(enc_input, attn_mask=None)
                    outputs = [enc_out]

                # past token
                for i in range(self.lineage_tokens):

                    # get and add temperal shift token
                    token = getattr(self, f'token_{i + 1}')
                    enc_input = enc_emb + token.repeat(enc_emb.shape[0], enc_emb.shape[1], 1).to(enc_emb.device)

                    # encoder
                    enc_out, attns = self.encoder(enc_input, attn_mask=None)
                    outputs.append(enc_out)

                enc_out = torch.stack(outputs).mean(dim=0) # [bs x n_vars x d_model]
        else:
            # encoder
            enc_out, attns = self.encoder(enc_emb, attn_mask=None)

        dec_out = self.head(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def temperal_extend_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        bs, seq_len, n_vars = x_enc.shape # [bs x seq_len x n_vars]
        new_seq_len = seq_len // self.lineage_tokens  # equal pre-training sequence length
        x_enc = x_enc.reshape(bs, self.lineage_tokens, new_seq_len, n_vars) # [bs x n x new_seq_len x n_vars]

        outputs = []
        for i in range(self.lineage_tokens):
            new_x_enc = x_enc[:, i, :, :] # [bs x new_seq_len x n_vars]

            # Embedding
            enc_emb = self.enc_embedding(new_x_enc)  # [bs x n_vars x d_model]
            token = getattr(self, f'token_{i + 1}')
            enc_input = enc_emb + token.repeat(enc_emb.shape[0], enc_emb.shape[1], 1).to(enc_emb.device)
            enc_out, attns = self.encoder(enc_input, attn_mask=None)  # [bs x n_vars x d_model]

            outputs.append(enc_out)

        enc_out = torch.cat(outputs, dim=-1)  # [bs x n_vars x (n * d_model)]

        dec_out = self.head(enc_out).permute(0, 2, 1) # [bs x seq_len x n_vars]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.head(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.head(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.head(output)  # (batch_size, num_classes)
        return output

    def pretrain_simmtm(self, x_enc, x_mark_enc, batch_x, mask):

        # Normalization
        x_enc, means, stdev = self.norm(x_enc, mask)

        # Encoder
        enc_out = self.enc_embedding(x_enc)
        p_enc_out, attns = self.encoder(enc_out)  # p_enc_out: [bs x n_vars x d_model]

        # series-wise representation
        s_enc_out = self.pooler(p_enc_out) # s_enc_out: [bs x dimension]

        # series weight learning
        loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(s_enc_out) # similarity_matrix: [bs x bs]
        rebuild_weight_matrix, agg_enc_out = self.aggregation(similarity_matrix, p_enc_out) # agg_enc_out: [bs x n_vars x d_model]

        # Decoder
        dec_out = self.projection(agg_enc_out)  # [bs x n_vars x seq_len]
        dec_out = dec_out.permute(0, 2, 1)  # [bs x seq_len x n_vars]

        # De-Normalization
        dec_out = self.denorm(dec_out, means, stdev, self.seq_len)

        pred_batch_x = dec_out[:batch_x.shape[0]]

        # series reconstruction
        loss_rb = self.mse(pred_batch_x, batch_x.detach())

        # loss
        loss = self.awl(loss_cl, loss_rb)

        return loss, loss_cl, loss_rb, positives_mask, logits, rebuild_weight_matrix, pred_batch_x

    def pretrain_timesiam(self, past_x_enc, past_x_mark_enc, fur_x_enc, fur_x_mark_enc, segment, mask):

        # Normalization
        past_x_enc, _, _ = self.norm(past_x_enc) # [bs x seq_len x n_vars]
        fur_x_enc, means, stdev = self.norm(fur_x_enc, mask=mask) # [bs x seq_len x n_vars]

        """past window"""
        # past window embedding
        past_x_enc = self.enc_embedding(past_x_enc, past_x_mark_enc) # [bs x n_vars x d_model]

        # add temperal shift token
        if self.lineage_tokens:
            selected_tensors = []
            for i in segment:
                token = getattr(self, f'token_{i + 1}')
                selected_tensors.append(token)
            lineages = torch.cat(selected_tensors, dim=0)
            lineages = lineages.repeat(1, past_x_enc.shape[1], 1).to(past_x_enc.device)  # [bs x n_vars x d_model]
            past_x_enc = past_x_enc + lineages

        # encoder - past window
        past_enc_out, attns = self.encoder(past_x_enc, attn_mask=None) # past_enc_out: [bs x n_vars x d_model]

        """future window"""
        # current window embedding
        fur_x_enc = self.enc_embedding(fur_x_enc, fur_x_mark_enc)

        # add current temporal shift token
        if self.lineage_tokens and self.configs.current_token:
            fur_x_enc = fur_x_enc + self.token_0.repeat(fur_x_enc.shape[0], fur_x_enc.shape[1], 1).to(fur_x_enc.device)  # cur_x_enc: [(bs*n_vars)  x patch_num x d_model]

        # encoder - future window
        fur_enc_out, attns = self.encoder(fur_x_enc, attn_mask=None) # cur_enc_out: [bs x n_vars x d_model]

        """cross past and current window"""
        # decoder - cross attention
        dec_out = self.decoder(fur_enc_out, past_enc_out).permute(0, 2, 1) # dec_out: [bs x seq_len x n_vars]

        # De-Normalization
        dec_out = self.denorm(dec_out, means, stdev, self.seq_len)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, segment=None, mask=None):
        if self.task_name == 'simmtm':
            return self.pretrain_simmtm(x_enc, x_mark_enc, x_dec, mask)
        if self.task_name == 'timesiam':
            return self.pretrain_timesiam(x_enc, x_mark_enc, x_dec, x_mark_dec, segment, mask)
        if  self.task_name in ['linear_probe', 'fine_tune', 'fine_tune_part']:
            return self.temperal_shift_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None