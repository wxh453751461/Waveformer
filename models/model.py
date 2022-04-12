import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import models.gcnn
from models.AutoCorrelation import AutoCorrelation
from models.SeriesDecomp import series_decomp
from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding, TemporalEmbedding, TimeFeatureEmbedding
from models.interactor import Interactor, Splitting
from models.gcnn import GCNN
from utils import gcn_tools
import pywt

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0'),dataname='ETTh1',decomp = 'DWT'):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embeddingA1 = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.enc_embeddingD1 = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embeddingA1 = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        self.dec_embeddingD1 = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        if attn == 'prob':
            Attn = ProbAttention
        elif attn == 'full':
            Attn = FullAttention
        else:
            Attn = AutoCorrelation
        # Encoder
        self.encoderA1 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.encoderD1 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoderA1 = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    dataname=dataname
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.decoderD1 = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    dataname=dataname
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection_x_A1 = nn.Linear(d_model, c_out, bias=True)
        self.projection_x_D1 = nn.Linear(d_model, c_out, bias=True)


        # 增加图卷积模块，学习节点之间的项目空间位置关系

        self.enc_gcnn_layerA1 = GCNN(data=dataname,
                                   in_channels=1,
                                   out_channels=d_model)

        self.enc_gcnn_layerD1 = GCNN(data=dataname,
                                   in_channels=1,
                                   out_channels=d_model)

        self.enc_conv1A1 = nn.Conv1d(in_channels=d_model * self.enc_gcnn_layerA1.get_adj_matrix_shape()[0],
                                   out_channels=d_model,
                                   kernel_size=3, padding=1, padding_mode='circular')

        self.enc_conv1D1 = nn.Conv1d(in_channels=d_model * self.enc_gcnn_layerD1.get_adj_matrix_shape()[0],
                                   out_channels=d_model,
                                   kernel_size=3, padding=1, padding_mode='circular')

        self.decomp = decomp
        if self.decomp == 'DWT':
            inputdem = int(seq_len/2)
        else:
            inputdem = int(seq_len)


        self.enc_conv2A1 = nn.Conv1d(in_channels=inputdem, out_channels=int (inputdem/ (2 ** (e_layers - 1))), kernel_size=1)
        self.enc_conv2D1 = nn.Conv1d(in_channels=inputdem, out_channels=int (inputdem/ (2 ** (e_layers - 1))), kernel_size=1)


        self.interactor = Interactor(in_planes=d_model,splitting=False,kernel = 5, dropout=0.5, groups = 1, hidden_size = 1, INN = True)
        self.interactor2 = Interactor(in_planes=d_model,splitting=False,kernel = 5, dropout=0.5, groups = 1, hidden_size = 1, INN = True)


        self.dec_gcnn_layerA1 = GCNN(data=dataname,
                                   in_channels=1,
                                   out_channels=d_model)
        self.dec_gcnn_layerD1 = GCNN(data=dataname,
                                   in_channels=1,
                                   out_channels=d_model)
        self.dec_conv2A1 = nn.Conv1d(in_channels=d_model * self.dec_gcnn_layerA1.get_adj_matrix_shape()[0],
                                   out_channels=d_model,
                                   kernel_size=3, padding=1, padding_mode='circular')
        self.dec_conv2D1 = nn.Conv1d(in_channels=d_model * self.dec_gcnn_layerD1.get_adj_matrix_shape()[0],
                                   out_channels=d_model,
                                   kernel_size=3, padding=1, padding_mode='circular')

        self.dec_attA1=AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                       d_model, n_heads, mix=mix)
        self.dec_attD1=AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                       d_model, n_heads, mix=mix)

        self.temporal_embedding_enc = TemporalEmbedding(d_model=d_model, embed_type=embed,
                                                    freq=freq) if embed != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed, freq=freq)

        self.temporal_embedding_dec = TemporalEmbedding(d_model=d_model, embed_type=embed,
                                                    freq=freq) if embed != 'timeF' else TimeFeatureEmbedding(
        d_model=d_model, embed_type=embed, freq=freq)

        self.decomp_enc = series_decomp(kernel_size = 25)
        self.decomp_dec = series_decomp(kernel_size = 25)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        #Encoder

        # time_enc = self.temporal_embedding_enc(x_mark_enc)
        # time_enc_odd = time_enc[:,1::2,:]
        # time_enc_even = time_enc[:,::2,:]

        if self.decomp == 'DWT':

            device = x_enc.device
            enc_A1, enc_D1 = pywt.wavedec(x_enc.cpu().transpose(1,2), 'haar', level=1)

            enc_A1 = torch.from_numpy(enc_A1)
            enc_D1 = torch.from_numpy(enc_D1)
            enc_A1 = enc_A1.transpose(1,2).to(device)
            enc_D1 = enc_D1.transpose(1,2).to(device)

            enc_A1_out = self.enc_embeddingA1(enc_A1)
            enc_D1_out = self.enc_embeddingD1(enc_D1)
        else:
            enc_A1, enc_D1 = self.decomp_enc(x_enc)

            enc_A1_out = self.enc_embeddingA1(enc_A1) + self.temporal_embedding_enc(x_mark_enc)
            enc_D1_out = self.enc_embeddingD1(enc_D1) + self.temporal_embedding_enc(x_mark_enc)

        enc_A1_out, attns_A1 = self.encoderA1(enc_A1_out, attn_mask=enc_self_mask)
        enc_D1_out, attns_D1 = self.encoderD1(enc_D1_out, attn_mask=enc_self_mask)

        B, L, D = enc_A1.shape
        gcn_in_A1 = enc_A1.transpose(1, 2).reshape(B, D, L, 1)
        gcn_out_A1 = self.enc_gcnn_layerA1(gcn_in_A1).transpose(1, 2).reshape(B, L, -1)
        gcn_out_A1 = self.enc_conv1A1(gcn_out_A1.permute(0, 2, 1)).transpose(1, 2)
        gcn_out_A1 = self.enc_conv2A1(gcn_out_A1)
        enc_A1_out = enc_A1_out + gcn_out_A1

        gcn_in_D1 = enc_D1.transpose(1, 2).reshape(B, D, L, 1)
        gcn_out_D1 = self.enc_gcnn_layerA1(gcn_in_D1).transpose(1, 2).reshape(B, L, -1)
        gcn_out_D1 = self.enc_conv1A1(gcn_out_D1.permute(0, 2, 1)).transpose(1, 2)
        gcn_out_D1 = self.enc_conv2A1(gcn_out_D1)
        enc_D1_out = enc_D1_out + gcn_out_D1

        enc_A1_out, enc_D1_out = self.interactor(enc_A1_out,enc_D1_out)


        #Decoder


        # time_dec = self.temporal_embedding_dec(x_mark_dec)
        # time_dec_odd = time_dec[:,1::2,:]
        # time_dec_even = time_dec[:,::2,:]

        if self.decomp == 'DWT':


            device = x_dec.device
            dec_A1, dec_D1 = pywt.wavedec(x_dec.cpu().transpose(1,2), 'haar', level=1)
            dec_A1 = torch.from_numpy(dec_A1)
            dec_D1 = torch.from_numpy(dec_D1)
            dec_A1 = dec_A1.transpose(1,2).to(device)
            dec_D1 = dec_D1.transpose(1,2).to(device)
            dec_A1_out = self.dec_embeddingA1(dec_A1)
            dec_D1_out = self.dec_embeddingD1(dec_D1)

        else:
            dec_A1, dec_D1 = self.decomp_dec(x_dec)
            dec_A1_out = self.dec_embeddingA1(dec_A1) + self.temporal_embedding_dec(x_mark_dec)
            dec_D1_out = self.dec_embeddingD1(dec_D1) + self.temporal_embedding_dec(x_mark_dec)

        # inter_decoder = time.time()


        B, L, D = dec_A1.shape

        de_gcn_in_A1 = dec_A1.transpose(1, 2).reshape(B, D, L, 1)
        de_gcn_out_A1 = self.dec_gcnn_layerA1(de_gcn_in_A1).transpose(1, 2).reshape(B, L, -1)
        de_gcn_out_A1 = self.dec_conv2A1(de_gcn_out_A1.permute(0, 2, 1)).transpose(1, 2)

        de_gcn_in_D1 = dec_D1.transpose(1, 2).reshape(B, D, L, 1)
        de_gcn_out_D1 = self.dec_gcnn_layerD1(de_gcn_in_D1).transpose(1, 2).reshape(B, L, -1)
        de_gcn_out_D1 = self.dec_conv2D1(de_gcn_out_D1.permute(0, 2, 1)).transpose(1, 2)

        dec_A1_out = self.dec_attA1(dec_A1_out,dec_A1_out,dec_A1_out,attn_mask=None)[0] + dec_A1_out
        dec_D1_out = self.dec_attD1(dec_D1_out,dec_D1_out,dec_D1_out,attn_mask=None)[0] + dec_D1_out

        dec_A1_out = dec_A1_out + de_gcn_out_A1
        dec_D1_out = dec_D1_out + de_gcn_out_D1
        dec_A1_out, dec_D1_out = self.interactor(dec_A1_out,dec_D1_out)

        dec_out_A1 = self.decoderA1(dec_A1_out, enc_A1_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out_D1 = self.decoderD1(dec_D1_out, enc_D1_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        dec_out_x_A1 = self.projection_x_A1(dec_out_A1)
        dec_out_x_D1 = self.projection_x_D1(dec_out_D1)


        if self.decomp == 'DWT':

            if self.output_attention:
                return dec_out_x_A1[:, -int(self.pred_len / 2):, :], dec_out_x_D1[:, -int(self.pred_len / 2):, :], attns_A1, attns_D1
            else:
                return dec_out_x_A1[:, -int(self.pred_len / 2):, :], dec_out_x_D1[:, -int(self.pred_len / 2):, :]
        else:
            dec_out = dec_out_x_D1 + dec_out_x_A1
            if self.output_attention:
                return dec_out[:, -(self.pred_len):, :]
            else:
                return dec_out[:, -(self.pred_len):, :]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1], d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers)))  # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, duration_enc, duration_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, duration_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, duration_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
