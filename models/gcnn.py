# 引入GCN网络对空间数据进行编码和自学习。
# 图卷积网络考虑图的动态学习能力，引入注意力机制。
# 考虑图谱卷积中矩阵乘法的采样计算方式，而非全矩阵的乘法

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dynamic_gcn import spatialAttentionGCN, spatialAttentionScaledGCN
from utils import gcn_tools


class GCNN(nn.Module):
    dataset_dict = {
        'ETTh1': gcn_tools.get_ETTh1_adjacency_matrix,
        'ETTh2': gcn_tools.get_ETTh2_adjacency_matrix,
        'ECL': gcn_tools.get_ECL_adjacency_matrix,
        'WTH': gcn_tools.get_WTH_adjacency_matrix,
        'ETTm1': gcn_tools.get_ETTm1_adjacency_matrix,
        'PEMS03':gcn_tools.get_PEMS03_adjacency_matrix,
        'GEN':gcn_tools.get_GEN_adjacency_matrix,
        'EXC':gcn_tools.get_EXC_adjacency_matrix,
        'ILI':gcn_tools.get_ILI_adjacency_matrix,
        'Traffic':gcn_tools.get_Traffic_adjacency_matrix,
    }

    def __init__(self, data, in_channels, out_channels, dropout=0.0):
        super(GCNN, self).__init__()

        # 默认构造全连接的邻接矩阵
        adj_matrix = GCNN.dataset_dict[data]()
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        norm_adj_matrix = torch.from_numpy(gcn_tools.norm_Adj(adj_matrix)).type(torch.FloatTensor)

        # self.spatial_attention_GCN_model = \
        self.spatial_attention_scaled_GCN_model = \
            spatialAttentionScaledGCN(sym_norm_Adj_matrix=norm_adj_matrix,
                                      in_channels=in_channels,
                                      out_channels=out_channels,
                                      dropout=dropout)
        self.adj_matrix_shape = adj_matrix.shape

        print("生成GCN模型，邻接矩阵大小为：{}".format(self.adj_matrix_shape))

    def get_adj_matrix_shape(self):
        return self.adj_matrix_shape

    def forward(self, x):
        # print("实现GCN attention layer {}".format(x.shape))
        return self.spatial_attention_scaled_GCN_model(x=x)
