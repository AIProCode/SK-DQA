from openvqa.ops.fc import MLP
from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
from torch.nn.utils.weight_norm import weight_norm

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()
        self.mhatt = SARoutingBlock(__C)
        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)
        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)
        self.mha = nn.MultiheadAttention(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE, dropout=__C.DROPOUT_R, batch_first=True)

    def forward(self, frcn_feat, captions, x_masks, caption_mask, tau, training):
        """1"""
        frcn_feat = self.norm1(frcn_feat + self.dropout1(
            self.mhatt(v=frcn_feat, k=frcn_feat, q=frcn_feat, masks=x_masks, tau=tau, training=training)
        ))

        frcn_feat_intra = frcn_feat

        frcn_feat1 = self.norm3(frcn_feat + self.dropout3(
            self.mhatt2(captions, captions, frcn_feat, mask=None)
        ))
        ff = frcn_feat1

        frcn_feat = frcn_feat + 0.8 * frcn_feat1

        return frcn_feat, frcn_feat_intra, ff


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # h.shape: (B, N, in_features), Wh.shape: (B, N, out_features)
        B, N, _ = Wh.size()

        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, Wh)  # h_prime shape: (B, N, out_feature)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        B, N, H = Wh.size()
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating],
                                            dim=-1)  # Reshaping into [batch_size, num_nodes*num_nodes, 2 * feature_dim]
        e = torch.matmul(all_combinations_matrix.view(B, N * N, 2 * H), self.a).view(B, N,
                                                                                     N)  # Reshaping back to [batch_size, num_nodes, num_nodes] to get E.
        return self.leakyrelu(e)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True),
            GraphAttentionLayer(nhid, nout, dropout=dropout, alpha=alpha, concat=False)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.attentions[0](x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.attentions[1](x, adj)
        return x



class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()
        self.mhatt = SARoutingBlock(__C)
        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.mhatt3 = MHAtt(__C)
        self.mhatt4 = MHAtt(__C)

        self.gat1 = GAT(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.gat2 = GAT(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.ffn1 = FFN(__C)
        self.ffn2 = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout4 = nn.Dropout(__C.DROPOUT_R)
        self.norm4 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout5 = nn.Dropout(__C.DROPOUT_R)
        self.norm5 = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout6 = nn.Dropout(__C.DROPOUT_R)
        self.norm6 = LayerNorm(__C.HIDDEN_SIZE)

        # self.interaction_module = MultiModalInteraction(__C.HIDDEN_SIZE, 8)
    def _create_attention_mask(self, features1, features2):
        adj = F.softmax(torch.bmm(features1, features2.transpose(1, 2)), dim=-1)
        return adj

    def forward(self, frcn_feat, frcn_feat_intra, lang_feat, x_masks, frcn_feat_mask, lang_feat_mask, tau, training):

        frcn_feat2 = self.mhatt2(v=lang_feat, k=lang_feat, q=frcn_feat, mask=lang_feat_mask)
        # frcn_feat = self.norm2(frcn_feat + self.dropout2(frcn_feat2))
        frcn_feat_inter = frcn_feat2
        frcn_feat = frcn_feat_intra + 0.8 * frcn_feat_inter

        lang_feat1 = self.mhatt3(v=lang_feat, k=lang_feat, q=lang_feat, mask=lang_feat_mask)
        lang_feat_intra = lang_feat1
        lang_feat2 = self.mhatt4(v=frcn_feat, k=frcn_feat, q=lang_feat, mask=frcn_feat_mask)
        lang_feat_inter = lang_feat2
        lang_feat = lang_feat_intra + 0.1 * lang_feat_inter
        return frcn_feat, lang_feat, frcn_feat_inter


class MultiModalInteraction(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiModalInteraction, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 模态内和模态间的线性变换
        self.W_q_img = nn.Linear(dim, dim)
        self.W_k_img = nn.Linear(dim, dim)
        self.W_v_img = nn.Linear(dim, dim)
        self.W_q_ques = nn.Linear(dim, dim)
        self.W_k_ques = nn.Linear(dim, dim)
        self.W_v_ques = nn.Linear(dim, dim)

        # 输出投影
        self.out_proj_img = nn.Linear(dim, dim)
        self.out_proj_ques = nn.Linear(dim, dim)

    def forward(self, question_features, image_features, enhanced_img_intra, question_mask, image_mask):
        B, T, _ = question_features.shape
        _, N, _ = image_features.shape

        # 模态内注意力 - 图像特征
        # query_img = self.split_heads(self.W_q_img(image_features))
        # key_img = self.split_heads(self.W_k_img(image_features))
        # value_img = self.split_heads(self.W_v_img(image_features))
        # attn_img = self.attention(query_img, key_img, value_img, image_mask)
        # attn_img = self.combine_heads(attn_img)
        # enhanced_img_intra = self.out_proj_img(attn_img)

        # 模态内注意力 - 问题特征
        query_ques = self.split_heads(self.W_q_ques(question_features))
        key_ques = self.split_heads(self.W_k_ques(question_features))
        value_ques = self.split_heads(self.W_v_ques(question_features))
        attn_ques = self.attention(query_ques, key_ques, value_ques, question_mask)
        attn_ques = self.combine_heads(attn_ques)
        enhanced_ques_intra = self.out_proj_ques(attn_ques)

        # 模态间注意力 - 图像关注问题
        query_img_to_ques = self.split_heads(self.W_q_img(image_features))
        key_ques_for_img = self.split_heads(self.W_k_ques(question_features))
        value_ques_for_img = self.split_heads(self.W_v_ques(question_features))
        attn_img_to_ques = self.attention(query_img_to_ques, key_ques_for_img, value_ques_for_img, question_mask)
        attn_img_to_ques = self.combine_heads(attn_img_to_ques)
        enhanced_img_inter = self.out_proj_img(attn_img_to_ques)

        # 模态间注意力 - 问题关注图像
        query_ques_to_img = self.split_heads(self.W_q_ques(question_features))
        key_img_for_ques = self.split_heads(self.W_k_img(image_features))
        value_img_for_ques = self.split_heads(self.W_v_img(image_features))
        attn_ques_to_img = self.attention(query_ques_to_img, key_img_for_ques, value_img_for_ques, image_mask)
        attn_ques_to_img = self.combine_heads(attn_ques_to_img)
        enhanced_ques_inter = self.out_proj_ques(attn_ques_to_img)

        # 融合模态内和模态间的增强特征
        enhanced_img = enhanced_img_intra + enhanced_img_inter
        enhanced_ques = enhanced_ques_intra + enhanced_ques_inter

        return enhanced_img, enhanced_ques, enhanced_img_inter

    def split_heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        B, _, L, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, self.dim)

    def attention(self, query, key, value, mask):
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = mask.expand(-1, self.num_heads, -1, -1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, value)


def getImgMasks(scale=16, order=2):
    """
    :param scale: Feature Map Scale
    :param order: Local Window Size, e.g., order=2 equals to windows size (5, 5)
    :return: masks = (scale**2, scale**2)
    """
    masks = []
    _scale = scale
    assert order < _scale, 'order size be smaller than feature map scale'

    for i in range(_scale):
        for j in range(_scale):
            mask = np.ones([_scale, _scale], dtype=np.float32)
            for x in range(i - order, i + order + 1, 1):
                for y in range(j - order, j + order + 1, 1):
                    if (0 <= x < _scale) and (0 <= y < _scale):
                        mask[x][y] = 0
            mask = np.reshape(mask, [_scale * _scale])
            masks.append(mask)
    masks = np.array(masks)
    masks = np.asarray(masks, dtype=np.bool_)
    return masks

def getMasks(x_mask, __C):
    mask_list = []
    ORDERS = __C.ORDERS
    for order in ORDERS:
        if order == 0:
            mask_list.append(x_mask)
        else:
            mask = torch.from_numpy(getImgMasks(__C.IMG_SCALE, order)).byte().cuda()

            mask = torch.logical_or(x_mask, mask)
            mask_list.append(mask)
    return mask_list


class GraphTransformer(nn.Module):
    def __init__(self, __C):
        super(GraphTransformer, self).__init__()
        self.__C = __C
        self.tau = __C.TAU_MAX
        self.training = True
        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.GRAPH_LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.GRAPH_LAYER*1)])

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, frcn_feat, captions, lang_feat, frcn_feat_mask, captions_mask, lang_feat_mask):

        x_masks = getMasks(frcn_feat_mask, self.__C)

        for enc in self.enc_list:
            frcn_feat, frcn_feat_intra, F_cr = enc(frcn_feat, captions, x_masks, captions_mask, self.tau, self.training)

        for dec in self.dec_list:
            frcn_feat, lang_feat, F_qr = dec(frcn_feat, frcn_feat_intra, lang_feat, x_masks, frcn_feat_mask, lang_feat_mask, self.tau, self.training)

        return frcn_feat, lang_feat, F_cr, F_qr

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.HIDDEN_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.HIDDEN_SIZE
        )
        self.norm = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        att = self.mlp(x)

        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        x_atted = self.norm(x_atted)
        return x_atted


class SimScore(nn.Module):
    def __init__(self, __C):
        super(SimScore, self).__init__()
        self.__C = __C

    def forward(self, patch, lang_feat_global):
        B, Nt, E = patch.shape
        patch = patch / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)

        attn = torch.bmm(patch, lang_feat_global.transpose(-2, -1))
        attn = nn.Softmax(dim=1)(attn).squeeze(-1)

        return attn

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted, atted_map = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        atted = att_map
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value), atted


class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)
