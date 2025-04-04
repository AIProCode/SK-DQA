from openvqa.utils.make_mask import make_mask
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.COADQA.coadqa import MHAtt, SimScore, GraphTransformer, AttFlat

import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            bidirectional=False,
            batch_first=True
        )



        # Flatten to vector
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        # Classification layers
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)
        imgfeat_linear_size = __C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][1]
        self.frcn_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)
        self.cap_linear = nn.Linear(768, __C.HIDDEN_SIZE)
        self.patch_linear = nn.Linear(512, __C.HIDDEN_SIZE)
        self.linear = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.lang_linear = nn.Linear(__C.HIDDEN_SIZE*2, __C.HIDDEN_SIZE)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.backbone = GraphTransformer(__C)

    def zero_out_irrelevant_features(self, question_features, patch_description_features, patch_features, top_k=5):
        # 扩展问题特征的维度以匹配图块描述特征的维度
        question_features = question_features.expand(-1, 16, -1)
        # 计算余弦相似度
        similarities = F.cosine_similarity(question_features, patch_description_features, dim=-1)
        # 获取每个批次中相似度最高的 top_k 个图块的索引
        _, top_k_indices = torch.topk(similarities, k=top_k, dim=-1)

        # 创建一个全零的掩码，形状为 [B, 16]
        batch_size = question_features.size(0)
        mask = torch.zeros(batch_size, 16, dtype=torch.bool).cuda()
        # 根据 top_k 索引将掩码中对应的位置置为 True
        batch_indices = torch.arange(batch_size).unsqueeze(-1)
        mask[batch_indices, top_k_indices] = True

        # 扩展掩码的维度以匹配图块特征和图块描述特征的维度
        mask = mask.unsqueeze(-1).expand(-1, -1, patch_features.size(-1))
        # 根据掩码将不相关的图块描述特征和图块特征置为零
        processed_patch_description_features = patch_description_features * mask.float()
        processed_patch_features = patch_features * mask.float()
        return processed_patch_description_features, processed_patch_features

    def forward(self, frcn_feat, patch, patch_img, patch_cap, ques_ix):

        B, patch_num, _ = patch.size()

        patch = self.patch_linear(patch)
        frcn_feat = self.frcn_linear(frcn_feat)
        patch_cap = self.cap_linear(patch_cap)
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))  # [B,14]--[B,1,1,14]

        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)  # [B,T,512]
        lang_feat_global = lang_feat.mean(dim=1).unsqueeze(1)
        # lang_feat_global = self.avgpool(lang_feat.permute(0, 2, 1)).permute(0, 2, 1)  # [B,1,512]

        frcn_feat_mask = make_mask(frcn_feat)

        """计算问题和哪些图块最相关"""

        processed_patch_description_features, processed_patch_features = self.zero_out_irrelevant_features(
            lang_feat_global, patch_cap, patch, top_k=5
        )
        patch_cap = processed_patch_description_features
        # patch = processed_patch_features
        captions_mask = make_mask(patch_cap)

        """Graph Transformer"""
        frcn_feat, lang_feat, F_cr, F_qr = self.backbone(frcn_feat, patch_cap, lang_feat, frcn_feat_mask, captions_mask, lang_feat_mask)



        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            frcn_feat,
            frcn_feat_mask
        )

        proj_feat = lang_feat + img_feat

        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat, F_cr, F_qr

