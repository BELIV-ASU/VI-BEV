import os
import numpy as np
import random
import math
import json
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# from torch.cuda.amp import autocast
from math import gcd
from functools import reduce
import time
import matplotlib.pyplot as plt

# torch.autograd.set_detect_anomaly(True)
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    reshape_q = q.view(q.size()[0]*q.size()[1], q.size()[2], q.size()[3])
    reshape_k = k.view(k.size()[0]*k.size()[1], k.size()[2], k.size()[3])
    sub_q = F.adaptive_avg_pool2d(reshape_q,(16,q.size()[3]))
    sub_k = F.adaptive_avg_pool2d(reshape_k,(16,k.size()[3]))
    recons_q = sub_q.view(q.size()[0], q.size()[1], 16, -1)
    recons_k = sub_k.view(k.size()[0], k.size()[1], 16, -1)
    #attn_logits = torch.matmul(q, k.transpose(-2, -1))

    attn_logits_1 = torch.matmul(q, recons_k.transpose(-2, -1))
    attn_logits_1 = attn_logits_1 / math.sqrt(d_k)
    attention_1 = F.softmax(attn_logits_1, dim=-1)
    
    attn_logits_2 = torch.matmul(recons_q, k.transpose(-2, -1))
    attn_logits_2 = attn_logits_2 / math.sqrt(d_k)
    attention_2 = F.softmax(attn_logits_2, dim=-1)
    
    #attn_logits = attn_logits / math.sqrt(d_k)
    #attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True)[0]
    #attention = F.softmax(attn_logits, dim=-1)
    attention = torch.matmul(attention_1,attention_2)
    values = torch.matmul(attention, v)
    return values, attention

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x_r = x.view(1,-1,x.shape[-2],x.shape[-1])
        x_r = self.proj(x_r)  # [B, embed_dim, H/P, W/P]
        x_r = x_r.flatten(2)  # [B, embed_dim, (H/P)*(W/P)]
        x_r = x_r.transpose(1, 2)  # [B, (H/P)*(W/P), embed_dim]
        return x_r

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads, cross_label, bev_cross_label):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.cross_label = cross_label
        self.bev_cross_label = bev_cross_label
        self.input_dim = input_dim

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim,dtype=torch.float32)
        self.o_proj = nn.Linear(embed_dim, embed_dim,dtype=torch.float32)

        self.patch_in = PatchEmbedding(in_channels=3, patch_size=16, embed_dim=embed_dim)
        self.patch_pe = PatchEmbedding(in_channels=256, patch_size=16, embed_dim=embed_dim)

        self.q_list = []
        self.k_list = []
        self.v_list = []
        self.ini_trigger = False

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.weight.data = self.qkv_proj.weight.data.to(torch.float32)
        self.qkv_proj.bias.data = torch.zeros_like(self.qkv_proj.bias.data, dtype=torch.float32)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.zero_()
            
    def forward(self, x, feat_tag=0, mask=None, return_attention=False, pe=None, mask_map=None):
        q_list = []
        k_list = []
        v_list = []
        
        extend_map = mask_map.unsqueeze(0).unsqueeze(0)
        conv_layer = nn.AvgPool2d(kernel_size=(16,16), stride=(16,16), padding=(0,0))
        patch_map = conv_layer(extend_map)
        patch_map = patch_map.squeeze(0).squeeze(0)
        patch_mask = patch_map > 0
        
        if feat_tag == 0:  # initial the concated K,V
            for i in range(len(x)):
                patch_x = self.patch_in(x[i])
                patch_x = patch_x.view(1,16,44,-1)
                flatten_x = patch_x[:,patch_mask,:].view(1,-1, self.input_dim)

                batch_size, seq_length, _ = flatten_x.size()
            
                qkv = self.qkv_proj(flatten_x.float())
                qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
                qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
                q, k, v = qkv.chunk(3, dim=-1)
                q_list.append(q)
                k_list.append(k)
                v_list.append(v)
            self.q_list, self.k_list, self.v_list = q_list, k_list, v_list
        else:
            q_list, k_list, v_list = self.q_list, self.k_list, self.v_list

        patch_x = self.patch_in(x[feat_tag])
        patch_x = patch_x.view(1,16,44,-1)
        
        flatten_x = patch_x[:,patch_mask,:].reshape(1,-1, self.input_dim)
        batch_size, seq_length, _ = flatten_x.size()
           
        patch_pe = self.patch_pe(pe[feat_tag])
        patch_pe = patch_pe.view(1,16,44,-1)

        flatten_pe = patch_pe[:,patch_mask,:].reshape(batch_size,seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        qkv = self.qkv_proj(flatten_x.float())
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
            
        q, k, v = qkv.chunk(3, dim=-1)

        q_list[feat_tag] = q
            

        batch_size, seq_length, _ = flatten_x.size()
        k_concat = torch.cat(k_list[:feat_tag] + k_list[feat_tag+1:],dim=-2)
        v_concat = torch.cat(v_list[:feat_tag] + v_list[feat_tag+1:],dim=-2)

        # flatten_pe = pe[feat_tag].reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values, attention = scaled_dot_product(q_list[feat_tag]+flatten_pe, k_concat, v_concat)

        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
       
        values = values.reshape(batch_size, seq_length, self.embed_dim)

        o = self.o_proj(values)
        if return_attention:
            return o, attention
        else:
            return o


class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, cross_label=False, bev_cross_label=False, dropout=0.3):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads, cross_label, bev_cross_label)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.ReLU(inplace=False),
            nn.Linear(dim_feedforward,dim_feedforward),
            nn.ReLU(inplace=False),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.patch = PatchEmbedding(in_channels=3, patch_size=16, embed_dim=input_dim)

        self.input_dim = input_dim

    def forward(self, x, feat_tag=0, mask=None, pe=None, mask_map=None):
        # Attention part

        attn_out = self.self_attn(x, feat_tag=feat_tag, mask=mask, pe=pe, mask_map=mask_map)


        #print('The scale-dot process take', end_time - start_time)

        # flatten features, the last number equals to input_dim
        with torch.no_grad():
            patch_x = self.patch(x[feat_tag])
        patch_x = patch_x.view(1,16,44,-1)
        
        if mask_map != None:
            extend_map = mask_map.unsqueeze(0).unsqueeze(0)
            conv_layer = nn.AvgPool2d(kernel_size=(16,16), stride=(16,16), padding=(0,0))
            patch_map = conv_layer(extend_map)
            patch_map = patch_map.squeeze(0).squeeze(0)
            patch_mask = patch_map > 0
            flatten_x_tag = patch_x[:,patch_mask,:].reshape(1,-1, self.input_dim)
            #flatten_x_tag = torch.masked_select(x[feat_tag], mask_map.cuda().bool()).reshape(1, -1, self.input_dim)
        else:
            flatten_x_tag = patch_x.reshape(1,-1, self.input_dim)
            #flatten_x_tag = x[feat_tag].view(1,-1,self.input_dim)

      
        flatten_x_tag = flatten_x_tag + attn_out
        flatten_x_tag = self.norm1(flatten_x_tag)

        # MLP part

        linear_out = self.linear_net(flatten_x_tag)

        flatten_x_tag = flatten_x_tag + self.dropout(linear_out)
        flatten_x_tag = self.norm2(flatten_x_tag)

        
        # convert flatten features back, img [3,256,704]
       
        if mask_map != None:
            mask = mask_map == 1
            x_zeros = torch.zeros(3,x[feat_tag].shape[-2],x[feat_tag].shape[-1]).cuda()
            
            #masked_x_tag = x[feat_tag][:, mask]  # Perform masking once
            flatten_x_tag_r = flatten_x_tag.view(1,-1,flatten_x_tag.shape[-2],flatten_x_tag.shape[-1])
            resized_flatten_x_tag = F.interpolate(flatten_x_tag_r, size=(256,704), mode='bilinear',align_corners=False)
            image = resized_flatten_x_tag.expand(1,3,256,704).squeeze()
            x_zeros[:,mask] = image[:,mask] + x[feat_tag][:,mask]
            
            #fill_x = flatten_x_tag.reshape(-1, masked_x_tag.shape[-2], masked_x_tag.shape[-1])
            #x_zeros[:,mask]=fill_x
            x = x_zeros
        else:
            x = flatten_x_tag.view(-1,x[feat_tag].shape[-2],x[feat_tag].shape[-1])
        # x = self.norm(x)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])
    def forward(self, x, feat_tag=0, mask_map = None,  mask=None, pe=None):
        x_clone = [tensor.clone() for tensor in x]       
        for l in self.layers:
            # x[feat_tag] = l(x, feat_tag=feat_tag, mask=mask, pe=pe, mask_map=mask_map)
            x_clone[feat_tag] = l(x_clone, feat_tag=feat_tag, mask=mask, pe=pe, mask_map=mask_map)

        return x_clone[feat_tag].view(1,-1,x[feat_tag].shape[-2],x[feat_tag].shape[-1])


