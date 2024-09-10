#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn
import math
from .cross_vit import *
import os

IDENT_FUNCT_DICT = {
                    'phakirms': lambda x,y: 'Video_{:02d}/frame_{:06d}.png'.format(x,y),
                    }

class MultiSequenceTransformerBasicHead(nn.Module):
    """
    Frame Classification Head of TAPIS.
    """

    def __init__(
        self,
        cfg,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        cls_embed=False,
        recognition=False,
        sm_patch_size = 12,
        sm_enc_depth = 1,
        sm_enc_heads = 8,
        sm_enc_mlp_dim = 2048,
        sm_enc_dim_head = 64,
        lg_patch_size = 16,
        lg_enc_depth = 4,
        lg_enc_heads = 8,
        lg_enc_mlp_dim = 2048,
        lg_enc_dim_head = 64,
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(MultiSequenceTransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        self.dataset_name = cfg.TEST.DATASET
        self.parallel = cfg.NUM_GPUS > 1
        self.cls_embed = cls_embed
        self.recognition = recognition
        self.act_func = act_func
        self.multiscale_encoder = nn.ModuleList([])
        self.full_sequence_self_attention = nn.ModuleList([])
        self.num_classes = num_classes

        self.mvit_feats_enable = cfg.MVIT_FEATS.ENABLE
        self.mvit_feats_path = cfg.MVIT_FEATS.PATH

        self.full_self_attention = cfg.MVIT.FULL_SELF_ATTENTION
        self.full_self_attention_type = cfg.MVIT.FULL_SELF_ATTENTION_TYPE

        self.num_sequences = len(cfg.DATA.MULTI_SAMPLING_RATE)
        self.logit_join_type = cfg.MVIT.LOGIT_JOIN_TYPE

        self.cross_attention = cfg.MVIT.CROSS_ATTENTION 

        if self.cross_attention:
            for _ in range(self.num_sequences):
                self.multiscale_encoder.append(MultiScaleEncoder(depth=1,
                                                        sm_dim=768,
                                                        lg_dim=768,
                                                        cross_attn_depth = 2,
                                                        cross_attn_heads = 2,
                                                        cross_attn_dim_head = 64,
                                                        dropout=0.1
                                                        ))
            
            if self.full_self_attention:
                for _ in range(self.num_sequences):
                    self.self_attn_layers = nn.ModuleList([])
                    for i in range(2):
                        self.self_attn_layers.append(torch.nn.MultiheadAttention(embed_dim=768, 
                                                                                    num_heads=4, 
                                                                                    batch_first=True, 
                                                                                    dropout=0.1))
                    self.full_sequence_self_attention.append(self.self_attn_layers)
                    

        self.mlp_heads = nn.ModuleList([])

        if self.logit_join_type in ["sum", "ada"]:
            for _ in range(self.num_sequences):
                self.mlp_heads.append(nn.Sequential(nn.LayerNorm(768), nn.Linear(768, num_classes)))

        '''
        if self.logit_join_type in ["ada"]:
            self.pool = AdaPool1d(kernel_size=(self.num_sequences), beta=(1)).cuda()
            self.make_contiguous = Contiguous()
        '''

        if self.logit_join_type in ["mlp"]:
            input_size = 768 * self.num_sequences
            self.mlp_logits_embedding = nn.Sequential(
                                            nn.Dropout(p=0.1),
                                            nn.Linear(input_size, input_size, bias=True),
                                        )
            
            self.mlp_classifier = nn.Sequential(
                                            nn.Tanh(),
                                            nn.Dropout(p=0.1),
                                            nn.Linear(input_size, num_classes, bias=True)
                                        )

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def save_cls_tokens(self, x, image_names):
        #json_path = "/home/aperezr20/endovis/GraSP/TAPIS/association_30fps.json"
        #json_data = self.upload_json_file(json_path)
        json_data = {}

        if self.parallel:
            image_names = [IDENT_FUNCT_DICT[self.dataset_name](*name) for name in image_names]
        
        for idx, frame_name in enumerate(image_names):
            if frame_name in json_data:
                name = json_data[frame_name]
            else:
                name = frame_name
            
            mvit_feats_dictionary = []
            video_name = name.split('/')[0]
            #print(video_name)

            if not os.path.exists(os.path.join(self.mvit_feats_path, video_name)):
                os.makedirs(os.path.join(self.mvit_feats_path, video_name))

            sequence_embeddings = x[idx].data.cpu().numpy()
            mvit_feats_dictionary.extend([sequence_embeddings])
            feat_name = name.split('.')[0] + '.pth'
            path = os.path.join(self.mvit_feats_path, feat_name)
            torch.save(mvit_feats_dictionary, path)

    def upload_json_file(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    

    def forward(self, x, features=None, boxes_mask=None, image_names=None):
        b_size, seq_dim, embed_dim = x[0].shape
        all_sequences = x

        sequences = []

        for idx in range(self.num_sequences):
            main_seq = all_sequences[idx]
            other_seqs = all_sequences[:idx] + all_sequences[idx+1:]
            sequences.append((main_seq, tuple(other_seqs)))

        #logits_debug = torch.zeros((b_size, self.num_classes)).cuda()
        logits = []
        tokens = torch.zeros((b_size, len(x), embed_dim)).cuda()
        
        for idx, (seq_tokens, context) in enumerate(sequences):
            if self.cross_attention:
                encoded_seq = self.multiscale_encoder[idx](seq_tokens, context)

                if self.full_self_attention and self.full_self_attention_type == "cross_output":
                    context = torch.cat(context, dim=1)
                    encoded_seq = torch.cat((encoded_seq, seq_tokens), dim=1)
                    for i in range(2):
                        encoded_seq = self.full_sequence_self_attention[idx][i](encoded_seq, encoded_seq, encoded_seq)[0]
                
                cls_token = encoded_seq[:, 0]

                tokens[:, idx, :] = cls_token
            
            else:
                cls_token = seq_tokens[:, 0]

                tokens[:, idx, :] = cls_token

            if self.logit_join_type in ["sum", "ada"]:
                logits.append(self.mlp_heads[idx](cls_token))

            elif self.logit_join_type in ["mlp"]:
                logits.append(cls_token)
        
        logits = torch.stack(logits).cuda()

        if self.logit_join_type == "sum":
            logits = torch.sum(logits, dim=0)
        
        if self.logit_join_type == "mlp":
            logits = logits.permute(1, 0, 2)
            logits = logits.reshape(logits.shape[0], -1)
            embeddings = self.mlp_logits_embedding(logits)
            logits = self.mlp_classifier(embeddings)
        
        if self.act_func == "sigmoid" or not self.training:
            x = self.act(logits)

        if image_names is not None:
            self.save_cls_tokens(embeddings, image_names)

        return None

class ClassificationBasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        cfg,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ClassificationBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)
        self.act_func = act_func

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            if cfg.TEMPORAL_MODULE.CHUNKS:
                self.act = nn.Softmax(dim=2)
            else:
                self.act = nn.Softmax(dim=1)
                
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if self.act_func == "sigmoid" or not self.training:
            x = self.act(x)
        return x