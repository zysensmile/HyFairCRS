# -*- encoding: utf-8 -*-
# @Time    :   2021/5/26
# @Author  :   Chenzhan Shang
# @email   :   czshang@outlook.com

r"""
PCR
====
References:
    Chen, Qibin, et al. `"Towards Knowledge-Based Recommender Dialog System."`_ in EMNLP 2019.

.. _`"Towards Knowledge-Based Recommender Dialog System."`:
   https://www.aclweb.org/anthology/D19-1189/

"""

import json
import os.path
import random
from typing import List

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from tqdm import tqdm
from torch_geometric.nn import RGCNConv, HypergraphConv, GCNConv

import numpy as np
import torch
import numba
from numba import njit, prange

from crslab.config import DATASET_PATH
from crslab.model.base import BaseModel
from crslab.model.crs.mhim.attention import MHItemAttention
from crslab.model.utils.functions import edge_to_pyg_format
from crslab.model.utils.modules.attention import SelfAttentionBatch, SelfAttentionSeq
from crslab.model.utils.modules.transformer import TransformerEncoder
from crslab.model.crs.mhim.decoder import TransformerDecoderKG

numba.set_num_threads(128)

class MHIMModel(BaseModel):
    """

    Attributes:
        vocab_size: A integer indicating the vocabulary size.
        pad_token_idx: A integer indicating the id of padding token.
        start_token_idx: A integer indicating the id of start token.
        end_token_idx: A integer indicating the id of end token.
        token_emb_dim: A integer indicating the dimension of token embedding layer.
        pretrain_embedding: A string indicating the path of pretrained embedding.
        n_entity: A integer indicating the number of entities.
        n_relation: A integer indicating the number of relation in KG.
        num_bases: A integer indicating the number of bases.
        kg_emb_dim: A integer indicating the dimension of kg embedding.
        user_emb_dim: A integer indicating the dimension of user embedding.
        n_heads: A integer indicating the number of heads.
        n_layers: A integer indicating the number of layer.
        ffn_size: A integer indicating the size of ffn hidden.
        dropout: A float indicating the dropout rate.
        attention_dropout: A integer indicating the dropout rate of attention layer.
        relu_dropout: A integer indicating the dropout rate of relu layer.
        learn_positional_embeddings: A boolean indicating if we learn the positional embedding.
        embeddings_scale: A boolean indicating if we use the embeddings scale.
        reduction: A boolean indicating if we use the reduction.
        n_positions: A integer indicating the number of position.
        longest_label: A integer indicating the longest length for response generation.
        user_proj_dim: A integer indicating dim to project for user embedding.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        self.device  = device
        self.gpu     = opt.get("gpu", -1)
        self.dataset = opt.get("dataset", None)
        assert self.dataset in ['HReDial', 'HTGReDial']
        # vocab
        self.pad_token_idx   = vocab['tok2ind']['__pad__']
        self.start_token_idx = vocab['tok2ind']['__start__']
        self.end_token_idx   = vocab['tok2ind']['__end__']
        self.vocab_size      = vocab['vocab_size']
        self.token_emb_dim   = opt.get('token_emb_dim', 300)
        self.pretrain_embedding = side_data.get('embedding', None)
        # kg
        self.n_entity   = vocab['n_entity']
        self.entity_kg  = side_data['entity_kg']
        self.n_relation = self.entity_kg['n_relation']
        self.edge_idx, self.edge_type = edge_to_pyg_format(self.entity_kg['edge'], 'RGCN')
        self.edge_idx     = self.edge_idx.to(device)
        self.edge_type    = self.edge_type.to(device)
        self.num_bases    = opt.get('num_bases', 8)
        self.kg_emb_dim   = opt.get('kg_emb_dim', 300)
        self.user_emb_dim = self.kg_emb_dim
        # transformer
        self.n_heads  = opt.get('n_heads', 2)
        self.n_layers = opt.get('n_layers', 2)
        self.ffn_size = opt.get('ffn_size', 300)
        self.dropout  = opt.get('dropout', 0.1)
        self.attention_dropout = opt.get('attention_dropout', 0.0)
        self.relu_dropout      = opt.get('relu_dropout', 0.1)
        self.embeddings_scale  = opt.get('embedding_scale', True)
        self.learn_positional_embeddings = opt.get('learn_positional_embeddings', False)
        self.reduction     = opt.get('reduction', False)
        self.n_positions   = opt.get('n_positions', 1024)
        self.longest_label = opt.get('longest_label', 30)
        self.user_proj_dim = opt.get('user_proj_dim', 512)
        self.n_linear_layers = opt.get('n_linear_layers', 1)
        self.n_hyper_layers  = opt.get('n_hyper_layers', 1)
        # pooling
        self.pooling = opt.get('pooling', None)
        assert self.pooling == 'Attn' or self.pooling == 'Mean'
        # MHA
        self.mha_n_heads    = opt.get('mha_n_heads', 4)
        self.extension_strategy = opt.get('extension_strategy', None)
        self.pretrain       = opt.get('pretrain', False)
        self.pretrain_data  = None
        self.pretrain_epoch = opt.get('pretrain_epoch', 9999)

        super(MHIMModel, self).__init__(opt, device)
        return

    # 构建模型
    def build_model(self, *args, **kwargs):
        if self.pretrain:
            pretrain_file = os.path.join('pretrain', self.dataset, str(self.pretrain_epoch) + '-epoch.pth')
            self.pretrain_data = torch.load(pretrain_file, map_location=torch.device('cuda:' + str(self.gpu[0])))
            logger.info(f"[Load Pretrain Weights from {pretrain_file}]")
        self._build_copy_mask()
        self._build_adjacent_matrix()
        self._build_embedding()
        self._build_kg_layer()
        self._build_recommendation_layer()
        self._build_conversation_layer()

    # 构建 mask
    def _build_copy_mask(self):
        """
        生成掩码，用于指示哪些 token 可以被“复制”到生成的序列中，而不是通过模型生成
        """
        if self.dataset == 'HReDial':
            token_filename = os.path.join(DATASET_PATH, "hredial", "nltk", "token2id.json")
        else:
            token_filename = os.path.join(DATASET_PATH, "htgredial", "pkuseg", "token2id.json")
        token_file = open(token_filename, 'r', encoding="utf-8")
        token2id   = json.load(token_file)
        id2token   = {token2id[token]: token for token in token2id}
        self.copy_mask = list()
        for i in range(len(id2token)):
            token = id2token[i]
            if token[0] == '@':
                self.copy_mask.append(True)
            else:
                self.copy_mask.append(False)
        self.copy_mask = torch.as_tensor(self.copy_mask).to(self.device)
        return

    # 构建关联矩阵
    def _build_adjacent_matrix(self):
        # 构建图结构
        graph = dict()
        for head, tail, relation in tqdm(self.entity_kg['edge']):
            # 填充图结构
            # graph[head] <- tail
            graph[head] = graph.get(head, []) + [tail]
        # 构建邻接矩阵
        adj   = dict()
        for entity in tqdm(range(self.n_entity)):
            adj[entity] = set()
            if entity not in graph:
                continue
            last_hop = {entity}
            for _ in range(1):
                buffer = set()
                for source in last_hop:
                    adj[entity].update(graph[source])
                    buffer.update(graph[source])
                last_hop = buffer
        self.adj = adj
        logger.info(f"[Build adjacent matrix]")
        return

    # 构建编码层
    def _build_embedding(self):
        # 使用预训练词向量
        if self.pretrain_embedding is not None:
            self.token_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(self.pretrain_embedding, dtype=torch.float), freeze=False,
                padding_idx=self.pad_token_idx)
        # 不预训练
        else:
            self.token_embedding = nn.Embedding(self.vocab_size, self.token_emb_dim, self.pad_token_idx)
            nn.init.normal_(self.token_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
            nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0)

        # 知识图谱的 embedding
        self.kg_embedding = nn.Embedding(self.n_entity, self.kg_emb_dim, 0)
        nn.init.normal_(self.kg_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        nn.init.constant_(self.kg_embedding.weight[0], 0)

        # 上下文的 embedding
        self.con_embedding = nn.Embedding(self.n_entity, self.kg_emb_dim, 0)
        nn.init.normal_(self.con_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        nn.init.constant_(self.con_embedding.weight[0], 0)
        logger.debug('[Build embedding]')
        return

    # 构建编码层、超图卷积层
    def _build_kg_layer(self):
        # 编码层
        self.kg_encoder  = RGCNConv(self.kg_emb_dim, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
        self.con_encoder = RGCNConv(self.kg_emb_dim, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
        if self.pretrain:
            self.kg_encoder.load_state_dict(self.pretrain_data['encoder'])
        # 线图卷积层
        self.linear_conv_item   = []
        self.linear_conv_entity = []
        self.linear_conv_word   = []
        self.linear_conv_review = []
        for _ in range(self.n_linear_layers):
            self.linear_conv_item.append(GCNConv(self.kg_emb_dim, self.kg_emb_dim))
            self.linear_conv_entity.append(GCNConv(self.kg_emb_dim, self.kg_emb_dim))
            self.linear_conv_word.append(GCNConv(self.kg_emb_dim, self.kg_emb_dim))
            self.linear_conv_review.append(GCNConv(self.kg_emb_dim, self.kg_emb_dim))
        # 超图卷积层
        self.hyper_conv_session   = []
        self.hyper_conv_knowledge = []
        self.hyper_conv_word      = []
        self.hyper_conv_review    = []
        for _ in range(self.n_hyper_layers):
            self.hyper_conv_session.append(HypergraphConv(self.kg_emb_dim, self.kg_emb_dim))
            self.hyper_conv_knowledge.append(HypergraphConv(self.kg_emb_dim, self.kg_emb_dim))
            self.hyper_conv_word.append(HypergraphConv(self.kg_emb_dim, self.kg_emb_dim))
            self.hyper_conv_review.append(HypergraphConv(self.kg_emb_dim, self.kg_emb_dim))
        self.concept_edge_sets = self.concept_edge_list4GCN()
        # 注意力层
        self.item_attn = MHItemAttention(self.kg_emb_dim, self.mha_n_heads)
        # 池化层
        if self.pooling == 'Attn':
            self.kg_attn = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
            self.kg_attn_his = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)

        logger.debug('[Build kg layer]')
        return

    # # 构建编码层、超图卷积层
    # def _build_kg_layer(self):
    #     # 编码层
    #     self.kg_encoder = RGCNConv(self.kg_emb_dim, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
    #     self.con_encoder = RGCNConv(self.kg_emb_dim, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
    #     if self.pretrain:
    #         self.kg_encoder.load_state_dict(self.pretrain_data['encoder'])
    #     # 线图卷积层
    #     self.linear_conv_item = GCNConv(self.kg_emb_dim, self.kg_emb_dim)
    #     self.linear_conv_entity = GCNConv(self.kg_emb_dim, self.kg_emb_dim)
    #     self.linear_conv_word = GCNConv(self.kg_emb_dim, self.kg_emb_dim)
    #     self.linear_conv_review = GCNConv(self.kg_emb_dim, self.kg_emb_dim)
    #     # 超图卷积层
    #     self.hyper_conv_session = HypergraphConv(self.kg_emb_dim, self.kg_emb_dim)
    #     self.hyper_conv_knowledge = HypergraphConv(self.kg_emb_dim, self.kg_emb_dim)
    #     self.hyper_conv_conceptnet = HypergraphConv(self.kg_emb_dim, self.kg_emb_dim)
    #     self.hyper_conv_review = HypergraphConv(self.kg_emb_dim, self.kg_emb_dim)
    #     self.concept_edge_sets = self.concept_edge_list4GCN()
    #     # print(self.concept_edge_sets.shape)
    #     # 注意力层
    #     self.item_attn = MHItemAttention(self.kg_emb_dim, self.mha_n_heads)
    #     # 池化层
    #     if self.pooling == 'Attn':
    #         self.kg_attn = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
    #         self.kg_attn_his = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
    #     logger.debug('[Build kg layer]')
    #     return

    # 构建推荐模块
    def _build_recommendation_layer(self):
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.rec_loss = nn.CrossEntropyLoss()
        # self.review_encoder = TransformerEncoder(
        #     self.n_heads,
        #     self.n_layers,
        #     self.token_emb_dim,
        #     self.ffn_size,
        #     self.vocab_size,
        #     self.token_embedding,
        #     self.dropout,
        #     self.attention_dropout,
        #     self.relu_dropout,
        #     self.pad_token_idx,
        #     self.learn_positional_embeddings,
        #     self.embeddings_scale,
        #     self.reduction,
        #     self.n_positions
        # )
        # self.review_decoder = nn.Sequential(
        #     SelfAttentionSeq(
        #         self.token_emb_dim,
        #         self.kg_emb_dim,
        #     ),
        #     nn.Linear(self.token_emb_dim, self.kg_emb_dim),
        #     nn.ReLU(),
        # )
        logger.debug('[Build recommendation layer]')
        return

    # 构建对话模块
    def _build_conversation_layer(self):
        self.register_buffer('START', torch.tensor([self.start_token_idx], dtype=torch.long))
        self.entity_to_token = nn.Linear(self.kg_emb_dim, self.token_emb_dim, bias=True)
        self.related_encoder = TransformerEncoder(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.pad_token_idx,
            self.learn_positional_embeddings,
            self.embeddings_scale,
            self.reduction,
            self.n_positions
        )
        self.context_encoder = TransformerEncoder(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.pad_token_idx,
            self.learn_positional_embeddings,
            self.embeddings_scale,
            self.reduction,
            self.n_positions
        )
        self.decoder = TransformerDecoderKG(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.embeddings_scale,
            self.learn_positional_embeddings,
            self.pad_token_idx,
            self.n_positions
        )
        # 处理用户 embedding
        self.user_proj_1 = nn.Linear(self.user_emb_dim, self.user_proj_dim)
        self.user_proj_2 = nn.Linear(self.user_proj_dim, self.vocab_size)
        self.conv_loss   = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)

        # 处理复制部分 embedding
        self.copy_proj_1 = nn.Linear(2 * self.token_emb_dim, self.token_emb_dim)
        self.copy_proj_2 = nn.Linear(self.token_emb_dim, self.vocab_size)
        logger.debug('[Build conversation layer]')
        return

    # 获取 Session 超图
    def _get_session_hypergraph(self, session_related_entities):
        hypergraph_nodes, hypergraph_edges, hyper_edge_counter = list(), list(), 0
        for related_entities in session_related_entities:
            if len(related_entities) == 0:
                continue
            hypergraph_nodes += related_entities
            hypergraph_edges += [hyper_edge_counter] * len(related_entities)
            hyper_edge_counter += 1
        # 创建超边索引
        hyper_edge_index = torch.tensor([hypergraph_nodes, hypergraph_edges], device=self.device)
        # 返回前进行去重
        return list(set(hypergraph_nodes)), hyper_edge_index

    # 获取 Review 超图
    def _get_review_hypergraph(self, review_entities):
        hypergraph_nodes, hypergraph_edges, hyper_edge_counter = list(), list(), 0
        for related_entities in review_entities:
            if len(related_entities) == 0:
                continue
            hypergraph_nodes += related_entities
            hypergraph_edges += [hyper_edge_counter] * len(related_entities)
            hyper_edge_counter += 1
        hyper_edge_index = torch.tensor([hypergraph_nodes, hypergraph_edges], device=self.device)
        return list(set(hypergraph_nodes)), hyper_edge_index

    # 获取 Knowledge 超图
    def _get_knowledge_hypergraph(self, session_related_items):
        # 去重
        related_items_set = set()
        for related_items in session_related_items:
            related_items_set.update(related_items)
        session_related_items = list(related_items_set)

        hypergraph_nodes, hypergraph_edges, hyper_edge_counter = list(), list(), 0
        for item in session_related_items:
            hypergraph_nodes.append(item)
            hypergraph_edges.append(hyper_edge_counter)
            neighbors = list(self.adj[item])
            hypergraph_nodes += neighbors
            hypergraph_edges += [hyper_edge_counter] * len(neighbors)
            hyper_edge_counter += 1
        hyper_edge_index = torch.tensor([hypergraph_nodes, hypergraph_edges], device=self.device)
        return list(set(hypergraph_nodes)), hyper_edge_index

    # 得到知识图谱中每个节点的 embedding
    def _get_knowledge_embedding(self, hypergraph_items, raw_knowledge_embedding, knowledge_tot_sub):
        knowledge_embedding_list = []
        for item in hypergraph_items:
            sub_graph = [item] + list(self.adj[item])   # 构建子图
            sub_graph = [knowledge_tot_sub[item] for item in sub_graph]    # 获取子图的节点索引
            sub_graph_embedding = raw_knowledge_embedding[sub_graph]    # 得到子图 embedding
            sub_graph_embedding = torch.mean(sub_graph_embedding, dim=0)    # 取平均得到节点的最终 embedding
            knowledge_embedding_list.append(sub_graph_embedding)
        return torch.stack(knowledge_embedding_list, dim=0)    # [num_items, embedding_dim]

    # 构建边列表，用于 GCN 的输入
    def concept_edge_list4GCN(self):
        # 加载数据
        node2index = json.load(open(R'data/dataset/hredial/nltk/key2index_3rd.json',encoding='utf-8'))
        f          = open(R'data/dataset/hredial/nltk/conceptnet_edges2nd.txt',encoding='utf-8')
        edges      = set()
        stopwords  = set([word.strip() for word in open(R'data/dataset/hredial/nltk/stopwords.txt',encoding='utf-8')])
        # 处理边数据
        for line in f:
            lines  = line.strip().split('\t')
            # 提取两个端点
            entity0= node2index[lines[1].split('/')[0]]
            entity1= node2index[lines[2].split('/')[0]]
            if lines[1].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords:
                # 忽略包含停用词的边
                continue
            # 添加双向边
            edges.add((entity0,entity1))
            edges.add((entity1,entity0))
        # 构建边列表
        edge_set = [[co[0] for co in list(edges)],[co[1] for co in list(edges)]]
        concept_edge_list = torch.LongTensor(edge_set).cuda()
        # print("concept_edge_list shape", concept_edge_list.shape)
        return concept_edge_list

    # 获取 ConceptNet 超图
    def _get_conceptnet_hypergraph(self, session_related_items):
        # print("session_related_items", session_related_items)
        # 去重
        related_items_set = set()
        for related_items in session_related_items:
            related_items_set.update(related_items)
        session_related_items = list(related_items_set)

        hypergraph_nodes, hypergraph_edges, hyper_edge_counter = list(), list(), 0
        for item in session_related_items:
            hypergraph_nodes.append(item)
            hypergraph_edges.append(hyper_edge_counter)
            neighbors = list(self.adj[item])
            hypergraph_nodes += neighbors
            hypergraph_edges += [hyper_edge_counter] * len(neighbors)
            hyper_edge_counter += 1
        hyper_edge_index = torch.tensor([hypergraph_nodes, hypergraph_edges], device=self.device)
        if hyper_edge_index.shape[1] == 0:
            hyper_edge_index = torch.tensor([[0], [0]], device=self.device)
        # print("hyper_edge_index", hyper_edge_index)
        return list(set(hypergraph_nodes)), hyper_edge_index

    # 得到 ConceptNet 中每个节点的 embedding
    def _get_conceptnet_embedding(self, hypergraph_items, raw_conceptnet_embedding, conceptnet_tot2sub):
        conceptnet_embedding_list = []
        for item in hypergraph_items:
            sub_graph = [item] + list(self.adj[item])
            sub_graph = [conceptnet_tot2sub[item] for item in sub_graph]
            sub_graph_embedding = raw_conceptnet_embedding[sub_graph]
            sub_graph_embedding = torch.mean(sub_graph_embedding, dim=0)
            conceptnet_embedding_list.append(sub_graph_embedding)
        return torch.stack(conceptnet_embedding_list, dim=0)

    @staticmethod
    def flatten(inputs):
        outputs = set()
        for li in inputs:
            for i in li:
                outputs.add(i)
        return list(outputs)

    # 把所有 embedding 使用注意力机制融合，并池化得到最终 user embedding
    def _attention_and_gating(self,
                              hg_session_embedding,
                              hg_knowledge_embedding,
                              hg_conceptnet_embedding,
                              context_embedding,
                              hg_review_embedding,
                              lg_session_embedding,
                              lg_knowledge_embedding,
                              lg_conceptnet_embedding,
                              lg_review_embedding
                              ):
        zero_embedding = torch.zeros_like(hg_session_embedding)
        related_embedding = torch.cat(
                      (hg_session_embedding,
                              hg_knowledge_embedding,
                              hg_conceptnet_embedding,
                              hg_review_embedding,
                              lg_session_embedding,
                              lg_knowledge_embedding,
                              lg_conceptnet_embedding,
                              lg_review_embedding),dim=0)
        if context_embedding is None:
            if self.pooling == 'Attn':
                user_repr = self.kg_attn_his(related_embedding)
            else:
                assert self.pooling == 'Mean'
                user_repr = torch.mean(related_embedding, dim=0)
        elif self.pooling == 'Attn':
            attentive_related_embedding = self.item_attn(related_embedding, context_embedding)
            user_repr = self.kg_attn_his(attentive_related_embedding)
            user_repr = torch.unsqueeze(user_repr, dim=0)
            user_repr = torch.cat((context_embedding, user_repr), dim=0)
            user_repr = self.kg_attn(user_repr)
        else:
            assert self.pooling == 'Mean'
            attentive_related_embedding = self.item_attn(related_embedding, context_embedding)
            user_repr = torch.mean(attentive_related_embedding, dim=0)
            user_repr = torch.unsqueeze(user_repr, dim=0)
            user_repr = torch.cat((context_embedding, user_repr), dim=0)
            user_repr = torch.mean(user_repr, dim=0)
        return user_repr

    # 把评论数据转换为模型输入
    def _get_review_token(self, _review):
        review_token = []
        # 嵌套列表转换回 tensor
        for review in _review:
            if len(review) == 0:
                review = review + [[0]]
            review = review[0]
            review_token.append(torch.tensor(review + [0]))
        # 进行填充，保证长度相同
        review_token = torch.nn.utils.rnn.pad_sequence(review_token, batch_first=True, padding_value=0)
        return review_token.long().cuda()

    # 编码用户
    def encode_user(self, batch_related_entities, batch_related_items, batch_context_entities, review, kg_embedding,
                    con_embedding):

        # def get_shape(nested_list):
        #     if isinstance(nested_list, list):
        #         return [len(nested_list)] + get_shape(nested_list[0])
        #     else:
        #         return []

        # print(f"Shape:\n"
        #       f"Batch related entities: {get_shape(batch_context_entities)}\n"  # [batch_size, entities]
        #       f"Batch related items:    {get_shape(batch_related_items)}\n"     # [batch_size, 39, 5]
        #       f"Batch context entities: {get_shape(batch_related_entities)}\n"  # [batch_size, 39, 8]
        #       f"Review:                 {review}")  # Dict, {review_tokens, review_entities}

        batch_size      = len(batch_related_entities)
        user_repr_list = []
        batch_ssl_loss = 0.0
        for session_related_entities, session_related_items, context_entities in zip(batch_related_entities,
                                                                                     batch_related_items,
                                                                                     batch_context_entities):
            sample_ssl_loss = 0.0
            flattened_session_related_items = self.flatten(session_related_items)

            # 冷启动情形
            if len(flattened_session_related_items) == 0:
                if len(context_entities) == 0:
                    user_repr = torch.zeros(self.user_emb_dim, device=self.device)
                elif self.pooling == 'Attn':
                    user_repr = kg_embedding[context_entities]
                    user_repr = self.kg_attn(user_repr)
                else:
                    assert self.pooling == 'Mean'
                    user_repr = kg_embedding[context_entities]
                    user_repr = torch.mean(user_repr, dim=0)
                user_repr_list.append(user_repr)
                continue

            # 获取会话超图，并且计算 HG/LG embedding
            item_list, session_hyper_edge_index = self._get_session_hypergraph(session_related_items)
            sub_session_embedding, sub_session_edge_index, session_tot2sub = self._before_hyperconv(kg_embedding,
                                                                                                    item_list,
                                                                                                    session_hyper_edge_index)
            session_embedding    = self._get_hyper_emb(sub_session_embedding, sub_session_edge_index, self.hyper_conv_session)
            hg_session_embedding = session_embedding[
                [session_tot2sub[items] for items in item_list]]  # [num_items, emb_dim]
            lg_session_embedding = self._get_linear_emb(item_list, sub_session_edge_index, hg_session_embedding,
                                                        self.linear_conv_item)

            # 获取知识图谱超图，并计算 HG/LG embedding
            entity_list, knowledge_hyper_edge_index = self._get_knowledge_hypergraph(session_related_items)
            sub_knowledge_embedding, sub_knowledge_edge_index, knowledge_tot2sub = self._before_hyperconv(kg_embedding,
                                                                                                          item_list,
                                                                                                          knowledge_hyper_edge_index)
            raw_knowledge_embedding = self._get_hyper_emb(sub_knowledge_embedding, sub_knowledge_edge_index, self.hyper_conv_knowledge)
            hg_knowledge_embedding  = self._get_knowledge_embedding(item_list, raw_knowledge_embedding,
                                                                   knowledge_tot2sub)
            lg_knowledge_embedding  = self._get_linear_emb(item_list, sub_knowledge_edge_index, hg_knowledge_embedding,
                                                          self.linear_conv_entity)

            # 获取 ConceptNet 超图，并计算 HG/LG embedding
            word_list, conceptnet_hyper_edge_index = self._get_conceptnet_hypergraph(session_related_items)
            sub_conceptnet_embedding, sub_conceptnet_edge_index, conceptnet_tot2sub = self._before_hyperconv(
                kg_embedding, item_list, conceptnet_hyper_edge_index)
            raw_conceptnet_embedding = self._get_hyper_emb(sub_conceptnet_embedding, sub_conceptnet_edge_index, self.hyper_conv_word)
            hg_conceptnet_embedding  = self._get_conceptnet_embedding(item_list, raw_conceptnet_embedding,
                                                                     conceptnet_tot2sub)
            lg_conceptnet_embedding  = self._get_linear_emb(item_list, sub_conceptnet_edge_index,
                                                           hg_conceptnet_embedding, self.linear_conv_word)

            # 获取评论超图，并计算 HG/LG embedding
            review_entity = review['review_entitys']
            review_list, review_hyper_edge_index = self._get_review_hypergraph(review_entity)
            sub_review_embedding, sub_review_edge_index, review_tot2sub = self._before_hyperconv(kg_embedding,
                                                                                                 review_list,
                                                                                                 review_hyper_edge_index)
            review_embedding    = self._get_hyper_emb(sub_review_embedding, sub_review_edge_index, self.hyper_conv_review)
            hg_review_embedding = review_embedding[[review_tot2sub[items] for items in review_list]]
            lg_review_embedding = self._get_linear_emb(review_list, sub_review_edge_index, hg_review_embedding,
                                                       self.linear_conv_review)

            def pad_embeddings(embedding, max_num_nodes):
                """对 embedding 进行填充"""
                # embedding: [num_nodes, n_dim]
                num_nodes, emb_dim = embedding.size()

                # 如果节点数小于 max_num_nodes，进行零填充
                if num_nodes < max_num_nodes:
                    # 生成零填充
                    padding = torch.zeros((max_num_nodes - num_nodes, emb_dim), device=embedding.device)
                    # 拼接填充
                    padded_embedding = torch.cat([embedding, padding], dim=0)
                else:
                    # 如果节点数已经达到或超过 max_num_nodes，则不需要填充
                    padded_embedding = embedding

                # padded embedding: [max_num_nodes, n_dim]
                return padded_embedding

            # 获取各个超图 embedding 的节点数量
            num_session_nodes    = hg_session_embedding.size(0)
            num_knowledge_nodes  = hg_knowledge_embedding.size(0)
            num_conceptnet_nodes = hg_conceptnet_embedding.size(0)
            num_review_nodes     = hg_review_embedding.size(0)

            # 找到最大节点数
            max_num_nodes        = max(num_session_nodes, num_knowledge_nodes, num_conceptnet_nodes, num_review_nodes)

            # 填充 embedding，保证尺寸一致
            hg_session_embedding    = pad_embeddings(hg_session_embedding, max_num_nodes)
            hg_knowledge_embedding  = pad_embeddings(hg_knowledge_embedding, max_num_nodes)
            hg_conceptnet_embedding = pad_embeddings(hg_conceptnet_embedding, max_num_nodes)
            hg_review_embedding     = pad_embeddings(hg_review_embedding, max_num_nodes)

            # lg_session_embedding    = torch.zeros_like(hg_session_embedding)
            # lg_knowledge_embedding  = torch.zeros_like(hg_knowledge_embedding)
            # lg_conceptnet_embedding = torch.zeros_like(hg_conceptnet_embedding)
            # lg_review_embedding     = torch.zeros_like(hg_review_embedding)

            lg_session_embedding    = pad_embeddings(lg_session_embedding, max_num_nodes)
            lg_knowledge_embedding  = pad_embeddings(lg_knowledge_embedding, max_num_nodes)
            lg_conceptnet_embedding = pad_embeddings(lg_conceptnet_embedding, max_num_nodes)
            lg_review_embedding     = pad_embeddings(lg_review_embedding, max_num_nodes)

            # 计算对比损失
            hyper_embs_list = [hg_session_embedding, hg_knowledge_embedding, hg_conceptnet_embedding,
                               hg_review_embedding]
            linear_embs_list = [lg_session_embedding, lg_knowledge_embedding, lg_conceptnet_embedding,
                                lg_review_embedding]

            # 1. 逐对计算超图和线图的对比损失
            for hyper_emb, linear_emb in zip(hyper_embs_list, linear_embs_list):
                # print(f"Shape:\n"
                #       f"Hypergraph embedding: {hyper_emb.shape}\n"  # [num_nodes, n_dim]
                #       f"Linegraph  embedding: {linear_emb.shape}")  # [num_nodes, n_dim]
                linear_ssl_loss = self.SSL(hyper_emb,linear_emb)
                sample_ssl_loss += linear_ssl_loss
                # print(f"Linear SSL loss: {linear_ssl_loss}")

            # 2. 计算不同超图和线图 embedding 之间的对比损失
            for i in range(len(hyper_embs_list)):
                for j in range(i + 1, len(hyper_embs_list)):
                    # 超图间对比损失
                    hg_ssl_loss = self.SSL(hyper_embs_list[i], hyper_embs_list[j])
                    lg_ssl_loss = self.SSL(linear_embs_list[i], linear_embs_list[j])
                    sample_ssl_loss += hg_ssl_loss
                    sample_ssl_loss += lg_ssl_loss
                    # print(f"HG SSL loss: {hg_ssl_loss}\n"
                    #       f"LG SSL loss: {lg_ssl_loss}")

            # 考虑所有 embedding (hg/lg) 进行融合
            if len(context_entities) == 0:
                user_repr = self._attention_and_gating(hg_session_embedding, hg_knowledge_embedding,
                                                       hg_conceptnet_embedding,
                                                       None, hg_review_embedding,
                                                       lg_session_embedding, lg_knowledge_embedding,
                                                       lg_conceptnet_embedding, lg_review_embedding)
            else:
                context_embedding = kg_embedding[context_entities]
                user_repr = self._attention_and_gating(hg_session_embedding, hg_knowledge_embedding,
                                                       hg_conceptnet_embedding, context_embedding, hg_review_embedding,
                                                       lg_session_embedding, lg_knowledge_embedding,
                                                       lg_conceptnet_embedding, lg_review_embedding)

            # print(f"--- Sample SSL loss: {sample_ssl_loss} ---\n")
            user_repr_list.append(user_repr)
            batch_ssl_loss += sample_ssl_loss

        # User embedding : [batch_size, n_dim]
        user_embedding = torch.stack(user_repr_list, dim=0)
        batch_ssl_loss /= batch_size

        # print(f"--- Batch SSL loss :{batch_ssl_loss} ---\n")
        return user_embedding, batch_ssl_loss

    @staticmethod
    @njit(parallel=True)
    def compute_adjacency_matrix(n, hyper_edge_index):
        adjacency_matrix = np.zeros((n, n), dtype=np.float32)

        for i in prange(n):
            for j in prange(i + 1, n):
                i_edges = hyper_edge_index[0][hyper_edge_index[1] == i]
                j_edges = hyper_edge_index[0][hyper_edge_index[1] == j]
                union_len = np.union1d(i_edges, j_edges).size
                if union_len == 0:
                    w = 0.0
                else:
                    w = np.intersect1d(i_edges, j_edges).size / union_len
                adjacency_matrix[i, j] = w
                adjacency_matrix[j, i] = w

        return adjacency_matrix

    def _get_linear_emb(self, hypergraph_nodes, hyper_edge_index, embedding, linear_conv):
        """计算线图 embedding"""
        # from time import perf_counter

        n = len(hypergraph_nodes)
        # start = perf_counter()

        # 将 hyper_edge_index 从 PyTorch 张量转换为 NumPy 数组
        hyper_edge_index_np = hyper_edge_index.cpu().numpy()

        # 使用 njit 加速计算邻接矩阵
        adjacency_matrix = self.compute_adjacency_matrix(n, hyper_edge_index_np)

        # end = perf_counter()
        # print(n, end - start)

        # 转换为 PyTorch 张量
        adjacency_matrix = torch.tensor(adjacency_matrix, device=embedding.device)

        # 得到边和权重
        edge_index  = torch.nonzero(adjacency_matrix, as_tuple=False).T.to(embedding.device)
        edge_weight = adjacency_matrix[edge_index[0], edge_index[1]]

        for conv in linear_conv:
            conv      = conv.to(embedding.device)
            embedding = conv(embedding, edge_index, edge_weight)

        return embedding

    def _get_hyper_emb(self, embedding, index, hyper_conv):
        for conv in hyper_conv:
            conv      = conv.to(embedding.device)
            embedding = conv(embedding, index)
        return embedding

    # def _get_linear_emb(self, hypergraph_nodes, hyper_edge_index, hyper_embs, conv):
    #     """计算线图 embedding"""
    #     from time import perf_counter
    #     # 构造线图的边
    #     n = len(hypergraph_nodes)
    #     adjacency_matrix = torch.zeros((n, n), dtype=torch.float32, device=hyper_embs.device)
    #
    #     start = perf_counter()
    #     # 计算交并比作为相似度，作为线图边权
    #     for i in range(n):
    #         for j in range(i + 1, n):
    #             i_set = set(hyper_edge_index[0][hyper_edge_index[1] == i].tolist())
    #             j_set = set(hyper_edge_index[0][hyper_edge_index[1] == j].tolist())
    #             if len(i_set | j_set) == 0:
    #                 w = 0.0
    #             else:
    #                 w = len(i_set & j_set) / len(i_set | j_set)
    #             adjacency_matrix[i, j] = w
    #             adjacency_matrix[j, i] = w
    #     end = perf_counter()
    #     print(n, end - start)
    #
    #     # 得到边和权重
    #     edge_index  = torch.nonzero(adjacency_matrix, as_tuple=False).T.to(hyper_embs.device)
    #     edge_weight = adjacency_matrix[edge_index[0], edge_index[1]]
    #
    #     # 用 GCN 卷积计算新的线图嵌入
    #     linear_embs = conv(x=hyper_embs, edge_index=edge_index, edge_weight=edge_weight)
    #
    #     return linear_embs

    # 自监督对比学习
    def SSL(self, first_emb, second_emb):
        # L2 归一化
        first_emb  = F.normalize(first_emb, p=2, dim=1)
        second_emb = F.normalize(second_emb, p=2, dim=1)

        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        # 计算正样本和负样本的对比损失
        pos  = score(first_emb, second_emb)
        neg1 = score(second_emb, row_column_shuffle(first_emb))
        one  = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)

        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg1))))
        return con_loss

    # 推荐模块
    def recommend(self, batch, mode):
        # 获取数据
        related_entities, related_items = batch['related_entities'], batch['related_items']
        review = batch['review']
        context_entities, item = batch['context_entities'], batch['item']
        kg_embedding = self.kg_encoder(self.kg_embedding.weight, self.edge_idx, self.edge_type)  # (n_entity, emb_dim)
        con_embedding = self.con_encoder(self.con_embedding.weight, self.edge_idx, self.edge_type)
        extended_items = batch['extended_items']
        for i in range(len(related_items)):
            truncate = min(int(max(2, int(len(related_items[i]) / 4))), len(extended_items[i]))
            if self.extension_strategy == 'Adaptive':
                related_items[i] = related_items[i] + extended_items[i][:truncate]
            else:
                assert self.extension_strategy == 'Random'
                extended_items_sample = random.sample(extended_items[i], truncate)
                related_items[i] = related_items[i] + extended_items_sample

        # 获取用户编码
        user_embedding, ss_loss = self.encode_user(
            related_entities,
            related_items,
            context_entities,
            review,
            kg_embedding,
            con_embedding,
        )  # (batch_size, emb_dim)

        # 计算各实体得分
        scores = F.linear(user_embedding, kg_embedding, self.rec_bias.bias)  # (batch_size, n_entity)
        if self.dataset == 'HReDial':
            beta   = 0.01  # ReDial
            # beta   = 0.005  # .4337
            # beta   = 0.05  # .4279
            # beta   = 0.002  # .4354
            # beta   = 0.001  # .4354
        else:
            # beta   = 0.0001 # .08392
            # beta   = 0.001 # .08425
            # beta   = 0.0005 # .08392
            beta   = 0.0002 # .08458，SOTA
            # beta   = 0.005 # .07859
        loss   = (1.0 - beta) * self.rec_loss(scores, item) + beta * ss_loss
        return loss, scores

    def _starts(self, batch_size):
        """Return bsz start tokens."""
        return self.START.detach().expand(batch_size, 1)

    def freeze_parameters(self):
        freeze_models = [
            self.kg_embedding,
            self.kg_encoder,
            self.hyper_conv_session,
            self.hyper_conv_knowledge,
            self.item_attn,
            self.rec_bias
        ]
        if self.pooling == "Attn":
            freeze_models.append(self.kg_attn)
            freeze_models.append(self.kg_attn_his)
        for model in freeze_models:
            for p in model.parameters():
                p.requires_grad = False

    # 预处理
    def _before_hyperconv(self, embeddings:torch.FloatTensor, hypergraph_items:List[int], edge_index:torch.LongTensor):
        sub_items  = []
        edge_index = edge_index.cpu().numpy()
        for item in hypergraph_items:
            sub_items += [item] + list(self.adj[item])
        sub_items  = list(set(sub_items))
        tot2sub    = {tot:sub for sub, tot in enumerate(sub_items)}
        # print(tot2sub)
        # 准备子图的 embedding 和边索引
        sub_embeddings = embeddings[sub_items]
        sub_edge_index = torch.LongTensor([[tot2sub[v] for v in edge_index[0]], edge_index[1]]).to(self.device)
        return sub_embeddings, sub_edge_index, tot2sub

    # 编码会话
    def encode_session(self, batch_related_entities, batch_related_items, batch_context_entities, kg_embedding,
                       con_embedding, review):
        """
            Return: session_repr (batch_size, batch_seq_len, token_emb_dim), mask (batch_size, batch_seq_len)
        """
        session_repr_list = []
        for session_related_entities, session_related_items, context_entities in zip(batch_related_entities,
                                                                                     batch_related_items,
                                                                                     batch_context_entities):
            flattened_session_related_items = self.flatten(session_related_items)

            # COLD START
            if len(flattened_session_related_items) == 0:
                if len(context_entities) == 0:
                    session_repr_list.append(None)
                else:
                    session_repr = kg_embedding[context_entities]
                    session_repr_list.append(session_repr)
                continue

            # 获取会话超图，并计算 embedding
            hypergraph_items, session_hyper_edge_index = self._get_session_hypergraph(session_related_items)
            sub_session_embedding, sub_session_edge_index, session_tot2sub = self._before_hyperconv(kg_embedding, hypergraph_items, session_hyper_edge_index)
            session_embedding = self.hyper_conv_session(sub_session_embedding, sub_session_edge_index)
            session_embedding = session_embedding[[session_tot2sub[items] for items in hypergraph_items]]

            # 获取知识图谱超图，并计算 embedding
            _, knowledge_hyper_edge_index = self._get_knowledge_hypergraph(session_related_items)
            sub_knowledge_embedding, sub_knowledge_edge_index, knowledge_tot2sub = self._before_hyperconv(kg_embedding, hypergraph_items, knowledge_hyper_edge_index)
            raw_knowledge_embedding = self.hyper_conv_knowledge(sub_knowledge_embedding, sub_knowledge_edge_index)
            knowledge_embedding = self._get_knowledge_embedding(hypergraph_items, raw_knowledge_embedding, knowledge_tot2sub)

            # 获取 ConceptNet 超图，并计算 embedding
            _, conceptnet_hyper_edge_index = self._get_conceptnet_hypergraph(session_related_items)
            sub_conceptnet_embedding, sub_conceptnet_edge_index, conceptnet_tot2sub = self._before_hyperconv(kg_embedding, hypergraph_items, conceptnet_hyper_edge_index)
            raw_conceptnet_embedding = self.hyper_conv_conceptnet(sub_conceptnet_embedding, sub_conceptnet_edge_index)
            conceptnet_embedding = self._get_conceptnet_embedding(hypergraph_items, raw_conceptnet_embedding, conceptnet_tot2sub)

            # 获取评论数据，并计算 embedding
            review_entity = review['review_entitys']
            hypergraph_items, review_hyper_edge_index = self._get_review_hypergraph(review_entity)
            sub_review_embedding, sub_review_edge_index, review_tot2sub = self._before_hyperconv(kg_embedding, hypergraph_items, review_hyper_edge_index)
            review_embedding = self.hyper_conv_review(sub_review_embedding, sub_review_edge_index)
            review_embedding = review_embedding[[review_tot2sub[items] for items in hypergraph_items]]

            # 数据拼接
            if len(context_entities) == 0:
                session_repr = torch.cat(
                    (session_embedding, knowledge_embedding, conceptnet_embedding, review_embedding), dim=0)
                session_repr_list.append(session_repr)
            else:
                context_embedding = kg_embedding[context_entities]
                session_repr = torch.cat(
                    (session_embedding, knowledge_embedding, context_embedding, conceptnet_embedding, review_embedding),
                    dim=0)
                session_repr_list.append(session_repr)

        batch_seq_len = max([session_repr.size(0) for session_repr in session_repr_list if session_repr is not None])
        mask_list = []
        for i in range(len(session_repr_list)):
            if session_repr_list[i] is None:
                mask_list.append([False] * batch_seq_len)
                zero_repr = torch.zeros((batch_seq_len, self.kg_emb_dim), device=self.device, dtype=torch.float)
                session_repr_list[i] = zero_repr
            else:
                mask_list.append(
                    [False] * (batch_seq_len - session_repr_list[i].size(0)) + [True] * session_repr_list[i].size(0))
                zero_repr = torch.zeros((batch_seq_len - session_repr_list[i].size(0), self.kg_emb_dim),
                                        device=self.device, dtype=torch.float)
                session_repr_list[i] = torch.cat((zero_repr, session_repr_list[i]), dim=0)

        session_repr_embedding = torch.stack(session_repr_list, dim=0)
        session_repr_embedding = self.entity_to_token(session_repr_embedding)
        # print("session_repr_embedding.shape", session_repr_embedding.shape) # [6, 7, 300]
        return session_repr_embedding, torch.tensor(mask_list, device=self.device, dtype=torch.bool)

    # 生成对话 - teacher forcing 策略，使用真实的目标词作为输入，来生成下一个词的预测，用于训练阶段
    def decode_forced(self, related_encoder_state, context_encoder_state, session_state, user_embedding, resp):
        bsz    = resp.size(0)
        seqlen = resp.size(1)
        inputs = resp.narrow(1, 0, seqlen - 1)  # 去掉最后一个词
        inputs = torch.cat([self._starts(bsz), inputs], 1)  # 添加起始标记
        latent, _    = self.decoder(inputs, related_encoder_state, context_encoder_state, session_state)

        # 计算logits
        token_logits = F.linear(latent, self.token_embedding.weight)
        user_logits  = self.user_proj_2(torch.relu(self.user_proj_1(user_embedding))).unsqueeze(1)

        # 计算复制相关的 logits
        user_latent  = self.entity_to_token(user_embedding)
        user_latent  = user_latent.unsqueeze(1).expand(-1, seqlen, -1)
        copy_latent  = torch.cat((user_latent, latent), dim=-1)
        copy_logits  = self.copy_proj_2(torch.relu(self.copy_proj_1(copy_latent)))
        if self.dataset == 'HReDial':
            copy_logits = copy_logits * self.copy_mask.unsqueeze(0).unsqueeze(0)  # not for tg-redial

        # 总 logits
        sum_logits = token_logits + user_logits + copy_logits
        _, preds   = sum_logits.max(dim=-1)
        return sum_logits, preds

    # 生成对话 - greedy 策略，每一步选择概率最高的词作为输出，并将其作为下一步的输入，用于推理阶段
    def decode_greedy(self, related_encoder_state, context_encoder_state, session_state, user_embedding):
        bsz = context_encoder_state[0].shape[0]
        xs  = self._starts(bsz)
        incr_state = None
        logits     = []
        for i in range(self.longest_label):
            scores, incr_state = self.decoder(xs, related_encoder_state, context_encoder_state, session_state, incr_state)  # incr_state is always None
            scores = scores[:, -1:, :]
            token_logits = F.linear(scores, self.token_embedding.weight)
            user_logits  = self.user_proj_2(torch.relu(self.user_proj_1(user_embedding))).unsqueeze(1)

            user_latent  = self.entity_to_token(user_embedding)
            user_latent  = user_latent.unsqueeze(1).expand(-1, 1, -1)
            copy_latent  = torch.cat((user_latent, scores), dim=-1)
            copy_logits  = self.copy_proj_2(torch.relu(self.copy_proj_1(copy_latent)))
            if self.dataset == 'HReDial':
                copy_logits = copy_logits * self.copy_mask.unsqueeze(0).unsqueeze(0)  # not for tg-redial
            sum_logits   = token_logits + user_logits + copy_logits
            probs, preds = sum_logits.max(dim=-1)
            logits.append(scores)
            xs = torch.cat([xs, preds], dim=1)
            # check if everyone has generated an end token
            all_finished = ((xs == self.end_token_idx).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    # 对话模块训练
    def converse(self, batch, mode):
        # 获取数据
        related_tokens   = batch['related_tokens']
        context_tokens   = batch['context_tokens']
        related_items    = batch['related_items']
        related_entities = batch['related_entities']
        context_entities = batch['context_entities']
        review           = batch['review']
        response         = batch['response']
        kg_embedding     = self.kg_encoder(self.kg_embedding.weight, self.edge_idx, self.edge_type)  # (n_entity, emb_dim)
        con_embedding    = self.con_encoder(self.con_embedding.weight, self.edge_idx, self.edge_type)

        # 获取超图编码
        session_state = self.encode_session(
            related_entities,
            related_items,
            context_entities,
            kg_embedding,
            con_embedding,
            review,
        )  # (batch_size, batch_seq_len, token_emb_dim)

        # 获取用户编码
        user_embedding, _ = self.encode_user(
            related_entities,
            related_items,
            context_entities,
            review,
            kg_embedding,
            con_embedding,
        )  # (batch_size, emb_dim)

        # 获取 X_c、X_h
        related_encoder_state = self.related_encoder(related_tokens)
        context_encoder_state = self.context_encoder(context_tokens)

        # 对话生成
        if mode != 'test':
            self.longest_label = max(self.longest_label, response.shape[1])
            logits, preds = self.decode_forced(related_encoder_state, context_encoder_state, session_state,
                                               user_embedding, response)
            logits = logits.view(-1, logits.shape[-1])
            labels = response.view(-1)
            return self.conv_loss(logits, labels), preds
        else:
            _, preds = self.decode_greedy(related_encoder_state, context_encoder_state, session_state, user_embedding)
            return preds

    # 推荐模块和对话模块分开训练
    def forward(self, batch, mode, stage):
        if len(self.gpu) >= 2:
            self.edge_idx = self.edge_idx.cuda(torch.cuda.current_device())
            self.edge_type = self.edge_type.cuda(torch.cuda.current_device())
        if stage == "conv":
            return self.converse(batch, mode)
        if stage == "rec":
            return self.recommend(batch, mode)