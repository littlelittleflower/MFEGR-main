import os
import logging
import math
import random

import numpy
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import BatchType, ModeType, TestDataset
from model.compgcn_conv import CompGCNConv, EventConv
from model.compgcn_conv_basis import CompGCNConvBasis
os.environ['CUDA_VISBLE_DEVICES'] = ' '
class KGEModel(nn.Module, ABC):
    """
    Must define
        `self.entity_embedding`
        `self.relation_embedding`
    in the subclasses.
    """
    #
    # @abstractmethod
    # def get_classify(self, path):
    #     """
    #     path: [batch_size, path_length]
    #     """
    #     ...

    # def calculate_mse_loss(self, sub_emb, rel_emb, third_tri_index, obj_embeed, w):
    #     # obj_id = actor2[third_tri_index]
    #     w = w[third_tri_index]
    #     rel_emb = rel_emb[third_tri_index]
    #     ht = self.layer(sub_emb, obj_embeed)
    #     htr = torch.unsqueeze(torch.sum(rel_emb * (ht), dim=1), dim=-1)
    #     f_prob_h = self.liner(htr)  # 就是这个，记住
    #     f_prob_h = torch.squeeze(f_prob_h, dim=-1)
    #     f_score_h = torch.sum(torch.square(f_prob_h - w)) / len(third_tri_index)
    #     return f_score_h, f_prob_h

    @abstractmethod
    def evaluate_mse(self, prediction, truth):
        ...

    @abstractmethod
    def calculate_mse_loss(self, paths,sub_emb, rel_emb, third_tri_index, obj_embeed, w,syml,pl_sym):
        ...

    @abstractmethod
    def forward_for_kg_completion(self, *arg_list):
        """
        path: [batch_size, path_length]
        """
        ...
    @abstractmethod
    def get_rule_loss(self, path):
        """
        path: [batch_size, path_length]
        """
        ...

    @abstractmethod
    def func(self, head, rel, tail, batch_type):
        """
        Different tensor shape for different batch types.
        BatchType.SINGLE:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]

        BatchType.HEAD_BATCH:
            head: [batch_size, negative_sample_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]

        BatchType.TAIL_BATCH:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, negative_sample_size, hidden_dim]
        """
        ...

    @abstractmethod
    def comp(self, path):
        """
        path: [batch_size, path_length]
        """
        ...

    def forward(self, sample,classfy, *event_related_args,batch_type=BatchType.SINGLE):#都会来到这，调用其他的函数
        """
        #positive_ents, negative_ents, paths
        #
        Given the indexes in `sample`, extract the corresponding embeddings,
        and call func().

        Args:
            batch_type: {SINGLE, HEAD_BATCH, TAIL_BATCH},
                - SINGLE: positive samples in training, and all samples in validation / testing,
                - HEAD_BATCH: (?, r, t) tasks in training,
                - TAIL_BATCH: (h, r, ?) tasks in training.

            sample: different format for different batch types.
                - SINGLE: tensor with shape [batch_size, 3]
                - {HEAD_BATCH, TAIL_BATCH}: (positive_sample, negative_sample)
                    - positive_sample: tensor with shape [batch_size, 3]
                    - negative_sample: tensor with shape [batch_size, negative_sample_size]
        """
        syml,fuzzy_all_tri,sub,rel,init_ent_list, sub_index_list, obj_index_list, node_pos_dict, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type = event_related_args
        if syml == 'train':
            fuzzy_tri = fuzzy_all_tri[:,3]
        else:
            fuzzy_tri = torch.from_numpy(numpy.array([fuzzy_all_tri[3].item()])).cuda()
        arg_list = [syml,sub, rel, sub_index_list, init_ent_list,final_ent_list, ent_neighbors, rel_evt_idxs,
                                                                   evt_neighbors, new_event_index,
                                                                   new_entity_event_index, new_entity_mask,
                                                                   new_entity_list, new_entity_type]

        # sub_emb, rel_emb, all_ent = self.forward_for_kg_completion(*arg_list)#这里就死学习到的嵌入

        first_tri_index, first_tri, second_tri_index, second_tri, third_tri_index, third_tri = classfy
        if batch_type == BatchType.SINGLE:
            positive_ents, paths = sample
            positive_ents, paths = positive_ents.long(), paths.long()

            # head = torch.index_select(
            #     self.entity_embedding,
            #     dim=0,
            #     index=positive_ents[:, 0]
            # ).unsqueeze(1)
            #
            # relation = self.comp(paths)
            #
            # tail = torch.index_select(
            #     self.entity_embedding,
            #     dim=0,
            #     index=positive_ents[:, 1]
            # ).unsqueeze(1)

            if len(third_tri_index) == 0:#entity,entity
                third_tri_head,third_tri_tail,third_tri_relation = [],[],[]
            else:
                #[512,1000]?
                # index = third_tri[:, 0]
                # third_tri_negative_ents = negative_ents[third_tri_index]

                #模糊度预测的部分
                # obj_embeed = self.init_embed[o[third_tri_index]]
                # loss_mse, pred_mse = self.calculate_mse_loss(sub_emb, rel_emb, third_tri_index, obj_embeed, fuzzy_low)
                # mse, mae = self.evaluate_mse(pred_mse, fuzzy_low[third_tri_index])  # 举一个例子

                third_tri_head_entity = [i[0] for i in third_tri]
                third_tri_tail_entity = [i[1] for i in third_tri]

                # if syml == 'train':
                #     sub_emb, rel_emb, all_ent = self.forward_for_kg_completion(*arg_list)  # 这里就学习到的嵌入
                #     ent2ent_loss_mse, ent2ent_pred_mse = self.calculate_mse_loss(sub_emb, rel_emb,
                #                                                                          third_tri_index,
                #                                                                          obj_embeed,
                #                                                                          fuzzy_tri)
                #     third_tri_head = sub_emb.unsqueeze(1)
                # else:
                #     ent2ent_loss_mse, ent2ent_pred_mse = self.calculate_mse_loss(self.init_embed, self.relation_embedding,
                #                                                                          third_tri_index,
                #                                                                          obj_embeed,
                #                                                                          fuzzy_tri)
                third_tri_head = torch.index_select(
                    self.init_embed,#换成事件图谱的嵌入？
                    dim=0,
                    index=torch.tensor(third_tri_head_entity).cuda()
                ).unsqueeze(1)
                tail = positive_ents[:, 1]
                obj_embeed = self.init_embed[tail[third_tri_index]]
                if syml == 'train':
                    sub_emb, rel_emb, all_ent = self.forward_for_kg_completion(*arg_list)  # 这里就学习到的嵌入
                    pl = 'yes'
                else:
                    rel_emb = self.fc(self.relation_embedding)#？？
                    sub_emb = third_tri_head.squeeze(1)
                    pl = 'no'

                ent2ent_loss_mse, ent2ent_pred_mse = self.calculate_mse_loss(paths,sub_emb, rel_emb,
                                                                                     third_tri_index, obj_embeed,
                                                                                     fuzzy_tri,syml,pl)


                third_tri_relation = self.comp(paths[third_tri_index])  # 三个一样的tuple

                third_tri_tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=torch.tensor(third_tri_tail_entity).cuda()
                ).unsqueeze(1)

            if len(first_tri_index) == 0:#evt,?,evt
                first_tri_head,first_tri_tail,first_tri_rel = [],[],[]
            else:


                # if syml == 'train':
                #     sub_emb, rel_emb, all_ent = self.forward_for_kg_completion(*arg_list)  # 这里就学习到的嵌入


                first_tri_tail_entity = [i[1] for i in first_tri]
                first_tri_head_entity = [i[0] for i in first_tri]
                if syml == 'test':
                    if first_tri_head_entity[0]>=8888:
                        first_tri_head_entity = [random.randint(0,8887)]

                first_tri_head = torch.index_select(
                    self.event_embed,
                    dim=0,
                    index=torch.tensor(first_tri_head_entity).cuda()
                ).unsqueeze(1)

                tail = positive_ents[:, 1]
                obj_embeed = self.init_embed[tail[first_tri_index]]#event_embed
                # if syml == 'train':#只有实体才会用forward_for_kg_completion，其余不用
                #     sub_emb, rel_emb, all_ent = self.forward_for_kg_completion(*arg_list)  # 这里就学习到的嵌入
                # else:
                rel_emb = self.fc(self.relation_embedding)
                sub_emb = first_tri_head.squeeze(1)
                event2event_loss_mse, event2event_pred_mse = self.calculate_mse_loss(paths,sub_emb, rel_emb,first_tri_index,obj_embeed,fuzzy_tri,syml,'no')

                first_tri_relation = self.comp(paths[first_tri_index])  # 三个一样的tuple

                first_tri_tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=torch.tensor(first_tri_tail_entity).cuda()
                ).unsqueeze(1)

            if len(second_tri_index) == 0:#evt,?,arg
                second_tri_head,second_tri_tail,second_tri_relation = [],[],[]
            else:

                # tail = positive_ents[:, 1]
                # # head = positive_ents[:, 1]
                # obj_embeed = self.init_embed[tail[second_tri_index]]
                # # sub_emb = self.event_embed[second_tri_index]
                # if syml == 'train':
                #     sub_emb, rel_emb, all_ent = self.forward_for_kg_completion(*arg_list)  # 这里就学习到的嵌入
                #     event2arg_loss_mse, event2arg_pred_mse = self.calculate_mse_loss(sub_emb, rel_emb,second_tri_index,obj_embeed,fuzzy_tri)


                # second_tri_negative_ents = negative_ents[second_tri_index]
                second_tri_head_entity = [i[0] for i in second_tri]
                second_tri_tail_entity = [i[1] for i in second_tri]
                second_tri_head = torch.index_select(
                    self.event_embed,
                    dim=0,
                    index=torch.tensor(second_tri_head_entity).cuda()
                ).unsqueeze(1)

                tail = positive_ents[:, 1]
                obj_embeed = self.init_embed[tail[second_tri_index]]#实体，不是事件
                # if syml == 'train':#只有实体用，其余不用
                #     sub_emb, rel_emb, all_ent = self.forward_for_kg_completion(*arg_list)  # 这里就学习到的嵌入
                # else:
                rel_emb = self.fc(self.relation_embedding)
                sub_emb = second_tri_head.squeeze(1)
                event2arg_loss_mse, event2arg_pred_mse = self.calculate_mse_loss(paths,sub_emb, rel_emb,second_tri_index,obj_embeed,fuzzy_tri,syml,'no')

                second_tri_relation = self.comp(paths[second_tri_index])  # 三个一样的tuple

                second_tri_tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=torch.tensor(second_tri_tail_entity).cuda()
                ).unsqueeze(1)


        elif batch_type == BatchType.HEAD_BATCH:
            positive_ents, negative_ents, paths = sample
            positive_ents, negative_ents, paths = positive_ents.long(), negative_ents.long(), paths.long()
            batch_size, negative_sample_size = negative_ents.size(0), negative_ents.size(1)
            # relation = self.comp(paths)
            if len(third_tri_index) == 0:
                third_tri_head,third_tri_tail,third_tri_relation = [],[],[]
            else:
                #[512,1000]?
                # index = third_tri[:, 0]
                third_tri_negative_ents = negative_ents[third_tri_index]
                third_tri_head_entity = [i[1] for i in third_tri]#尾变成了头

                third_tri_head = torch.index_select(
                    self.init_embed,
                    dim=0,
                    index=torch.tensor(third_tri_head_entity).cuda()
                ).unsqueeze(1)
                tail = positive_ents[:, 1]
                obj_embeed = self.init_embed[tail[third_tri_index]]
                if syml == 'train':
                    sub_emb, rel_emb, all_ent = self.forward_for_kg_completion(*arg_list)  # 这里就学习到的嵌入
                    pl = 'yes'
                else:
                    rel_emb = self.fc(self.relation_embedding)#？？
                    sub_emb = third_tri_head.squeeze(1)
                    pl = 'no'

                ent2ent_loss_mse, ent2ent_pred_mse = self.calculate_mse_loss(paths,sub_emb, rel_emb,
                                                                             third_tri_index, obj_embeed,
                                                                             fuzzy_tri,syml,pl)

                third_tri_relation = self.comp(paths[third_tri_index])  # 三个一样的tuple

                third_tri_tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=third_tri_negative_ents.view(-1)
                ).view(len(third_tri_index), negative_sample_size, -1)

            if len(first_tri_index) == 0:
                first_tri_head,first_tri_tail,first_tri_relation = [],[],[]
            else:
                first_tri_negative_ents = negative_ents[first_tri_index]#超过了实体的范围
                first_tri_head_entity = [i[0] for i in first_tri]
                first_tri_head = torch.index_select(
                    self.event_embed,
                    dim=0,
                    index=torch.tensor(first_tri_head_entity).cuda()
                ).unsqueeze(1)

                tail = positive_ents[:, 1]
                obj_embeed = self.init_embed[tail[first_tri_index]]#event
                # if syml == 'train':#只有实体才会用forward_for_kg_completion，其余不用
                #     sub_emb, rel_emb, all_ent = self.forward_for_kg_completion(*arg_list)  # 这里就学习到的嵌入
                # else:
                rel_emb = self.fc(self.relation_embedding)
                sub_emb = first_tri_head.squeeze(1)
                event2event_loss_mse, event2event_pred_mse = self.calculate_mse_loss(paths,sub_emb, rel_emb, first_tri_index,
                                                                                     obj_embeed, fuzzy_tri,syml,'no')


                first_tri_relation = self.comp(paths[first_tri_index])  # 三个一样的tuple

                first_tri_tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=first_tri_negative_ents.view(-1)
                ).view(len(first_tri_index), negative_sample_size, -1)

            if len(second_tri_index) == 0:
                second_tri_head,second_tri_tail,second_tri_relation = [],[],[]
            else:
                second_tri_negative_ents = negative_ents[second_tri_index]
                second_tri_head_entity = [i[0] for i in second_tri]
                second_tri_head = torch.index_select(
                    self.event_embed,
                    dim=0,
                    index=torch.tensor(second_tri_head_entity).cuda()
                ).unsqueeze(1)


                tail = positive_ents[:, 1]
                obj_embeed = self.init_embed[tail[second_tri_index]]#实体，不是事件
                # if syml == 'train':#只有实体用，其余不用
                #     sub_emb, rel_emb, all_ent = self.forward_for_kg_completion(*arg_list)  # 这里就学习到的嵌入
                # else:
                rel_emb = self.fc(self.relation_embedding)
                sub_emb = second_tri_head.squeeze(1)
                event2arg_loss_mse, event2arg_pred_mse = self.calculate_mse_loss(paths,sub_emb, rel_emb,second_tri_index,obj_embeed,fuzzy_tri,syml,'no')

                second_tri_relation = self.comp(paths[second_tri_index])  # 三个一样的tuple

                second_tri_tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=second_tri_negative_ents.view(-1)
                ).view(len(second_tri_index), negative_sample_size, -1)
            # head = torch.index_select(
            #     self.entity_embedding,
            #     dim=0,
            #     index=positive_ents[:, 1]
            # ).unsqueeze(1)



            # tail = torch.index_select(
            #     self.entity_embedding,
            #     dim=0,
            #     index=negative_ents.view(-1)
            # ).view(batch_size, negative_sample_size, -1)

        elif batch_type == BatchType.TAIL_BATCH:#首先的位置
            positive_ents, negative_ents, paths = sample#negative_ents
            positive_ents, negative_ents, paths = positive_ents.long(), negative_ents.long(), paths.long()
            # print(torch.max(negative_ents))
            batch_size, negative_sample_size = negative_ents.size(0), negative_ents.size(1)
            # relation = self.comp(paths)
            if len(third_tri_index) == 0:
                third_tri_head,third_tri_tail,third_tri_relation = [],[],[]
            else:
                #[512,1000]?
                # index = third_tri[:, 0]
                third_tri_negative_ents = negative_ents[third_tri_index]
                third_tri_head_entity = [i[0] for i in third_tri]
                #(454,1,200)
                # if syml == 'train':
                #     sub_emb, rel_emb, all_ent = self.forward_for_kg_completion(*arg_list)  # 这里就学习到的嵌入
                #     third_tri_head = sub_emb.unsqueeze(1)
                # else:
                third_tri_head = torch.index_select(
                    self.init_embed,
                    dim=0,
                    index=torch.tensor(third_tri_head_entity).cuda()
                ).unsqueeze(1)

                tail = positive_ents[:, 1]
                obj_embeed = self.init_embed[tail[third_tri_index]]#这里越界了？
                if syml == 'train':
                    sub_emb, rel_emb, all_ent = self.forward_for_kg_completion(*arg_list)  # 这里就学习到的嵌入
                    pl = 'yes'
                else:
                    rel_emb = self.fc(self.relation_embedding)
                    sub_emb = third_tri_head.squeeze(1)
                    pl = 'no'

                ent2ent_loss_mse, ent2ent_pred_mse = self.calculate_mse_loss(paths,sub_emb, rel_emb,
                                                                             third_tri_index, obj_embeed,
                                                                             fuzzy_tri,syml,pl)
                #this place
                third_tri_relation = self.comp(paths[third_tri_index])  # 三个一样的tuple
                #entity_embedding
                #third_tri_negative_ents,可能是负样本超过了
                third_tri_tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=third_tri_negative_ents.view(-1)
                ).view(len(third_tri_index), negative_sample_size, -1)

            if len(first_tri_index) == 0:
                first_tri_head,first_tri_tail,first_tri_relation = [],[],[]
            else:
                first_tri_negative_ents = negative_ents[first_tri_index]#超过了实体的范围
                first_tri_head_entity = [i[0] for i in first_tri]
                # if syml == 'test':
                #     if first_tri_head_entity[0] >=8888:
                #         print('1')
                # 超过了实体的范围
                first_tri_head = torch.index_select(
                    self.init_embed,
                    dim=0,
                    index=torch.tensor(first_tri_head_entity).cuda()
                ).unsqueeze(1)

                tail = positive_ents[:, 1]#可能是这里，测试集总有问题？

                obj_embeed = self.init_embed[tail[first_tri_index]]#event_embed
                # if syml == 'train':#只有实体才会用forward_for_kg_completion，其余不用
                #     sub_emb, rel_emb, all_ent = self.forward_for_kg_completion(*arg_list)  # 这里就学习到的嵌入
                # else:
                rel_emb = self.fc(self.relation_embedding)#38小了
                sub_emb = first_tri_head.squeeze(1)
                event2event_loss_mse, event2event_pred_mse = self.calculate_mse_loss(paths,sub_emb, rel_emb, first_tri_index,
                                                                                     obj_embeed, fuzzy_tri,syml,'no')
                #这里
                first_tri_relation = self.comp(paths[first_tri_index])  # 三个一样的tuple

                first_tri_tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=first_tri_negative_ents.view(-1)
                ).view(len(first_tri_index), negative_sample_size, -1)

            if len(second_tri_index) == 0:
                second_tri_head,second_tri_tail,second_tri_relation = [],[],[]
            else:
                second_tri_negative_ents = negative_ents[second_tri_index]
                second_tri_head_entity = [i[0] for i in second_tri]
                #?? 5301
                second_tri_head = torch.index_select(
                    self.event_embed,
                    dim=0,
                    index=torch.tensor(second_tri_head_entity).cuda()
                ).unsqueeze(1)

                tail = positive_ents[:, 1]
                obj_embeed = self.init_embed[tail[second_tri_index]]  # 实体，不是事件
                # if syml == 'train':#只有实体用，其余不用
                #     sub_emb, rel_emb, all_ent = self.forward_for_kg_completion(*arg_list)  # 这里就学习到的嵌入
                # else:
                rel_emb = self.fc(self.relation_embedding)
                sub_emb = second_tri_head.squeeze(1)
                event2arg_loss_mse, event2arg_pred_mse = self.calculate_mse_loss(paths,sub_emb, rel_emb, second_tri_index,
                                                                                 obj_embeed, fuzzy_tri,syml,'no')

                second_tri_relation = self.comp(paths[second_tri_index])  # 三个一样的tuple

                second_tri_tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=second_tri_negative_ents.view(-1)
                ).view(len(second_tri_index), negative_sample_size, -1)

        else:
            raise ValueError('batch_type %s not supported!'.format(batch_type))
        if syml == 'train':
            rule_loss = self.get_rule_loss(paths)
        else:
            rule_loss = 0
        if len(first_tri_index) != 0:
            score1 = self.func(first_tri_head, first_tri_tail,first_tri_relation,batch_type)
        else:
            score1 = 0
            event2event_loss_mse = 0
            event2event_pred_mse = 0
        if len(second_tri_index) != 0:
            score2 = self.func(second_tri_head, second_tri_tail,second_tri_relation,batch_type)
        else:
            score2 = 0
            event2arg_loss_mse = 0
            event2arg_pred_mse = 0
        if len(third_tri_index)!=0:
            score3 = self.func(third_tri_head, third_tri_tail,third_tri_relation,batch_type)
        else:
            score3 = 0
            ent2ent_loss_mse = 0
            ent2ent_pred_mse = 0
        if syml == 'train':
            #关系也要切片
            return (score1,score2,score3),(event2event_loss_mse,event2arg_loss_mse,ent2ent_loss_mse),(first_tri_head, first_tri_tail,second_tri_head, second_tri_tail,third_tri_head,third_tri_tail),rule_loss
        else:
            if len(second_tri_index) != 0:
                event_entity_mse, event_entity_mae  = self.evaluate_mse(event2arg_pred_mse, fuzzy_tri[second_tri_index])
            else:
                event_entity_mse, event_entity_mae = 0,0
            if len(first_tri_index) != 0:
                event_event_mse, event_event_mae = self.evaluate_mse(event2event_pred_mse, fuzzy_tri[first_tri_index])
            else:
                event_event_mse, event_event_mae = 0,0
            if len(third_tri_index) != 0:
                ent_ent_mse, ent_ent_mae = self.evaluate_mse(ent2ent_pred_mse, fuzzy_tri[third_tri_index])
            else:
                ent_ent_mse, ent_ent_mae = 0,0
            mae = (event_entity_mae+event_event_mae+ent_ent_mae)/3
            mse = (event_entity_mse+event_event_mse+ent_ent_mse)/3
            return (score1, score2, score3), (mse,mae), rule_loss

    @staticmethod
    def train_step(model, rules,optimizer, train_iterator, args,model_event_graph):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()
        #positive_ents可能包括头实体和尾实体
        #len= 1 对应train_1_hop
        #len= 2 对应train_2_hop
        #len= 3 对应train_3_hop
        #读进来模糊度
        positive_ents, negative_ents, paths, subsampling_weight, batch_type,fuzzy_all_tri = next(train_iterator)
        first_tri_index = [] #evt,?,evt
        first_tri = []
        second_tri_index = []#evt,?,arg
        second_tri = []
        third_tri_index = [] #entity,entity
        third_tri = []
        #下面是事件图谱的信息
        sub_entity_index = []
        obj_entity_index = []
        rel_list = []

        sub_event_index = []
        obj_event_index = []

        for index,k in enumerate(zip(positive_ents,paths)):
            entities,rels = k
            head, tail = entities
            # fuzzy = fuzzy.tolist()

            head, tail = head.item(), tail.item()
            rels = rels.tolist()
            head_rel = rels[0]
            tail_ral = rels[len(rels)-1]
            ace_temp = {'after': 1277, '**after': 9399, 'arg': 1118, '**arg': 9242}
            # ere_temp = {'after':18,'**after':37,'arg':17,'**arg':36}
            ere_temp = {'after': 612, '**after': 1222, 'arg': 611, '**arg': 1221}
            if head_rel in [ere_temp['after'],ere_temp['**after']] and tail_ral in [ere_temp['after'],ere_temp['**after']]:#事件-事件
                first_tri_index.append(index)
                first_tri.append([head,tail])
                sub_event_index.append(head)
                obj_event_index.append(tail)
            elif head_rel == ere_temp['arg'] and tail_ral == ere_temp['**arg']:#事件-事件
                first_tri_index.append(index)
                first_tri.append([head, tail])
                sub_event_index.append(head)
                obj_event_index.append(tail)
            elif head_rel in [ere_temp['after'],ere_temp['**after'],ere_temp['arg']] and tail_ral == ere_temp['arg']:#事件-论元
                second_tri_index.append(index)
                second_tri.append([head, tail])
                sub_event_index.append(head)
            else:#实体-实体
                third_tri_index.append(index)
                third_tri.append([head, tail])
                sub_entity_index.append(head)
                obj_entity_index.append(tail)
            rel_list.append(head_rel)#假设只放进去头关系

        #仍然是事件图谱的信息
        init_ent_list = []
        sub_index_list = []
        obj_index_list = []
        node_pos_dict = {}

        for j, idx in enumerate(sub_entity_index):
            if idx not in node_pos_dict:
                init_ent_list.append(idx)
                node_pos_dict.update({idx: len(init_ent_list) - 1})
                sub_index_list.append(len(init_ent_list) - 1)
            else:
                sub_index_list.append(node_pos_dict[idx])

        for j, idx in enumerate(obj_entity_index):
            if idx not in node_pos_dict:
                init_ent_list.append(idx)
                node_pos_dict.update({idx: len(init_ent_list) - 1})
                obj_index_list.append(len(init_ent_list) - 1)
            else:
                obj_index_list.append(node_pos_dict[idx])

        # 得到了当前子图结构的一些信息，然后是怎么利用这些信息了，用图神经网络的形式
        final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type = model_event_graph.sample_subgraph(
            init_ent_list, 'train')
        sub = torch.tensor(sub_entity_index).cuda()
        rel = torch.tensor(rel_list).cuda()
        event_related_args = ['train',fuzzy_all_tri,sub,rel,init_ent_list,sub_index_list,obj_index_list,node_pos_dict,final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type]

        positive_ents = positive_ents.cuda()
        negative_ents = negative_ents.cuda()
        paths = paths.cuda()
        subsampling_weight = subsampling_weight.cuda()
        first_subsampling_weight = subsampling_weight[first_tri_index]
        second_subsampling_weight = subsampling_weight[second_tri_index]
        third_subsampling_weight = subsampling_weight[third_tri_index]

        classfy = (first_tri_index,first_tri,second_tri_index,second_tri,third_tri_index,third_tri)

        # negative scores
        negative_score,neg_mse_all, _,neg_rule_loss = model((positive_ents, negative_ents, paths),classfy,*event_related_args, batch_type=batch_type)
        neg_event2event_loss_mse, neg_event2arg_loss_mse, neg_ent2ent_loss_mse = neg_mse_all
        neg_first_score ,neg_second_score, neg_third_score = negative_score
        neg_first_score = (F.softmax(neg_first_score * args.adversarial_temperature, dim=1).detach()
                          * F.logsigmoid(-neg_first_score)).sum(dim=1)

        neg_second_score = (F.softmax(neg_second_score * args.adversarial_temperature, dim=1).detach()
                           * F.logsigmoid(-neg_second_score)).sum(dim=1)

        neg_third_score = (F.softmax(neg_third_score * args.adversarial_temperature, dim=1).detach()
                            * F.logsigmoid(-neg_third_score)).sum(dim=1)

        # positive scores
        positive_score, pos_mse_all,ent,pos_rule_loss = model((positive_ents, paths),classfy,*event_related_args)#12减去一个矩阵，这个是single模式（默认）
        pos_event2event_loss_mse, pos_event2arg_loss_mse, pos_ent2ent_loss_mse = pos_mse_all
        pos_first_score, pos_second_score, pos_third_score = positive_score

        first_tri_head, first_tri_tail, second_tri_head, second_tri_tail, third_tri_head, third_tri_tail = ent

        pos_first_score = F.logsigmoid(pos_first_score).squeeze(dim=1)
        pos_second_score = F.logsigmoid(pos_second_score).squeeze(dim=1)
        pos_third_score = F.logsigmoid(pos_third_score).squeeze(dim=1)

        first_positive_sample_loss = - (first_subsampling_weight * pos_first_score).sum() / first_subsampling_weight.sum()
        second_positive_sample_loss = - (
                    second_subsampling_weight * pos_second_score).sum() / second_subsampling_weight.sum()
        third_positive_sample_loss = - (
                    third_subsampling_weight * pos_third_score).sum() / third_subsampling_weight.sum()

        positive_sample_loss = (first_positive_sample_loss+second_positive_sample_loss+third_positive_sample_loss)/3
        first_negative_sample_loss = - (first_subsampling_weight * neg_first_score).sum() / first_subsampling_weight.sum()
        second_negative_sample_loss = - (second_subsampling_weight * neg_second_score).sum() / second_subsampling_weight.sum()
        third_negative_sample_loss = - (third_subsampling_weight * neg_third_score).sum() / third_subsampling_weight.sum()

        negative_sample_loss = (first_negative_sample_loss+second_negative_sample_loss+third_negative_sample_loss)/3
        # loss = (positive_sample_loss + negative_sample_loss) / 2
        mse_fuzzy_mean = (neg_event2event_loss_mse + neg_event2arg_loss_mse + neg_ent2ent_loss_mse)/2+(pos_event2event_loss_mse+ pos_event2arg_loss_mse+pos_ent2ent_loss_mse)/2

        loss = (positive_sample_loss + negative_sample_loss+neg_rule_loss + pos_rule_loss) / 2 + 0.1*mse_fuzzy_mean

        if args.regularization:
            # Use regularization
            first_regularization = args.regularization * (
                    first_tri_head.norm(p=args.reg_level) ** args.reg_level +
                    first_tri_tail.norm(p=args.reg_level) ** args.reg_level
            ) / ent[0].shape[0]
            first_loss = loss + first_regularization
        else:
            regularization = torch.tensor([0])

        loss.backward()

        optimizer.step()

        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
            'regularization': regularization.item()
        }

        return log

    @staticmethod
    def test_step(model, data_reader, mode, args,model_event_graph):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        test_dataset = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.TAIL_BATCH
            ),
            batch_size=1,#1？？
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        event_logs = []
        event_logs_hop = defaultdict(list)

        arg_logs = []
        arg_logs_hop = defaultdict(list)

        entity_logs = []
        entity_logs_hop = defaultdict(list)

        step = 0
        all_mse = 0
        all_mae = 0
        total_steps = len(test_dataset)

        with torch.no_grad():
            for positive_ents, negative_ents, paths, neg_size, batch_type,fuzzy_tri in test_dataset:

                first_tri_index = []  # evt,?,evt
                first_tri = []
                second_tri_index = []  # evt,?,arg
                second_tri = []
                third_tri_index = []  # entity,entity
                third_tri = []

                # 下面是事件图谱的信息
                sub_entity_index = []
                obj_entity_index = []
                rel_list = []

                sub_event_index = []
                obj_event_index = []

                for index, k in enumerate(zip(positive_ents, paths)):
                    entities, rels = k
                    head, tail = entities
                    head, tail = head.item(), tail.item()
                    rels = rels.tolist()
                    head_rel = rels[0]
                    tail_ral = rels[len(rels) - 1]
                    ace_temp = {'after': 1277, '**after': 9399, 'arg': 1118, '**arg': 9242}
                    # ere_temp = {'after': 18, '**after': 37, 'arg': 17, '**arg': 36}
                    ere_temp = {'after': 612, '**after': 1222, 'arg': 611, '**arg': 1221}
                    if head_rel in [ere_temp['after'], ere_temp['**after']] and tail_ral in [ere_temp['after'],
                                                                                             ere_temp[
                                                                                                 '**after']]:  # 事件-事件
                        first_tri_index.append(index)
                        first_tri.append([head, tail])
                        sub_event_index.append(head)
                        obj_event_index.append(tail)
                    elif head_rel == ere_temp['arg'] and tail_ral == ere_temp['**arg']:  # 事件-事件
                        first_tri_index.append(index)
                        first_tri.append([head, tail])
                        sub_event_index.append(head)
                        obj_event_index.append(tail)
                    elif head_rel in [ere_temp['after'], ere_temp['**after'], ere_temp['arg']] and tail_ral == ere_temp[
                        'arg']:  # 事件-论元
                        second_tri_index.append(index)
                        second_tri.append([head, tail])
                        sub_event_index.append(head)
                    else:  # 实体-实体
                        third_tri_index.append(index)
                        third_tri.append([head, tail])
                        sub_entity_index.append(head)
                        obj_entity_index.append(tail)
                    rel_list.append(head_rel)  # 假设只放进去头关系

                    # 仍然是事件图谱的信息
                    init_ent_list = []
                    sub_index_list = []
                    obj_index_list = []
                    node_pos_dict = {}

                    for j, idx in enumerate(sub_entity_index):
                        if idx not in node_pos_dict:
                            init_ent_list.append(idx)
                            node_pos_dict.update({idx: len(init_ent_list) - 1})
                            sub_index_list.append(len(init_ent_list) - 1)
                        else:
                            sub_index_list.append(node_pos_dict[idx])

                    for j, idx in enumerate(obj_entity_index):
                        if idx not in node_pos_dict:
                            init_ent_list.append(idx)
                            node_pos_dict.update({idx: len(init_ent_list) - 1})
                            obj_index_list.append(len(init_ent_list) - 1)
                        else:
                            obj_index_list.append(node_pos_dict[idx])

                    # 得到了当前子图结构的一些信息，然后是怎么利用这些信息了，用图神经网络的形式
                    final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type = model_event_graph.sample_subgraph(
                        init_ent_list, 'train')
                    sub = torch.tensor(sub_entity_index).cuda()
                    rel = torch.tensor(rel_list).cuda()
                    event_related_args = ['test',fuzzy_tri,sub, rel, init_ent_list, sub_index_list, obj_index_list, node_pos_dict,
                                          final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index,
                                          new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type]

                #变成分类数据的标志
                classfy = (first_tri_index, first_tri, second_tri_index, second_tri, third_tri_index, third_tri)
                # test_batch_size = 1
                positive_ents = positive_ents.cuda()
                negative_ents = negative_ents.cuda()
                paths = paths.cuda()
                neg_size = neg_size.cuda()

                #应该是这里参数不够的问题,test的batch是1
                negative_score, neg_goal,rule_loss = model((positive_ents, negative_ents, paths),classfy,*event_related_args, batch_type=batch_type)
                neg_mse,neg_mae = neg_goal

                positive_score, pos_goal,rule_loss = model((positive_ents,paths),classfy,*event_related_args)#事件的参数也要加进去
                pos_mse,pos_mae  = pos_goal

                all_mse = all_mse + (neg_mse+pos_mse)

                all_mae = all_mae + (neg_mae + pos_mae)

                neg_first_score, neg_second_score, neg_third_score = negative_score
                pos_first_score, pos_second_score, pos_third_score = positive_score
                length = len(paths[0])
                if is_tensor(neg_first_score):
                    event_d = get_eval(neg_first_score, pos_first_score, neg_size, 'event')
                    event_logs.append(event_d)
                    event_logs_hop[length].append(event_d)
                if is_tensor(neg_second_score):
                    arg_d = get_eval(neg_second_score, pos_second_score, neg_size, 'arg')
                    arg_logs.append(arg_d)
                    arg_logs_hop[length].append(arg_d)

                if is_tensor(neg_third_score):
                    entity_d = get_eval(neg_third_score, pos_third_score, neg_size, 'entity')
                    entity_logs.append(entity_d)
                    entity_logs_hop[length].append(entity_d)

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... ({}/{})'.format(step, total_steps))

                step += 1

        event_metrics = {}
        arg_metrics = {}
        entity_metrics = {}
        mse = all_mse /(len(event_logs)+len(arg_logs)+len(entity_logs))
        mae = all_mae /(len(event_logs)+len(arg_logs)+len(entity_logs))
        print('mae: '+str(mae)+'\t'+'mse: '+str(mse)+'\n')
        for metric in event_logs[0].keys():
            event_metrics[metric] = sum([log[metric] for log in event_logs]) / len(event_logs)

        for metric in arg_logs[0].keys():
            arg_metrics[metric] = sum([log[metric] for log in arg_logs]) / len(arg_logs)

        for metric in entity_logs[0].keys():
            entity_metrics[metric] = sum([log[metric] for log in entity_logs]) / len(entity_logs)

        event_metrics_hop = defaultdict(dict)
        arg_metrics_hop = defaultdict(dict)
        entity_metrics_hop = defaultdict(dict)

        for hop in sorted(event_logs_hop.keys()):
            for metric in event_logs_hop[hop][0].keys():
                event_metrics_hop[hop][metric] = sum([log[metric] for log in event_logs_hop[hop]]) / len(event_logs_hop[hop])

        for hop in sorted(arg_logs_hop.keys()):
            for metric in arg_logs_hop[hop][0].keys():
                arg_metrics_hop[hop][metric] = sum([log[metric] for log in arg_logs_hop[hop]]) / len(arg_logs_hop[hop])

        for hop in sorted(entity_logs_hop.keys()):
            for metric in entity_logs_hop[hop][0].keys():
                entity_metrics_hop[hop][metric] = sum([log[metric] for log in entity_logs_hop[hop]]) / len(entity_logs_hop[hop])
        return event_metrics, event_metrics_hop,arg_metrics, arg_metrics_hop,entity_metrics, entity_metrics_hop

def is_tensor(var):
    return isinstance(var, torch.Tensor)
def get_eval(negative_score,positive_score,neg_size,mode):
    # neg_first_score, neg_second_score, neg_third_score = negative_score
    # pos_first_score, pos_second_score, pos_third_score = positive_score
    rank = (positive_score < negative_score).sum(dim=1)
    mq = 1 - rank.float() / neg_size.squeeze(1).float()
    rank= rank.item()
    d = {
        mode + '_MQ': mq.item(),
        'HITS@1': 1.0 if rank < 1 else 0.0,
        'HITS@3': 1.0 if rank < 3 else 0.0,
        'HITS@10': 1.0 if rank < 10 else 0.0,
    }
    return d
class DCNE(KGEModel):
    def __init__(self, num_entity, num_event,num_relation,rules, hidden_dim, gamma, p,*arglist):
        super().__init__()
        args_event_graph, model_event_graph, edge_index, edge_type, event_edge_index, event_index, role_type, role_mask, entity_event_index, entity_mask = arglist
        #事件相关信息
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.event_edge_index = event_edge_index
        self.event_index = event_index
        self.role_type = role_type
        self.role_mask = role_mask
        self.entity_event_index = entity_event_index
        self.entity_mask = entity_mask
        self.p = args_event_graph
        self.act = torch.tanh

        #原始信息
        # self.num_entity = 55331#ace暂定
        self.num_entity = 58236#52947#20492#ere暂定
        self.num_event = 40743#5301#18888#暂定,有问题的
        self.num_relation = num_relation
        self.time_size = 14763
        rank = 100
        self.rank = rank
        self.rank_static = rank // 20
        self.hidden_dim = hidden_dim
        self.rules_dim = 100
        self.epsilon = 2.0
        self.pc = p
        self.rule1_p1, self.rule1_p2, self.rule2_p1, self.rule2_p2, self.rule2_p3, self.rule2_p4 = rules
        # [0]实体 [1]关系 [2]时间 [3]关系 [4] 1 dim =200
        # self.event_embed = self.get_param_device(len(model_event_graph.evt2id), self.hidden_dim)

        # self.event_conv1 = EventConv(self.p.init_dim, self.p.gcn_dim, self.p.role_type_dim, self.p.role_num,
        #                              act=self.act, params=self.p)
        self.fc = nn.Linear(100,200)
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * self.rules_dim, sparse=False)#sparse = True
            for s in [self.num_entity, self.num_relation, self.time_size, self.num_relation, 1]  # last embedding modules contains no_time embeddings
        ])
        init_size: float = 1e-3
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size
        self.embeddings[4].weight.data *= init_size  # time transition
        # self.static_embeddings[0].weight.data *= init_size  # static entity embedding
        # self.static_embeddings[1].weight.data *= init_size  # static relation embedding

        # if self.p.num_bases > 0:
        #     self.init_rel = self.get_param((self.p.num_bases, self.p.init_dim))
        # else:
        #     if self.p.score_func == 'transe':
        #         self.init_rel = self.get_param((self.num_relation, self.p.init_dim))
        #     else:
        #         self.init_rel = self.get_param((self.num_relation * 2, self.p.init_dim))

        #事件图谱的参数和嵌入初始化
        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        # self.evt_type_emb = torch.nn.Parameter(self.event_type_embed[event_type_idxs])
        # self.evt_type_emb = self.event_type_embed[event_type_idxs]

        self.role_type_embed = self.get_param_device(self.p.role_num, self.p.role_type_dim)
        # self.event_embed = get_param_device(len(model_event_graph.evt2id), self.p.init_dim)
        self.event_embed = self.get_param_device(self.num_event, 100 * 2)#暂定
        self.init_embed = self.get_param_device(self.num_entity, self.p.init_dim)  # 这个应该是实体的初始化吧
        if self.p.num_bases > 0:
            self.init_rel = self.get_param((self.p.num_bases, self.p.init_dim))
        else:
            if self.p.score_func == 'transe':
                self.init_rel = self.get_param((self.num_relation, self.p.init_dim))
            else:
                self.init_rel = self.get_param((self.num_relation * 2, self.p.init_dim))
        self.event_conv1 = EventConv(self.p.init_dim, self.p.gcn_dim, self.p.role_type_dim, self.p.role_num,
                                     act=self.act, params=self.p)

        event_type_idxs = [0 for _ in range(len(model_event_graph.event2type))]
        for idx in model_event_graph.event2type:
            event_type_idxs[idx] = model_event_graph.event2type[idx]
        # event_type_idxs = torch.LongTensor(event_type_idxs).to(self.device)
        self.evt_type_emb = torch.nn.Parameter(model_event_graph.event_type_embed[event_type_idxs])

        self.mr1 = torch.nn.Embedding(num_embeddings=self.p.init_dim, embedding_dim=self.p.init_dim)
        self.mr2 = torch.nn.Embedding(num_embeddings=self.p.init_dim, embedding_dim=self.p.init_dim)
        self.liner = torch.nn.Linear(1, 1)
        if self.p.num_bases > 0:
            # gcn_dim和embed_dim都是200
            self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, self.num_relation, self.p.num_bases, act=self.act,
                                          params=self.p)
            self.conv2 = CompGCNConv(self.p.gcn_dim, self.p.embed_dim, self.num_relation, act=self.act,
                                     params=self.p) if self.p.gcn_layer == 2 else None
        else:
            self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim, self.num_relation, act=self.act, params=self.p)
            self.conv2 = CompGCNConv(self.p.gcn_dim, self.p.embed_dim, self.num_relation, act=self.act,
                                     params=self.p) if self.p.gcn_layer == 2 else None



        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )


        self.entity_embedding = nn.Parameter(torch.zeros(self.num_entity, 100 * 2))#暂定维度为100，和事件的嵌入同
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_e = nn.Parameter(torch.zeros(num_relation, hidden_dim))
        self.relation_ie = nn.Parameter(torch.zeros(num_relation, hidden_dim))

        self.pi = 3.14159262358979323846

    def layer(self, h, t):
        """Defines the forward pass layer of the algorithm.

          Args:
              h (Tensor): Head entities ids.
              t (Tensor): Tail entity ids of the triple.
        """
        mr1h = torch.matmul(h, self.mr1.weight)  # h => [m, d], self.mr1 => [d, k]
        mr2t = torch.matmul(t, self.mr2.weight)  # t => [m, d], self.mr2 => [d, k]
        return torch.tanh(mr1h + mr2t)

    def calculate_mse_loss(self,paths, sub_emb, rel_emb, third_tri_index, obj_embeed, w,syml,pl):
        if syml != 'train':
            third_tri_index = [0]
            w = w[0]
            path = paths[:,0]#path为多数的时候，会有这个区别
            rel_emb = rel_emb[path[third_tri_index]]
            rel_emb = torch.squeeze(rel_emb, dim=1)
        else:
            w = w[third_tri_index]
            if pl == 'yes':
                rel_emb = rel_emb[third_tri_index]
            else:
                path = paths[:, 0]#path为多数的时候，会有这个区别
                rel_emb = rel_emb[path[third_tri_index]]
                # rel_emb = rel_emb[paths[third_tri_index][0]]
                rel_emb = torch.squeeze(rel_emb, dim=1)

            # obj_id = actor2[third_tri_index]
        # try:
        #     w = w[third_tri_index]
        # except:
        #     print(123)
        ht = self.layer(sub_emb, obj_embeed)
        htr = torch.unsqueeze(torch.sum(rel_emb * (ht), dim=1), dim=-1)
        f_prob_h = self.liner(htr)  # 就是这个，记住
        f_prob_h = torch.squeeze(f_prob_h, dim=-1)
        f_score_h = torch.sum(torch.square(f_prob_h - w)) / len(third_tri_index)
        return f_score_h, f_prob_h

    def evaluate_mse(self, prediction, truth):
        # pred = prediction.detach().cpu()
        # truth1 = torch.FloatTensor([i[3] for i in truth])
        # truth_np = truth1.detach().cpu()

        # mse = (np.square(pred - truth_np)).mean()
        # mse1 = (torch.mul(pred, pred) - torch.mul(truth_np, truth_np)).mean()
        # mse2 = F.mse_loss(pred,truth_np).item()
        pred = prediction.detach().cpu().numpy()
        truth_np = truth.detach().cpu().numpy()
        mse = (np.square(pred - truth_np)).mean()
        mae = (np.absolute(pred - truth_np)).mean()
        return mse, mae

    def forward_base(self, syml,sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs,
                     evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list,
                     new_entity_type):
        # 在这里其实就变成200的嵌入了
        # [16278,768]，就是一个初始的嵌入
        r = self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
        self.p.use_event = 1  # 暂时不用事件看看
        if syml == 'test':
            x = self.init_embed[ent_neighbors]
            return x[0:len(final_ent_list)], r
        else:
            if self.p.use_event:
                x = self.event_conv1(new_event_index, self.init_embed, self.event_index, self.role_type, self.role_mask,
                                     new_entity_event_index, new_entity_mask, self.role_type_embed, self.event_embed,
                                     self.evt_type_emb, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors)
            else:
                # 把391个实体邻居给嵌入了
                x = self.init_embed[ent_neighbors]
        # [391,768],[16278,768]
        x, r = self.conv1(x, new_entity_list, new_entity_type, rel_embed=r)
        x = self.hidden_drop(x)

        return x[0:len(final_ent_list)], r

    def forward_for_kg_completion(self,syml, sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors,
                                  rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask,
                                  new_entity_list, new_entity_type):
        # print(rel)
        # print(self.p.ent_num)
        # print(self.p.event_num)
        # print(self.edge_index.shape)

        bs = sub.shape[0]
        # 最后取得就是entity_emb，但是relation_emb也会用到
        entity_emb, relation_emb = self.forward_base(syml,sub, rel, sub_index_list, init_ent_list, final_ent_list,
                                                     ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index,
                                                     new_entity_event_index, new_entity_mask, new_entity_list,
                                                     new_entity_type)
        # entity_emb = (final_num, dim)
        # print(entity_emb.shape)
        # print(relation_emb.shape)
        # sub_emb	= torch.index_select(entity_emb, 0, sub)
        # entity_emb应当直接包含final_ent_list中的每一个entity的embedding
        sub_emb = entity_emb[sub_index_list]
        rel_emb = torch.index_select(relation_emb, 0, rel)
        # 这几个向量确实是密切相关的

        return sub_emb, rel_emb, entity_emb

    def get_param(self,shape):
        param = Parameter(torch.Tensor(*shape));
        xavier_normal_(param.data)
        return param

    def get_param_device(self,x,y):
        param = Parameter(torch.Tensor(x, y)).cuda()  # 转换到gpu上
        # torch.nn.init.xavier_uniform_(param.data)
        xavier_normal_(param.data)  # 梯度爆炸，或者梯度消失了？
        return param
    def comp(self, path):#这里才是实现，静态方法可能只是一个抽取的方法，这里是继承和实现
        rels_emb = self.relation_embedding[path]#三维的矩阵#[batch,path,embed],双塔的第一个塔
        rels_e = self.relation_e[path]#[batch,path,embed]
        rels_ie = self.relation_ie[path]#这两个没有区别，双塔的另一个塔
        assert rels_emb.shape == torch.Size([path.shape[0], path.shape[1], self.relation_embedding.shape[1]])
        return rels_emb, rels_e, rels_ie

    def calc(self, head, rel, rel_e, rel_ie):#关系也要切片
        head_e, head_ie = torch.chunk(head, 2, dim=2)#分成两块

        phase_relation = rel / (self.embedding_range.item() / self.pi)

        relation_one = torch.cos(phase_relation)
        relation_two = torch.sin(phase_relation)
        relation_three = rel_e
        relation_four = rel_ie

        result_e = 2 * relation_one * relation_four + 2 * relation_two * relation_three + \
                   (relation_one ** 2 - relation_two ** 2) * head_e - 2 * relation_one * relation_two * head_ie
        result_ie = -2 * relation_one * relation_three + 2 * relation_two * relation_four + \
                    (relation_one ** 2 - relation_two ** 2) * head_ie + 2 * relation_one * relation_two * head_e

        return torch.cat([result_e, result_ie], dim=2)#进行了旋转？

    def get_rule_loss(self,path):
        # path = torch.tensor([int(i[0].item()) for i in path_temp]).cuda()
        rule = 0.
        rule_num = 1
        transt = self.embeddings[4](torch.LongTensor([0]).cuda())
        transt = transt[:, :self.rank], transt[:, self.rank:]

        for rel_1_content in path:#这里就是每个path都过一遍的意思
            for rel_1 in rel_1_content:
                rel_1_str = str(rel_1)
                if rel_1_str in self.rule1_p2:  # [0]实体 [1]关系 [2]时间 [3]关系 [4] 1, dim =200
                    rel1_emb = self.embeddings[3](rel_1)  # [460,200]460个关系
                    for rel_2 in self.rule1_p2[rel_1_str]:
                        weight_r = self.rule1_p2[rel_1_str][rel_2]  # 数值
                        rel2_emb = self.embeddings[3](torch.LongTensor([int(rel_2)]).cuda())[0]
                        rule += weight_r * torch.sum(torch.abs(rel1_emb - rel2_emb) ** 3)  # ？？
                        rule_num += 1

        for rel_1_content in path:
            for rel_1 in rel_1_content:
                rel_1_str = str(rel_1)
                if rel_1_str in self.rule1_p2:
                    rel1_emb = self.embeddings[3](rel_1)
                    rel1_split = rel1_emb[:self.rank], rel1_emb[self.rank:]  # 复数空间？
                    for rel_2 in self.rule1_p2[rel_1_str]:
                        weight_r = self.rule1_p2[rel_1_str][rel_2]
                        rel2_emb = self.embeddings[3](torch.LongTensor([int(rel_2)]).cuda())[0]
                        rel2_split = rel2_emb[:self.rank], rel2_emb[self.rank:]
                        tt = rel2_split[0] * transt[0][0], rel2_split[1] * transt[0][0], rel2_split[0] * transt[1][0], \
                             rel2_split[1] * transt[1][0]
                        rtt = tt[0] - tt[3], tt[1] + tt[2]
                        # print("rel1_split:\t", rel1_split[0])
                        rule += weight_r * (torch.sum(torch.abs(rel1_split[0] - rtt[0]) ** 3) + torch.sum(
                            torch.abs(rel1_split[1] - rtt[1]) ** 3))
                        rule_num += 1

        for rel_1_content in path:
            for rel_1 in rel_1_content:
                if rel_1 in self.rule2_p1:  # ？？
                    rel1_emb = self.embeddings[3](rel_1)
                    rel1_split = rel1_emb[:self.rank], rel1_emb[self.rank:]
                    for body in self.rule2_p1[rel_1]:
                        rel_2, rel_3 = body
                        weight_r = self.rule2_p1[rel_1][body]
                        rel2_emb = self.embeddings[3](torch.LongTensor([rel_2]).cuda())[0]
                        rel3_emb = self.embeddings[3](torch.LongTensor([rel_3]).cuda())[0]
                        rel2_split = rel2_emb[:self.rank], rel2_emb[self.rank:]
                        rel3_split = rel3_emb[:self.rank], rel3_emb[self.rank:]
                        tt2 = rel2_split[0] * transt[0][0], rel2_split[1] * transt[0][0], rel2_split[0] * transt[1][0], \
                              rel2_split[1] * transt[1][0]
                        rtt2 = tt2[0] - tt2[3], tt2[1] + tt2[2]
                        ttt2 = rtt2[0] * transt[0][0], rtt2[1] * transt[0][0], rtt2[0] * transt[1][0], rtt2[1] * transt[1][
                            0]
                        rttt2 = ttt2[0] - ttt2[3], ttt2[1] + ttt2[2]
                        tt3 = rel3_split[0] * transt[0][0], rel3_split[1] * transt[0][0], rel3_split[0] * transt[1][0], \
                              rel3_split[1] * transt[1][0]
                        rtt3 = tt3[0] - tt3[3], tt3[1] + tt3[2]
                        tt = rtt3[0] * rttt2[0], rtt3[1] * rttt2[0], rtt3[0] * rttt2[1], rtt3[1] * rttt2[1]
                        rtt = tt[0] - tt[3], tt[1] + tt[2]
                        # print("rel1_split:\t", rel1_split[0])
                        rule += weight_r * (torch.sum(torch.abs(rel1_split[0] - rtt[0]) ** 3) + torch.sum(
                            torch.abs(rel1_split[1] - rtt[1]) ** 3))
                        rule_num += 1

        for rel_1_content in path:
            for rel_1 in rel_1_content:
                if rel_1 in self.rule2_p2:  # 不同的rule
                    rel1_emb = self.embeddings[3](rel_1)
                    rel1_split = rel1_emb[:self.rank], rel1_emb[self.rank:]
                    for body in self.rule2_p2[rel_1]:
                        rel_2, rel_3 = body
                        weight_r = self.rule2_p2[rel_1][body]
                        rel2_emb = self.embeddings[3](torch.LongTensor([rel_2]).cuda())[0]
                        rel3_emb = self.embeddings[3](torch.LongTensor([rel_3]).cuda())[0]
                        rel2_split = rel2_emb[:self.rank], rel2_emb[self.rank:]
                        rel3_split = rel3_emb[:self.rank], rel3_emb[self.rank:]
                        tt2 = rel2_split[0] * transt[0][0], rel2_split[1] * transt[0][0], rel2_split[0] * transt[1][0], \
                              rel2_split[1] * transt[1][0]
                        rtt2 = tt2[0] - tt2[3], tt2[1] + tt2[2]
                        tt3 = rel3_split[0] * transt[0][0], rel3_split[1] * transt[0][0], rel3_split[0] * transt[1][0], \
                              rel3_split[1] * transt[1][0]
                        rtt3 = tt3[0] - tt3[3], tt3[1] + tt3[2]
                        tt = rtt3[0] * rtt2[0], rtt3[1] * rtt2[0], rtt3[0] * rtt2[1], rtt3[1] * rtt2[1]
                        rtt = tt[0] - tt[3], tt[1] + tt[2]
                        # print("rel1_split:\t", rel1_split[0])
                        rule += weight_r * (torch.sum(torch.abs(rel1_split[0] - rtt[0]) ** 3) + torch.sum(
                            torch.abs(rel1_split[1] - rtt[1]) ** 3))
                        rule_num += 1

        for rel_1_content in path:
            for rel_1 in rel_1_content:
                if rel_1 in self.rule2_p3:
                    rel1_emb = self.embeddings[3](rel_1)
                    rel1_split = rel1_emb[:self.rank], rel1_emb[self.rank:]
                    for body in self.rule2_p3[rel_1]:
                        rel_2, rel_3 = body
                        weight_r = self.rule2_p3[rel_1][body]
                        rel2_emb = self.embeddings[3](torch.LongTensor([rel_2]).cuda())[0]
                        rel3_emb = self.embeddings[3](torch.LongTensor([rel_3]).cuda())[0]
                        rel2_split = rel2_emb[:self.rank], rel2_emb[self.rank:]
                        rtt3 = rel3_emb[:self.rank], rel3_emb[self.rank:]
                        tt2 = rel2_split[0] * transt[0][0], rel2_split[1] * transt[0][0], rel2_split[0] * transt[1][0], \
                              rel2_split[1] * transt[1][0]
                        rtt2 = tt2[0] - tt2[3], tt2[1] + tt2[2]
                        tt = rtt3[0] * rtt2[0], rtt3[1] * rtt2[0], rtt3[0] * rtt2[1], rtt3[1] * rtt2[1]
                        rtt = tt[0] - tt[3], tt[1] + tt[2]
                        # print("rel1_split:\t", rel1_split[0])
                        rule += weight_r * (torch.sum(torch.abs(rel1_split[0] - rtt[0]) ** 3) + torch.sum(
                            torch.abs(rel1_split[1] - rtt[1]) ** 3))
                        rule_num += 1

        for rel_1_content in path:
            for rel_1 in rel_1_content:
                if rel_1 in self.rule2_p4:
                    rel1_emb = self.embeddings[3](rel_1)
                    rel1_split = rel1_emb[:self.rank], rel1_emb[self.rank:]
                    for body in self.rule2_p4[rel_1]:
                        rel_2, rel_3 = body
                        weight_r = self.rule2_p4[rel_1][body]
                        rel2_emb = self.embeddings[3](torch.LongTensor([rel_2]).cuda())[0]
                        rel3_emb = self.embeddings[3](torch.LongTensor([rel_3]).cuda())[0]
                        rtt2 = rel2_emb[:self.rank], rel2_emb[self.rank:]
                        rtt3 = rel3_emb[:self.rank], rel3_emb[self.rank:]
                        tt = rtt3[0] * rtt2[0], rtt3[1] * rtt2[0], rtt3[0] * rtt2[1], rtt3[1] * rtt2[1]
                        rtt = tt[0] - tt[3], tt[1] + tt[2]
                        # print("rel1_split:\t", rel1_split[0])
                        rule += weight_r * (torch.sum(torch.abs(rel1_split[0] - rtt[0]) ** 3) + torch.sum(
                            torch.abs(rel1_split[1] - rtt[1]) ** 3))
                        rule_num += 1

        rule = rule / rule_num  # rule变成了偏差因子，# @实现矩阵乘法 .t()转置
        return rule

    def func(self, tri_head, tri_tail,rels,batch_type):
        rel_emb_path, rels_e_path, rels_ie_path = rels#三个tuple
        rel_embs = torch.chunk(rel_emb_path, rel_emb_path.shape[1], dim=1)#按照path的长度n，分成n份
        rels_e_s = torch.chunk(rels_e_path, rels_e_path.shape[1], dim=1)
        rels_ie_s = torch.chunk(rels_ie_path, rels_ie_path.shape[1], dim=1)

        for rel, rel_e, rel_ie in zip(rel_embs, rels_e_s, rels_ie_s):
            new_head = self.calc(tri_head, rel, rel_e, rel_ie)#head进行迭代计算n次
            tri_head = new_head
        #第三种
        head_e, head_ie = torch.chunk(tri_head, 2, dim=2)#分成两块
        tail_e, tail_ie = torch.chunk(tri_tail, 2, dim=2)

        score_e = head_e - tail_e#transE?
        score_ie = head_ie - tail_ie

        score = torch.stack([score_e, score_ie], dim=0)#[2,batch,1,dim]
        score = score.norm(dim=0, p=self.pc)#？[batch,1,dim]
        score = self.gamma.item() - score.sum(dim=2)##gamma是一个数，类似隔板？

        return score
