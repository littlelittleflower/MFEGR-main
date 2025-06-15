# Name: util
# Author: Reacubeth
# Time: 2021/6/25 17:08
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import os
import numpy as np
import torch
import argparse
from helper import *
# from ordered_set import OrderedSet
import json


class Runner(object):
    def __init__(self, params):
        """
        Constructor of the runner class

        Parameters
        ----------
        params:         List of hyperparameters of the model

        Returns
        -------
        Creates computational graph and optimizer

        """
        self.p = params
        # self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
        #
        # self.logger.info(vars(self.p))
        # pprint(vars(self.p))

        # if self.p.gpu != '-1' and torch.cuda.is_available():
        # 	self.device = torch.device('cuda:'+self.p.gpu)
        # 	torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        # 	torch.backends.cudnn.deterministic = True
        # else:
        # 	self.device = torch.device('cpu')

        # if self.p.gpu != '-1' and torch.cuda.is_available():
        #     self.device = int(self.p.gpu)
        #     torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        #     torch.backends.cudnn.deterministic = True
        # else:
        #     self.device = torch.device('cpu')
        # print('123')
        # self.load_data()
        # self.model = self.add_model(self.p.model, self.p.score_func)
        # for param in self.model[:2].parameters():
        #     param.requires_grad = False
        # for p in self.model.init_embed.data():
        # self.model[0].requires_grad = False
        # for name, param in self.model.named_parameters():
        #     if 'init_embed' in name:
        #         param.requires_grad = False
        # self.optimizer = self.add_optimizer(self.model.parameters())

    def load_events(self):
        ent2id_dir= self.p.dataset + "/entity_ids.json"
        event_type_dir =  self.p.dataset + "/event_types.json"
        event_ids_dir = self.p.dataset + "/event_ids.json"
        event_entity_dir =  self.p.dataset + "/event2ent.json"
        relation_ids_dir =  self.p.dataset + "/relation_ids.json"

        self.role2id = {}
        self.event_edges = {}  # {event_idx: [[ent_idx1, role_idx1], [ent_idx2, role_idx2], ...]}

        with open(ent2id_dir, "r", encoding="utf-8") as f:
            self.ent2id = json.load(f)

        with open(relation_ids_dir, "r", encoding="utf-8") as f:
            self.rel2id = json.load(f)

        with open(event_type_dir, "r", encoding="utf-8") as f:
            self.event_types = json.load(f)

        with open(event_ids_dir, "r", encoding="utf-8") as f:
            self.evt2id = json.load(f)

        with open(event_entity_dir, "r", encoding="utf-8") as f:
            evt2ent = json.load(f)

        self.evt2ent = evt2ent

        self.event_type_idxs = {}

        event_type_num = 0
        for event_id in self.event_types:
            if self.event_types[event_id] not in self.event_type_idxs:
                self.event_type_idxs.update({self.event_types[event_id]: event_type_num})
                event_type_num += 1

        event_type_num = len(self.event_type_idxs)
        self.event_type_embed = torch.randn(event_type_num, self.p.role_type_dim).cuda()

        self.event2type = {}

        for event_id in self.evt2id:
            # print(event_id)
            evt_idx = self.evt2id[event_id]
            # print(evt_idx)
            self.event2type.update({evt_idx: self.event_type_idxs[self.event_types[event_id]]})

        # print(evt2ent)
        # read event2ent into role ids
        role_num = 0

        for event_id in evt2ent:
            args = evt2ent[event_id]["args"]
            for arg in args:
                role_type = args[arg]
                if role_type not in self.role2id:
                    self.role2id.update({role_type: role_num})
                    role_num += 1

        for event_id in evt2ent:
            args = evt2ent[event_id]["args"]
            event_idx = self.evt2id[event_id]

            for arg in args:
                role_type_idx = self.role2id[args[arg]]
                ent_idx = self.ent2id[arg]  # 这里其实就有对id的正确映射，说明论元也成功映射了

                if event_idx not in self.event_edges:  # 这个应该是事件和论元之间的连接
                    self.event_edges.update({event_idx: [[ent_idx, role_type_idx]]})
                else:
                    self.event_edges[event_idx].append([ent_idx, role_type_idx].copy())

        role_num = len(self.role2id)
        self.p.role_num = role_num
        self.p.event_num = len(self.evt2id)
        max_role_num = max([len(evt2ent[key]["args"]) for key in evt2ent])
        self.p.max_role_num = max_role_num

        # print(self.event_edges)
        evt2ent_list = [[] for _ in range(self.p.event_num)]
        for evt_id in self.event_edges:
            for ent_id in self.event_edges[evt_id]:
                evt2ent_list[evt_id].append(ent_id[0])
        self.evt2ent_list = evt2ent_list

        self.id2evt = {idx: evt for evt, idx in self.evt2id.items()}

        self.edge_index, self.edge_type = self.construct_adj()
        # print(self.edge_type.shape)
        # print(self.edge_index.shape)
        self.event_edge_index, self.event_index, self.role_type, self.role_mask, self.entity_event_index, self.entity_mask = self.construct_event_adj()

        return self.edge_index,self.edge_type,self.event_edge_index, self.event_index, self.role_type, self.role_mask, self.entity_event_index, self.entity_mask
    # print(evt2ent_list)
    # print(evt2ent_list)
    # print(self.event_edges)
    # print(self.role2id)

    def load_data(self):
        """
		Reading in raw triples and converts it into a standard format.

		Parameters
		----------
		self.p.dataset:         Takes in the name of the dataset (FB15k-237)

		Returns
		-------
		self.ent2id:            Entity to unique identifier mapping
		self.id2rel:            Inverse mapping of self.ent2id
		self.rel2id:            Relation to unique identifier mapping
		self.num_ent:           Number of entities in the Knowledge graph
		self.num_rel:           Number of relations in the Knowledge graph
		self.embed_dim:         Embedding dimension used
		self.data['train']:     Stores the triples corresponding to training dataset
		self.data['valid']:     Stores the triples corresponding to validation dataset
		self.data['test']:      Stores the triples corresponding to test dataset
		self.data_iter:		The dataloader for different data splits

		"""

        # ent_set, rel_set = OrderedSet(), OrderedSet()
        # for split in ['train', 'test', 'valid']:
        #     for line in open('./data/{}/{}.txt'.format(self.p.dataset, split), encoding='utf-8'):
        #         sub, rel, obj = line.strip().split('\t')
        #         ent_set.add(sub)
        #         rel_set.add(rel)
        #         ent_set.add(obj)

        self.p.entity_embed_dir = "./data/" + self.p.dataset + "/entity_embed.npy"
        # print(self.p.entity_embed_dir)
        entity_ids_dir = "./data/" + self.p.dataset + "/entity_ids.json"
        entity_types_dir = "./data/" + self.p.dataset + "/entity_types.json"
        relation_ids_dir = "./data/" + self.p.dataset + "/relation_ids.json"

        with open(entity_ids_dir, "r", encoding="utf-8") as f:
            self.ent2id = json.loads(f.readline())

        with open(entity_types_dir, "r", encoding="utf-8") as f:
            entity_types = json.loads(f.readline())

        self.p.num_ent = len(self.ent2id)
        self.p.ent_num = len(self.ent2id)
        # load entity types to type_id:
        self.ent_type_to_id = {}
        entity_type_num = 0

        for ent_id in entity_types:
            if entity_types[ent_id] not in self.ent_type_to_id:
                self.ent_type_to_id.update({entity_types[ent_id]: entity_type_num})
                entity_type_num += 1

        self.id_to_ent_type = {idx: t for t, idx in self.ent_type_to_id.items()}
        self.p.entity_type_num = len(self.ent_type_to_id)

        train_ent_labels, test_ent_labels, valid_ent_labels = [], [], []
        train_ent_ids, test_ent_ids, valid_ent_ids = [], [], []

        train_num = int(self.p.ent_num * 0.8)
        valid_num = int(self.p.ent_num * 0.1) + 1
        test_num = self.p.ent_num - train_num - valid_num

        # print(train_num)

        for ent_id in self.ent2id:
            ent_idx = self.ent2id[ent_id]
            if ent_idx >= 0 and ent_idx < train_num:
                train_ent_ids.append(ent_idx)
                try:
                    train_ent_labels.append(self.ent_type_to_id[entity_types[str(self.ent2id[ent_id])]])
                except:
                    train_ent_labels.append(0)
            elif ent_idx >= train_num and ent_idx < train_num + valid_num:
                valid_ent_ids.append(ent_idx)
                try:
                    valid_ent_labels.append(self.ent_type_to_id[entity_types[str(self.ent2id[ent_id])]])
                except:
                    valid_ent_labels.append(0)
            # valid_ent_labels.append(self.ent_type_to_id[entity_types[str(self.ent2id[ent_id])]])
            else:
                test_ent_ids.append(ent_idx)
                try:
                    test_ent_ids.append(self.ent_type_to_id[entity_types[str(self.ent2id[ent_id])]])
                except:
                    test_ent_ids.append(0)
            # test_ent_labels.append(self.ent_type_to_id[entity_types[str(self.ent2id[ent_id])]])

        self.train_ent_labels = torch.LongTensor(train_ent_labels).cuda()
        self.train_ent_ids = torch.LongTensor(train_ent_ids).cuda()

        self.valid_ent_labels = torch.LongTensor(valid_ent_labels).cuda()
        self.valid_ent_ids = torch.LongTensor(valid_ent_ids).cuda()

        self.test_ent_labels = torch.LongTensor(test_ent_labels).cuda()
        self.test_ent_ids = torch.LongTensor(test_ent_ids).cuda()

        with open(relation_ids_dir, "r", encoding="utf-8") as f:
            self.rel2id = json.loads(f.readline())

        # self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        # self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}

        rel2id_dict = self.rel2id.copy()
        for key in rel2id_dict:
            self.rel2id.update({key + '_reverse': rel2id_dict[key] + len(rel2id_dict)})

        # print(self.rel2id)
        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        ''' 
		ent2id: {entity_id: 0}
		rel2id: {rel_id: 0, rel_id_reverse: 1}
		'''

        # also load relation classification dataset
        train_rel_labels, train_rel_start, train_rel_end = [], [], []
        valid_rel_labels, valid_rel_start, valid_rel_end = [], [], []
        test_rel_labels, test_rel_start, test_rel_end = [], [], []

        self.p.num_rel = len(self.rel2id) // 2

        for line in open('./data/{}/train.txt'.format(self.p.dataset), encoding='utf-8'):
            sub, rel, obj = line.strip().split('\t')
            try:
                start_idx = self.ent2id[sub]
                train_rel_start.append(start_idx)
                end_idx = self.ent2id[obj]
                train_rel_end.append(end_idx)
                rel_idx = self.rel2id[rel]
                train_rel_labels.append(rel_idx)
            except:
                print(sub, rel, obj)

        self.train_rel_labels = torch.LongTensor([train_rel_labels]).cuda().t()
        self.train_rel_start = torch.LongTensor([train_rel_start]).cuda().t()
        self.train_rel_end = torch.LongTensor([train_rel_end]).cuda().t()

        for line in open('./data/{}/valid.txt'.format(self.p.dataset), encoding='utf-8'):
            sub, rel, obj = line.strip().split('\t')
            start_idx = self.ent2id[sub]
            valid_rel_start.append(start_idx)
            end_idx = self.ent2id[obj]
            valid_rel_end.append(end_idx)
            rel_idx = self.rel2id[rel]
            valid_rel_labels.append(rel_idx)

        self.valid_rel_labels = torch.LongTensor([valid_rel_labels]).cuda().t()
        self.valid_rel_start = torch.LongTensor([valid_rel_start]).cuda().t()
        self.valid_rel_end = torch.LongTensor([valid_rel_end]).cuda().t()

        for line in open('./data/{}/test.txt'.format(self.p.dataset), encoding='utf-8'):
            sub, rel, obj = line.strip().split('\t')
            start_idx = self.ent2id[sub]
            test_rel_start.append(start_idx)
            end_idx = self.ent2id[obj]
            test_rel_end.append(end_idx)
            rel_idx = self.rel2id[rel]
            test_rel_labels.append(rel_idx)

        self.test_rel_labels = torch.LongTensor([test_rel_labels]).cuda().t()
        self.test_rel_start = torch.LongTensor([test_rel_start]).cuda().t()
        self.test_rel_end = torch.LongTensor([test_rel_end]).cuda().t()

        # 这里的num_rel的含义是relation type的数量 而不是relation edge的数量，其实edge的数量就等于triples的数量
        # num_rel在这里指的是：没有添加reverse边的时候的数量
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/fuzzy_{}.txt'.format(self.p.dataset, split), encoding='utf-8'):
                sub, rel, obj, sub_ambiguity, rel_ambiguity, obj_ambiguity, ambiguity = line.strip().split('\t')
                sub_ambiguity, rel_ambiguity, obj_ambiguity, ambiguity = float(sub_ambiguity), float(
                    rel_ambiguity), float(obj_ambiguity), float(ambiguity)
                try:
                    sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                except:
                    print(sub, rel, obj)
                    continue
                self.data[split].append((sub, rel, obj, sub_ambiguity, rel_ambiguity, obj_ambiguity, ambiguity))

                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + self.p.num_rel)].add(sub)
        # sr2o的格式：dict, {(h, r): Set({t1, t2, ...})}
        # for key in sr2o:
        # 	if len(sr2o[key]) > 1:
        # 		print(key)
        # 		print(sr2o[key])
        # 		print('\n')
        # print(sr2o)
        self.data = dict(self.data)
        # print(self.data)
        # self.data: {"train": [(h, r, t), (h, r, t)], "valid": [(h, r, t), (h, r, t)]}
        self.sr2o = {k: list(v) for k, v in sr2o.items()}  # only contains training sr2os
        # self.sr2o的格式：dict, {(h, r): [t1, t2, ...])}
        for split in ['test', 'valid']:
            for sub, rel, obj, sub_ambiguity, rel_ambiguity, obj_ambiguity, ambiguity in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}  # contains training, validataion, and testing sr2os
        # self.sr2o_all的格式：dict, {(h, r): [t1, t2, ...])}
        # print(type(self.sr2o))
        # print(type(self.sr2o[(6593,9)]))
        self.triples = ddict(list)
        # 这个sr2o不就是有点过滤的意思
        for sro_samp, data_samp in zip(self.sr2o.items(), self.data['train']):  # train里面的数据
            (sub, rel), obj = sro_samp
            _, _, _, sub_ambiguity, rel_ambiguity, obj_ambiguity, ambiguity = data_samp
            self.triples['train'].append(  # 就是这里，obj[0]可能有多个，因为数据的问题，可能会有脏的数据，然后保存在label里面
                {'triple': (sub, rel, -1), 'obj': [obj[0]], 'label': self.sr2o[(sub, rel)], 'sub_samp': 1,
                 'ambiguitys': (sub_ambiguity, rel_ambiguity, obj_ambiguity, ambiguity)})

        # self.triples['train']: {'triple': (h, r, -1), 'label': [t1, t2, t3,...], 'sub_samp': 1}
        for sub, rel, obj, sub_ambiguity, rel_ambiguity, obj_ambiguity, ambiguity in self.data['valid']:
            self.triples['valid_final'].append(
                {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)],
                 'ambiguitys': (sub_ambiguity, rel_ambiguity, obj_ambiguity, ambiguity)})
        # 	self.triples['valid_final'].append(
        # 		{'triple': (sub, rel, obj), 'label': ambiguity})
        len_valid = len(self.data['valid'])

        # 从一开始对同一个三元组就生成了两个三元组，主要表现就是关系变的不一样了，另一个关系变成了关系加上关系的数目
        # 然后评价的时候也是分成了两个部分进行评价

        for split in ['test', 'valid']:
            for sub, rel, obj, sub_ambiguity, rel_ambiguity, obj_ambiguity, ambiguity in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)],
                     'ambiguitys': (sub_ambiguity, rel_ambiguity, obj_ambiguity, ambiguity)})
                self.triples['{}_{}'.format(split, 'head')].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)],
                     'ambiguitys': (sub_ambiguity, rel_ambiguity, obj_ambiguity, ambiguity)})

        # 注意：self.triples["train"] 里面只包含了训练集的所有triples，"triple"的最后一个值是-1，但是"valid", "test"里面的label包含了所有的可能的labels，(包括训练集和测试集)

        self.triples = dict(self.triples)

    def construct_adj(self):
        """
        Constructor of the runner class

        Parameters
        ----------

        Returns
        -------
        Constructs the adjacency matrix for GCN

        """
        # edge_index: [] -- a list of tuples: [(subj1, obj1), (subj2, obj2), ...]
        # edge_type: [] -- a list of edge_type_idxs: [r1,r2,...]
        edge_index, edge_type = [], []
        # self.p.num_rel = len(self.rel2id) // 2
        # self.p.num_ent = len(self.ent2id)
        # self.p.ent_num = len(self.ent2id)

        self.p.num_rel = 1223 // 2
        self.p.num_ent = 58236
        self.p.ent_num = 58236

        # self.data[split]: list of (h,r,t)

        self.data = ddict(list) #需要导入三元组的数据

        for split in ['train', 'test', 'valid']:
            for line in open('{}/fuzzy_{}.txt'.format(self.p.dataset, split), encoding='utf-8'):
                sub, rel, obj, sub_ambiguity, rel_ambiguity, obj_ambiguity, ambiguity = line.strip().split('\t')
                sub_ambiguity, rel_ambiguity, obj_ambiguity, ambiguity = float(sub_ambiguity), float(
                    rel_ambiguity), float(obj_ambiguity), float(ambiguity)
                try:
                    sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                except:
                    print(sub, rel, obj)
                    continue
                self.data[split].append((sub, rel, obj, sub_ambiguity, rel_ambiguity, obj_ambiguity, ambiguity))

        for sub, rel, obj, _, _, _, _ in self.data['train']:#还是有很多其他的数据
            edge_index.append((sub, obj))
            edge_type.append(rel)#加入了rel？？？

        # Adding inverse edges
        for sub, rel, obj, _, _, _, _ in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)
        # print(self.device)
        edge_list = [[] for _ in range(self.p.ent_num)]
        for start, end in edge_index:
            edge_list[end].append(start)
        # print(edge_index)
        self.edge_list = edge_list
        # print(edge_list)
        # edge_index	= torch.LongTensor(edge_index).to(self.device).t()
        edge_index = torch.LongTensor(edge_index).cuda().t()#转为tensor的话就是看看情况了
        edge_type = torch.LongTensor(edge_type).cuda()

        # edge_index: torch.LongTensor(2, 2*edge_num)
        # edge_type: torch.LongTensor(2*edge_num)
        # print(edge_type.shape)
        # print(edge_index.shape)
        # construct a tensor for all 1-hop previous neighbors

        return edge_index, edge_type

    def construct_event_adj(self):
        event_index = torch.zeros(self.p.event_num, self.p.max_role_num, dtype=torch.long).cuda()
        role_type = torch.zeros(self.p.event_num, self.p.max_role_num, dtype=torch.long).cuda()
        role_mask = torch.zeros(self.p.event_num, self.p.max_role_num).cuda()

        # check the maximum number of events
        ent2evt = {}
        for evt_idx in self.event_edges:
            args = self.event_edges[evt_idx]
            for arg in args:
                if arg[0] not in ent2evt:
                    ent2evt.update({arg[0]: [evt_idx]})
                else:
                    ent2evt[arg[0]].append(evt_idx)

        self.ent2evt = ent2evt
        # print(self.ent2evt)
        # 改写ent2evt成为list形式
        ent2evt_list = [[] for _ in range(self.p.ent_num)]
        for ent_id in ent2evt:
            for evt_id in ent2evt[ent_id]:
                ent2evt_list[ent_id].append(evt_id)
        self.ent2evt_list = ent2evt_list
        # print(ent2evt_list)

        # max_evt_num = max([len(ent2evt[key]) for key in ent2evt])

        # entity_event_index = torch.zeros(max_evt_num, self.p.ent_num, dtype=torch.long).to(self.device)
        # role_mask = torch.zeros(max_evt_num, self.p.ent_num).to(self.device)

        entity_event_index = [[] for _ in range(self.p.ent_num)]
        entity_mask = [[] for _ in range(self.p.ent_num)]

        for event_idx in self.event_edges:
            end_ents = self.event_edges[event_idx]
            for i, arg in enumerate(end_ents):
                event_index[event_idx][i] = arg[0]
                role_type[event_idx][i] = arg[1]
                role_mask[event_idx][i] = 1.0

                entity_event_index[arg[0]].append(event_idx)
                entity_mask[arg[0]].append(1.0)

        event_index = event_index[:, 0:self.p.entity_sample_num]
        role_type = role_type[:, 0:self.p.entity_sample_num]
        role_mask = role_mask[:, 0:self.p.entity_sample_num]
        # embed to tensor
        entity_event_index_new = entity_event_index.copy()
        entity_mask_new = entity_mask.copy()

        max_event_num = max([len(i) for i in entity_event_index])

        for i, l in enumerate(entity_event_index_new):
            entity_event_index[i].extend([0] * (max_event_num - len(l)))

        for i, l in enumerate(entity_mask_new):
            entity_mask[i].extend([0.0] * (max_event_num - len(l)))
        entity_event_index = torch.LongTensor(entity_event_index)[:, 0:self.p.event_sample_num].cuda()
        entity_mask = torch.Tensor(entity_mask)[:, 0:self.p.event_sample_num].cuda()
        # event_neigh_list = [[] for _ in range(self.p.event_num)]
        with open(self.p.dataset + "/temp.json", "r", encoding="utf-8") as f:
            event_edge_index = json.loads(f.read())
        self.event_edge_list = event_edge_index
        # print(event_edge_index)
        event_neigh_list = [[] for _ in range(self.p.event_num)]

        for i in range(len(event_edge_index[0])):
            event_neigh_list[event_edge_index[1][i]].append(event_edge_index[0][i])

        self.event_neigh_list = event_neigh_list

        event_edge_index = torch.LongTensor(event_edge_index).cuda()

        return event_edge_index, event_index, role_type, role_mask, entity_event_index, entity_mask

    def sample_subgraph(self, init_ent_list,syml):
        '''
        这里需要的都是事件图谱的信息
        包括：
        self.ent2evt_list
        self.evt2ent_list
        self.edge_list
        self.event_neigh_list
        self.p.event_num
        self.p.ent_num
        self.device

        self.edge_index
        self.edge_type


        :param init_ent_list:
        :return:
        '''
        # print(1)
        rel_evt_idxs = []
        for ent_idx in init_ent_list:
            try:
                rel_evt_idxs = rel_evt_idxs + self.ent2evt_list[ent_idx]
            except:
                print(ent_idx,init_ent_list)
                # print(123)
        rel_evt_idxs = list(set(rel_evt_idxs))
        # if self.p.dataset == "kairos":
        #     rel_evt_idxs = rel_evt_idxs[0:5000]
        event_ent_idxs = []

        for evt_idx in rel_evt_idxs:
            event_ent_idxs += self.evt2ent_list[evt_idx]

        final_ent_list = init_ent_list.copy()
        # if self.p.dataset == "kairos":
        #     pass
        # else:
        #这里为什么会有问题
        # if syml == 'valid':
        #     print('187')
        #     pass
            # count1 = 0
            # for idx in event_ent_idxs:
            #     # print(count1)
            #     # count1 += 1
            #     if idx not in final_ent_list:
            #         final_ent_list.append(idx)
        # else:
        for idx in event_ent_idxs:
            if idx not in final_ent_list:
                final_ent_list.append(idx)
        # print('end')
        ent_neighbors, evt_neighbors = final_ent_list.copy(), rel_evt_idxs.copy()

        # if self.p.dataset == "kairos":
        #     for ent_idx in final_ent_list:
        #         idxs = self.edge_list[ent_idx]
        #         for idx in idxs:
        #             if idx not in ent_neighbors:
        #                 ent_neighbors.append(idx)
        #
        #     ent_neighbors = ent_neighbors[0:10000]

        # else:
        # ent_neighbors, evt_neighbors
        for ent_idx in final_ent_list:
            idxs = self.edge_list[ent_idx]
            for idx in idxs:
                if idx not in ent_neighbors:
                    ent_neighbors.append(idx)

        for evt_idx in rel_evt_idxs:
            idxs = self.event_neigh_list[evt_idx]
            for idx in idxs:
                if idx not in evt_neighbors:
                    evt_neighbors.append(idx)

        for ent_id in ent_neighbors:
            idxs = self.ent2evt_list[ent_id]
            for idx in idxs:
                if idx not in evt_neighbors:
                    evt_neighbors.append(idx)

        for evt_id in evt_neighbors:
            idxs = self.evt2ent_list[evt_id]
            for idx in idxs:
                if idx not in ent_neighbors:
                    ent_neighbors.append(idx)

        # assert (rel_evt_idxs == evt_neighbors[0:len(rel_evt_idxs)])
        # assert (final_ent_list == ent_neighbors[0:len(final_ent_list)])
        # print(4)

        evt_total_to_selected = [-1 for _ in range(self.p.event_num)]
        for j, evt_id in enumerate(evt_neighbors):
            evt_total_to_selected[evt_id] = j
        # print(5)

        entity_mask_list = self.entity_mask[ent_neighbors].tolist()
        entity_event_index_list = self.entity_event_index[ent_neighbors].tolist()

        new_entity_event_index_list = []
        for i in range(len(entity_event_index_list)):
            events_i = []
            for j in range(len(entity_event_index_list[0])):
                evt_mask = entity_mask_list[i][j]
                evt_idx = entity_event_index_list[i][j]
                if evt_mask != 0.0:
                    if evt_total_to_selected[evt_idx] == -1:
                        events_i.append(0)
                        entity_mask_list[i][j] = 0.0
                    else:
                        events_i.append(evt_total_to_selected[evt_idx])
                else:
                    events_i.append(0)

            new_entity_event_index_list.append(events_i)
        # print(6)

        new_entity_event_index = torch.LongTensor(new_entity_event_index_list).cuda()
        new_entity_mask = torch.FloatTensor(entity_mask_list).cuda()

        # if self.p.dataset == "kairos":
        #
        #     new_event_list = [[], []]
        #     for i in range(len(self.event_edge_list[0])):
        #         start = self.event_edge_list[0][i]
        #         end = self.event_edge_list[1][i]
        #         if end in rel_evt_idxs and start in rel_evt_idxs:
        #             new_event_list[0].append(evt_total_to_selected[start])
        #             new_event_list[1].append(evt_total_to_selected[end])
        # else:
        new_event_list = [[], []]
        for i in range(len(self.event_edge_list[0])):
            start = self.event_edge_list[0][i]
            end = self.event_edge_list[1][i]
            if end in rel_evt_idxs:
                new_event_list[0].append(evt_total_to_selected[start])
                new_event_list[1].append(evt_total_to_selected[end])
        # print(7)

        new_event_index = torch.LongTensor(new_event_list).cuda()

        # calculate the entities
        ent_total_to_selected = [-1 for _ in range(self.p.ent_num)]
        for j, ent_id in enumerate(ent_neighbors):
            ent_total_to_selected[ent_id] = j
        # print(8)

        # calculate the new entity edges
        new_entity_list = [[], []]
        new_entity_type = []

        edge_index_list = self.edge_index.tolist()
        edge_type_list = self.edge_type.tolist()

        # if self.p.dataset == "kairos":
        #
        #     for i in range(len(edge_index_list[0])):
        #         start = edge_index_list[0][i]
        #         end = edge_index_list[1][i]
        #         if end in final_ent_list and start in ent_neighbors:
        #             new_entity_list[0].append(ent_total_to_selected[start])
        #             new_entity_list[1].append(ent_total_to_selected[end])
        #             new_entity_type.append(edge_type_list[i])
        # else:

        for i in range(len(edge_index_list[0])):
            start = edge_index_list[0][i]
            end = edge_index_list[1][i]
            if end in final_ent_list:
                new_entity_list[0].append(ent_total_to_selected[start])
                new_entity_list[1].append(ent_total_to_selected[end])
                new_entity_type.append(edge_type_list[i])
        # print(9)

        new_entity_list = torch.LongTensor(new_entity_list).cuda()
        new_entity_type = torch.LongTensor(new_entity_type).cuda()

        return final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type

# def load_events():
#     pass

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1]), int(line_split[2])


def load_quadruples(inPath, fileName, fileName2=None, fileName3=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        quadruplefloatList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            fuzzy_low = float(line_split[4])
            fuzzy_high = float(line_split[5])
            quadrupleList.append([head, rel, tail, time])
            quadruplefloatList.append([fuzzy_low,fuzzy_high])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                fuzzy_low = float(line_split[4])
                fuzzy_high = float(line_split[5])
                quadrupleList.append([head, rel, tail, time])
                quadruplefloatList.append([fuzzy_low, fuzzy_high])
                times.add(time)

    if fileName3 is not None:
        with open(os.path.join(inPath, fileName3), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                fuzzy_low = float(line_split[4])
                fuzzy_high = float(line_split[5])
                quadrupleList.append([head, rel, tail, time])
                quadruplefloatList.append([fuzzy_low, fuzzy_high])
                times.add(time)
    times = list(times)
    times.sort()
    lt =  np.asarray(quadrupleList)
    return np.asarray(quadrupleList), np.asarray(quadruplefloatList),np.asarray(times)


def make_batch(a, b, c, d, e, f, g,h, batch_size, valid1=None, valid2=None):
    if valid1 is None and valid2 is None:
        for i in range(0, len(a), batch_size):
            yield [a[i:i + batch_size], b[i:i + batch_size], c[i:i + batch_size],
                   d[i:i + batch_size], e[i:i + batch_size], f[i:i + batch_size], g[i:i + batch_size],h[i:i + batch_size]]
    else:
        for i in range(0, len(a), batch_size):
            yield [a[i:i + batch_size], b[i:i + batch_size], c[i:i + batch_size],
                   d[i:i + batch_size], e[i:i + batch_size], f[i:i + batch_size], g[i:i + batch_size],h[i:i + batch_size],
                   valid1[i:i + batch_size], valid2[i:i + batch_size]]


def to_device(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor.cpu()


def isListEmpty(inList):
    if isinstance(inList, list):
        return all(map(isListEmpty, inList))
    return False


def get_sorted_s_r_embed_limit(s_hist, s, r, ent_embeds, limit):
    s_hist_len = to_device(torch.LongTensor(list(map(len, s_hist))))
    s_len, s_idx = s_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(s_len))
    s_len_non_zero = s_len[:num_non_zero]
    s_len_non_zero = torch.where(s_len_non_zero > limit, to_device(torch.tensor(limit)), s_len_non_zero)

    s_hist_sorted = []
    for idx in s_idx[:num_non_zero]:
        s_hist_sorted.append(s_hist[idx.item()])

    flat_s = []
    len_s = []

    for hist in s_hist_sorted:
        for neighs in hist[-limit:]:
            len_s.append(len(neighs))
            for neigh in neighs:
                flat_s.append(neigh[1])
    s_tem = s[s_idx]
    r_tem = r[s_idx]

    embeds = ent_embeds[to_device(torch.LongTensor(flat_s))]
    embeds_split = torch.split(embeds, len_s)
    return s_idx, s_len_non_zero, s_tem, r_tem, embeds, len_s, embeds_split


def get_sorted_s_r_embed(s_hist, s, r, ent_embeds):
    s_hist_len = to_device(torch.LongTensor(list(map(len, s_hist))))
    s_len, s_idx = s_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(s_len))
    s_len_non_zero = s_len[:num_non_zero]

    s_hist_sorted = []
    for idx in s_idx[:num_non_zero]:
        s_hist_sorted.append(s_hist[idx.item()])

    flat_s = []
    len_s = []

    for hist in s_hist_sorted:
        for neighs in hist:
            len_s.append(len(neighs))
            for neigh in neighs:
                flat_s.append(neigh[1])
    s_tem = s[s_idx]
    r_tem = r[s_idx]

    embeds = ent_embeds[to_device(torch.LongTensor(flat_s))]
    embeds_split = torch.split(embeds, len_s)
    """
    s_idx: id of descending by length in original list.  1 * batch
    s_len_non_zero: number of events having history  any
    s_tem: sorted s by length  batch
    r_tem: sorted r by length  batch
    embeds: event->history->neighbor
    lens_s: event->history_neighbor length
    embeds_split split by history neighbor length
    s_hist_dt_sorted: history interval sorted by history length without non
    """
    return s_idx, s_len_non_zero, s_tem, r_tem, embeds, len_s, embeds_split


def str2bool(v: str) -> bool:
    v = v.lower()
    if v == "true":
        return True
    elif v == "false":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected, got" + str(v) + ".")


def write2file(mae,mse,s_ranks, o_ranks, all_ranks,s_ranks2_event_entity, o_ranks2_event_entity, all_ranks2_event_entity,s_ranks2_event_event, o_ranks2_event_event, all_ranks2_event_event, file_test):
    #这个是第一次的情况
    s_ranks = np.asarray(s_ranks)
    s_mr_lk = np.mean(s_ranks)
    s_mrr_lk = np.mean(1.0 / s_ranks)

    #这个是第二次的情况
    s_ranks_event_entity = np.asarray(s_ranks2_event_entity)
    s_mr_lk_event_entity = np.mean(s_ranks_event_entity)
    s_mrr_lk_event_entity = np.mean(1.0 / s_ranks_event_entity)

    #这个是最后一次的情况
    s_ranks_event_event = np.asarray(s_ranks2_event_event)
    s_mr_lk_event_event = np.mean(s_ranks_event_event)
    s_mrr_lk_event_event = np.mean(1.0 / s_ranks_event_event)

    print("MAE: {:.6f}".format(mae))
    print("MSE: {:.6f}".format(mse))

    print("Entity-Entity Subject test MRR (lk): {:.6f}".format(s_mrr_lk))
    print("Entity-Entity Subject test MR (lk): {:.6f}".format(s_mr_lk))

    print("Event-Entity Subject test MRR (lk): {:.6f}".format(s_mrr_lk_event_entity))
    print("Event-Entity Subject test MR (lk): {:.6f}".format(s_mr_lk_event_entity))

    print("Event-Event Subject test MRR (lk): {:.6f}".format(s_mrr_lk_event_event))
    print("Event-Event Subject test MR (lk): {:.6f}".format(s_mr_lk_event_event))

    file_test.write("Entity-Entity Subject test MRR (lk): {:.6f}".format(s_mrr_lk) + '\n')
    file_test.write("Entity-Entity Subject test MR (lk): {:.6f}".format(s_mr_lk) + '\n')

    file_test.write("Event-Entity Subject test MRR (lk): {:.6f}".format(s_mrr_lk_event_entity) + '\n')
    file_test.write("Event-Entity Subject test MR (lk): {:.6f}".format(s_mr_lk_event_entity) + '\n')

    file_test.write("Event-Event Subject test MRR (lk): {:.6f}".format(s_mrr_lk_event_event) + '\n')
    file_test.write("Event-Event Subject test MR (lk): {:.6f}".format(s_mr_lk_event_event) + '\n')

    # print("MAE: {:.6f}".format(mae))
    # print("MSE: {:.6f}".format(mse))

    file_test.write("MAE: {:.6f}".format(mae) + '\n')
    file_test.write("MSE: {:.6f}".format(mse) + '\n')

    for hit in [1, 3, 10]:
        avg_count_sub_lk = np.mean((s_ranks <= hit))
        print("Entity-Entity Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk))
        file_test.write("Entity-Entity Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk) + '\n')
    #然后是第二次
    for hit in [1, 3, 10]:
        avg_count_sub_lk_event_entity = np.mean((s_ranks_event_entity  <= hit))
        print("Event-Entity Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk_event_entity ))
        file_test.write("Event-Entity Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk_event_entity ) + '\n')
    #最后是第三次
    for hit in [1, 3, 10]:
        avg_count_sub_lk_event_event = np.mean((s_ranks_event_event <= hit))
        print("Event-Event Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk_event_event))
        file_test.write("Event-Event Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk_event_event) + '\n')

    o_ranks = np.asarray(o_ranks)
    o_mr_lk = np.mean(o_ranks)
    o_mrr_lk = np.mean(1.0 / o_ranks)

    o_ranks_event_entity = np.asarray(o_ranks2_event_entity)
    o_mr_lk_event_entity = np.mean(o_ranks_event_entity)
    o_mrr_lk_event_entity = np.mean(1.0 / o_ranks_event_entity)

    o_ranks_event_event = np.asarray(o_ranks2_event_event)
    o_mr_lk_event_event  = np.mean(o_ranks_event_event)
    o_mrr_lk_event_event  = np.mean(1.0 / o_ranks_event_event )

    print("Entity-Entity Object test MRR (lk): {:.6f}".format(o_mrr_lk))
    print("Entity-Entity Object test MR (lk): {:.6f}".format(o_mr_lk))
    file_test.write("Entity-Entity Object test MRR (lk): {:.6f}".format(o_mrr_lk) + '\n')
    file_test.write("Entity-Entity Object test MR (lk): {:.6f}".format(o_mr_lk) + '\n')

    print("Event-Entity Object test MRR (lk): {:.6f}".format(o_mrr_lk_event_entity))
    print("Event-Entity Object test MR (lk): {:.6f}".format(o_mr_lk_event_entity))
    file_test.write("Event-Entity Object test MRR (lk): {:.6f}".format(o_mrr_lk_event_entity) + '\n')
    file_test.write("Event-Entity Object test MR (lk): {:.6f}".format(o_mr_lk_event_entity) + '\n')

    print("Event-Event Object test MRR (lk): {:.6f}".format(o_mrr_lk_event_event))
    print("Event-Event Object test MR (lk): {:.6f}".format(o_mr_lk_event_event))
    file_test.write("Event-Event Object test MRR (lk): {:.6f}".format(o_mrr_lk_event_event) + '\n')
    file_test.write("Event-Event Object test MR (lk): {:.6f}".format(o_mr_lk_event_event) + '\n')

    for hit in [1, 3, 10]:
        avg_count_obj_lk = np.mean((o_ranks <= hit))
        print("Entity-Entity Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk))
        file_test.write("Entity-Entity Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk) + '\n')

    for hit in [1, 3, 10]:
        avg_count_obj_lk_event_entity = np.mean((o_ranks_event_entity <= hit))
        print("Event-Entity Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk_event_entity))
        file_test.write("Event-Entity Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk_event_entity) + '\n')

    for hit in [1, 3, 10]:
        avg_count_obj_lk_event_event = np.mean((o_ranks_event_event <= hit))
        print("Event-Event Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk_event_event))
        file_test.write("Event-Event Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk_event_event) + '\n')

    all_ranks = np.asarray(all_ranks)
    all_mr_lk = np.mean(all_ranks)
    all_mrr_lk = np.mean(1.0 / all_ranks)

    all_ranks_event_entity = np.asarray(all_ranks2_event_entity)
    all_mr_lk_event_entity = np.mean(all_ranks_event_entity)
    all_mrr_lk_event_entity = np.mean(1.0 / all_ranks_event_entity)

    all_ranks_event_event = np.asarray(all_ranks2_event_event)
    all_mr_lk_event_event = np.mean(all_ranks_event_event)
    all_mrr_lk_event_event = np.mean(1.0 / all_ranks_event_event)

    print("Entity-Entity ALL test MRR (lk): {:.6f}".format(all_mrr_lk))
    print("Entity-Entity ALL test MR (lk): {:.6f}".format(all_mr_lk))
    file_test.write("Entity-Entity ALL test MRR (lk): {:.6f}".format(all_mrr_lk) + '\n')
    file_test.write("Entity-Entity ALL test MR (lk): {:.6f}".format(all_mr_lk) + '\n')

    print("Event-Entity ALL test MRR (lk): {:.6f}".format(all_mrr_lk_event_entity))
    print("Event-Entity ALL test MR (lk): {:.6f}".format(all_mr_lk_event_entity))
    file_test.write("Event-Entity ALL test MRR (lk): {:.6f}".format(all_mrr_lk_event_entity) + '\n')
    file_test.write("Event-Entity ALL test MR (lk): {:.6f}".format(all_mr_lk_event_entity) + '\n')

    print("Event-Event ALL test MRR (lk): {:.6f}".format(all_mrr_lk_event_event))
    print("Event-Event ALL test MR (lk): {:.6f}".format(all_mr_lk_event_event))
    file_test.write("Event-Event ALL test MRR (lk): {:.6f}".format(all_mrr_lk_event_event) + '\n')
    file_test.write("Event-Event ALL test MR (lk): {:.6f}".format(all_mr_lk_event_event) + '\n')

    for hit in [1, 3, 10]:
        avg_count_all_lk = np.mean((all_ranks <= hit))
        print("Entity-Entity ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk))
        file_test.write("Entity-Entity ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk) + '\n')

    for hit in [1, 3, 10]:
        avg_count_all_lk_event_entity = np.mean((all_ranks_event_entity <= hit))
        print("Event-Entity ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk_event_entity))
        file_test.write("Event-Entity ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk_event_entity) + '\n')

    for hit in [1, 3, 10]:
        avg_count_all_lk_event_event = np.mean((all_ranks_event_event <= hit))
        print("Event-Event ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk_event_event))
        file_test.write("Event-Event ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk_event_event) + '\n')
    return all_mrr_lk,all_mrr_lk_event_entity,all_mrr_lk_event_event
