import torch
import numpy as np
import sys
import os
from helper import *
from model.compgcn_conv import CompGCNConv, EventConv
from model.compgcn_conv_basis import CompGCNConvBasis
import torch.nn.functional as F

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5, torch.complex128: 1e-5}
# torch.autograd.set_detect_anomaly(True)

class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p = params
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()#BCELoss，BCEWithLogitsLoss

    def loss(self, pred, true_label):
        # assert (pred > torch.tensor(0.0) & pred < torch.tensor(1.0)).all()
        return self.bceloss(pred, true_label)


class CompGCNBase(BaseModel):
    def __init__(self, event_edge_index, edge_index, edge_type, event_index, role_type, role_mask, entity_event_index,
                 entity_mask, num_rel, params=None):
        super(CompGCNBase, self).__init__(params)
        if params.gpu != "-1":
            self.device = int(params.gpu)
        else:
            self.device = torch.device("cpu")
        self.init_embed = get_param_device(self.p.num_ent, self.p.init_dim, self.device)
        for p in self.parameters():#冻结参数没有用
            p.requires_grad = False
        self.event_edge_index = event_edge_index

        self.edge_index = edge_index
        self.edge_type = edge_type

        self.event_index = event_index
        self.role_type = role_type
        self.role_mask = role_mask
        self.entity_event_index = entity_event_index
        self.entity_mask = entity_mask

        self.p.gcn_dim = self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim



        # entity_typing_parameter
        self.linear_1 = torch.nn.Linear(self.p.embed_dim, self.p.embed_dim)
        self.linear_2 = torch.nn.Linear(self.p.embed_dim, self.p.entity_type_num)

        self.rel_linear_1 = torch.nn.Linear(2 * self.p.embed_dim, self.p.embed_dim)
        self.rel_linear_2 = torch.nn.Linear(self.p.embed_dim, self.p.num_rel)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)

        # ent_init_embed = torch.Tensor(np.load(params.entity_embed_dir))
        # self.init_embed = torch.nn.Parameter(ent_init_embed)



        # if params.gpu != "-1":
        # 	self.device		= torch.device("cuda:"+params.gpu)
        # else:
        # 	self.device		= torch.device("cpu")

        if self.p.num_bases > 0:
            self.init_rel = get_param((self.p.num_bases, self.p.init_dim))
        else:
            if self.p.score_func == 'transe':
                self.init_rel = get_param((num_rel, self.p.init_dim))
            else:
                self.init_rel = get_param((num_rel * 2, self.p.init_dim))

        if self.p.num_bases > 0:
            # gcn_dim和embed_dim都是200
            self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act,
                                          params=self.p)
            self.conv2 = CompGCNConv(self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act,
                                     params=self.p) if self.p.gcn_layer == 2 else None
        else:
            self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim, num_rel, act=self.act, params=self.p)
            self.conv2 = CompGCNConv(self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act,
                                     params=self.p) if self.p.gcn_layer == 2 else None
        # num_rel： 没有添加reverse边的数量
        # 两层GCN layer: init_dim --> gcn_dim --> embed_dim

        self.event_conv1 = EventConv(self.p.init_dim, self.p.gcn_dim, self.p.role_type_dim, self.p.role_num,
                                     act=self.act, params=self.p)
        self.event_conv2 = EventConv(self.p.gcn_dim, self.p.embed_dim, self.p.role_type_dim, self.p.role_num,
                                     act=self.act, params=self.p) if self.p.gcn_layer == 2 else None

        # self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

        self.load_events(params)

    def load_events(self, param):
        event_embed_dir = "./data/" + param.dataset + "/event_embed.npy"
        event_type_dir = "./data/" + param.dataset + "/event_types.json"
        event_ids_dir = "./data/" + param.dataset + "/event_ids.json"

        with open(event_type_dir, "r", encoding="utf-8") as f:
            self.event_types = json.loads(f.readline())

        with open(event_ids_dir, "r", encoding="utf-8") as f:
            self.evt2id = json.loads(f.readline())

        self.event_type_idxs = {}

        event_type_num = 0
        for event_id in self.event_types:
            if self.event_types[event_id] not in self.event_type_idxs:
                self.event_type_idxs.update({self.event_types[event_id]: event_type_num})
                event_type_num += 1

        event_type_num = len(self.event_type_idxs)
        # self.event_type_embed = get_param_device(event_type_num, param.role_type_dim, self.device)
        self.event_type_embed = torch.randn(event_type_num, param.role_type_dim).to(self.device)

        self.event2type = {}

        for event_id in self.evt2id:
            # print(event_id)
            evt_idx = self.evt2id[event_id]
            # print(evt_idx)
            self.event2type.update({evt_idx: self.event_type_idxs[self.event_types[event_id]]})
        # print(self.event_types)

        # evt_init_embed = torch.Tensor(np.load(event_embed_dir))
        # self.event_embed = torch.nn.Parameter(evt_init_embed)

        self.event_embed = get_param_device(len(self.evt2id), self.p.init_dim, self.device)

        # load event type_idx
        event_type_idxs = [0 for _ in range(len(self.event2type))]
        for idx in self.event2type:
            event_type_idxs[idx] = self.event2type[idx]
        # event_type_idxs = torch.LongTensor(event_type_idxs).to(self.device)
        self.evt_type_emb = torch.nn.Parameter(self.event_type_embed[event_type_idxs])
        # self.evt_type_emb = self.event_type_embed[event_type_idxs]

        self.role_type_embed = get_param_device(param.role_num, param.role_type_dim, self.device)

        # we have:
        # self.evt_embed, self.evt2id, self.event_types

        '''
			event information for implementing forward_base:
				1) self.event_index: (max_role_num, event_num): containing event idxs and entity idxs
				2) self.role_type: (max_role_num, event_num): containing role type idxs between each event and entity pair
				3) self.role_mask: (max_role_num, event_num): masked values
				4) self.role_type_embed: (role_type_num, role_type_dim): containing role_type_embeddings

				5) self.event_embed: (event_num, init_dim): containing trigger embeddings for each event
				6) self.event2type: dict: {event_idx: event_type_idx}
				7) self.event_type_embed: (event_type_num, init_dim): containing event_type embeddings for each type
		'''

    # print(self.evt2id)

    def forward_base(self, sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs,
                     evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list,
                     new_entity_type):
        # 在这里其实就变成200的嵌入了
        # [16278,768]，就是一个初始的嵌入
        r = self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
        self.p.use_event = 1 #暂时不用事件看看
        if self.p.use_event:
            x = self.event_conv1(new_event_index, self.init_embed, self.event_index, self.role_type, self.role_mask,
                                 new_entity_event_index, new_entity_mask, self.role_type_embed, self.event_embed,
                                 self.evt_type_emb, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors)
        else:
            # 把391个实体邻居给嵌入了
            x = self.init_embed[ent_neighbors]
        # [391,768],[16278,768]
        x, r = self.conv1(x, new_entity_list, new_entity_type, rel_embed=r)#这里有问题，有傅里叶变换
        x = self.hidden_drop(x)

        return x[0:len(final_ent_list)], r

    def predict_kg_base(self, sub, rel):
        r = self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)

        final_ent_list = [_ for _ in range(self.p.ent_num)]
        ent_neighbors = [_ for _ in range(self.p.ent_num)]
        rel_evt_idxs = [_ for _ in range(self.p.event_num)]
        evt_neighbors = [_ for _ in range(self.p.event_num)]

        if self.p.use_event:
            x = self.event_conv1(self.event_edge_index, self.init_embed, self.event_index, self.role_type,
                                 self.role_mask, self.entity_event_index, self.entity_mask, self.role_type_embed,
                                 self.event_embed, self.evt_type_emb, final_ent_list, ent_neighbors, rel_evt_idxs,
                                 evt_neighbors)
        else:
            x = self.init_embed
        # relation convolution layer 1
        x, r = self.conv1(x, self.edge_index, self.edge_type, rel_embed=r)
        x = self.hidden_drop(x)
        return x, r

    def forward_for_kg_completion(self, sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors,
                                  rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask,
                                  new_entity_list, new_entity_type):
        # print(rel)
        # print(self.p.ent_num)
        # print(self.p.event_num)
        # print(self.edge_index.shape)

        bs = sub.shape[0]
        # 最后取得就是entity_emb，但是relation_emb也会用到
        entity_emb, relation_emb = self.forward_base(sub, rel, sub_index_list, init_ent_list, final_ent_list,
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

    def predict_for_kg_completion(self, sub, rel):
        entity_emb, relation_emb = self.predict_kg_base(sub, rel)
        # print(relation_emb.shape)
        # print(rel.tolist())
        sub_emb = torch.index_select(entity_emb, 0, sub)
        rel_emb = torch.index_select(relation_emb, 0, rel)

        return sub_emb, rel_emb, entity_emb

    def forward_for_entity_typing(self, input_ent_id, input_labels, batch_ent_list, final_ent_list, ent_neighbors,
                                  rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask,
                                  new_entity_list, new_entity_type, predict=False):
        # entities: (batch_size)
        # labels: (batch_size)
        ent_id = input_ent_id.squeeze(1)
        labels = input_labels.squeeze(1)

        # sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type

        entities = \
        self.forward_base(None, None, batch_ent_list, batch_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs,
                          evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list,
                          new_entity_type)[0][0: len(batch_ent_list)]

        if not predict:
            logits = self.linear_2(torch.nn.Dropout(0.2)(torch.nn.ReLU()(self.linear_1(entities))))
            soft_logits = torch.softmax(logits, 1)
            loss = torch.nn.CrossEntropyLoss()(soft_logits, labels)
            return loss

        else:
            with torch.no_grad():
                logits = self.linear_2(torch.nn.Dropout(0.2)(torch.nn.ReLU()(self.linear_1(entities))))
                soft_logits = torch.softmax(logits, 1)
                eval_loss = torch.nn.CrossEntropyLoss()(soft_logits, labels)
                pred_labels = torch.argmax(soft_logits, 1)
                return eval_loss, pred_labels

    def forward_for_relation_typing(self, start_id, end_id, label, start_index_list, end_index_list, final_ent_list,
                                    ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index,
                                    new_entity_mask, new_entity_list, new_entity_type, predict=False):
        # start_ent_id: (batch_size, 1)
        # end_ent_id: (batch_size, 1)
        # labels: (batch_size, 1)
        start_ent_id = start_id.squeeze(1)
        end_ent_id = end_id.squeeze(1)
        labels = label.squeeze(1)

        # final_ent_list = [ _ for _ in range(self.p.ent_num)]
        # ent_neighbors = [ _ for _ in range(self.p.ent_num)]
        # rel_evt_idxs = [ _ for _ in range(self.p.event_num)]
        # evt_neighbors = [ _ for _ in range(self.p.event_num)]

        entity_emb = \
        self.forward_base(None, None, None, None, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors,
                          new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type)[0]

        if not predict:
            start_entities = entity_emb[start_index_list]
            end_entities = entity_emb[end_index_list]

            # start_entities = entity_emb[start_ent_id]
            # end_entities = entity_emb[end_ent_id]

            entities = torch.cat((start_entities, end_entities), 1)
            logits = self.rel_linear_2(torch.nn.Dropout(0.2)(torch.nn.ReLU()(self.rel_linear_1(entities))))
            soft_logits = torch.softmax(logits, 1)
            loss = torch.nn.CrossEntropyLoss()(soft_logits, labels)

            return loss

        else:
            with torch.no_grad():
                start_entities = entity_emb[start_index_list]
                end_entities = entity_emb[end_index_list]

                # start_entities = entity_emb[start_ent_id]
                # end_entities = entity_emb[end_ent_id]

                entities = torch.cat((start_entities, end_entities), 1)

                logits = self.rel_linear_2(torch.nn.Dropout(0.2)(torch.nn.ReLU()(self.rel_linear_1(entities))))
                soft_logits = torch.softmax(logits, 1)
                eval_loss = torch.nn.CrossEntropyLoss()(soft_logits, labels)
                pred_labels = torch.argmax(soft_logits, 1)
                # print(pred_labels)
                # print(label.squeeze(1))
                return eval_loss, pred_labels


class CompGCN_TransE(CompGCNBase):
    def __init__(self, event_edge_index, edge_index, edge_type, event_index, role_type, role_mask, entity_event_index,
                 entity_mask, params=None):
        super(self.__class__, self).__init__(event_edge_index, edge_index, edge_type, event_index, role_type, role_mask,
                                             entity_event_index, entity_mask, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs,
                evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list,
                new_entity_type, ambiguity):
        sub_emb, rel_emb, all_ent = self.forward_for_kg_completion(sub, rel, sub_index_list, init_ent_list,
                                                                   final_ent_list, ent_neighbors, rel_evt_idxs,
                                                                   evt_neighbors, new_event_index,
                                                                   new_entity_event_index, new_entity_mask,
                                                                   new_entity_list, new_entity_type)
        obj_emb = sub_emb + rel_emb

        x = self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        score = torch.sigmoid(x)

        return score


class CompGCN_DistMult(CompGCNBase):
    def __init__(self, event_edge_index, edge_index, edge_type, event_index, role_type, role_mask, entity_event_index,
                 entity_mask, params=None):
        super(self.__class__, self).__init__(event_edge_index, edge_index, edge_type, event_index, role_type, role_mask,
                                             entity_event_index, entity_mask, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs,
                evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list,
                new_entity_type):
        sub_emb, rel_emb, all_ent = self.forward_for_kg_completion(sub, rel, sub_index_list, init_ent_list,
                                                                   final_ent_list, ent_neighbors, rel_evt_idxs,
                                                                   evt_neighbors, new_event_index,
                                                                   new_entity_event_index, new_entity_mask,
                                                                   new_entity_list, new_entity_type)
        obj_emb = sub_emb * rel_emb

        x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score


class CompGCN_ConvE(CompGCNBase):
    def __init__(self, event_edge_index, edge_index, edge_type, event_index, role_type, role_mask, entity_event_index,
                 entity_mask, params=None):
        # edge_index: longtensor, (2, 2*rel_num) contains start nodes and end nodes (subj, obj)
        # edge_type: longtensor, (1, 1*rel_num) contains edge types (including inverse edges)
        super(self.__class__, self).__init__(event_edge_index, edge_index, edge_type, event_index, role_type, role_mask,
                                             entity_event_index, entity_mask, params.num_rel, params)
        self.c = torch.nn.Embedding(self.p.num_ent, 1)#双曲空间的曲率c，和关系的数目有关,怪怪的
        # self.entity = torch.nn.Embedding(self.p.num_ent, self.p.dim)
        # self.rel = torch.nn.Embedding(self.p.num_ent, self.p.dim)

        torch.nn.init.xavier_uniform_(self.c.weight)
        # self.dim = 128

        self.ent_embeddings_real = torch.nn.Embedding(num_embeddings=self.p.num_ent,
                                                      embedding_dim=self.p.dim)
        self.ent_embeddings_img = torch.nn.Embedding(num_embeddings=self.p.num_ent,
                                                     embedding_dim=self.p.dim)
        self.rel_embeddings_real = torch.nn.Embedding(num_embeddings=self.p.num_ent,
                                                      embedding_dim=self.p.dim)
        self.rel_embeddings_img = torch.nn.Embedding(num_embeddings=self.p.num_ent,
                                                     embedding_dim=self.p.dim)
        torch.nn.init.xavier_uniform_(self.ent_embeddings_real.weight)
        torch.nn.init.xavier_uniform_(self.ent_embeddings_img.weight)
        torch.nn.init.xavier_uniform_(self.rel_embeddings_real.weight)
        torch.nn.init.xavier_uniform_(self.rel_embeddings_img.weight)

        self.liner = torch.nn.Linear(1, 1).to(self.device)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz),
                                       stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)

    def project(self, x, c):
        """Project points to unit ball with curvature c.

        Args:
            x: re_x + im_x * 1j, torch.Tensor of size B * d with complex hyperbolic points
            c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

        Returns:
            torch.Tensor with projected complex hyperbolic points.
        """
        norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
        eps = 1e-5
        maxnorm = (1 - eps) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def tanh(self, x):
        return x.clamp(-15, 15).tanh()

    def expmap0(self, u, c):
        """Exponential map taken at the origin of the Poincare ball with curvature c.

        Args:
        u: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures
        Returns:
        torch.Tensor with tangent points.
        """

        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
        gamma_1 = self.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)#gamma_1是复数？
        return self.project(gamma_1, c)#变成了复双曲球？

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.p.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def layer(self, h, t):
        """Defines the forward pass layer of the algorithm.

          Args:
              h (Tensor): Head entities ids.
              t (Tensor): Tail entity ids of the triple.
        """
        mr1h = torch.matmul(h, self.mr1.weight)  # h => [m, d], self.mr1 => [d, k]
        mr2t = torch.matmul(t, self.mr2.weight)  # t => [m, d], self.mr2 => [d, k]
        return torch.tanh(mr1h + mr2t)

    def embed(self, h, r, t):
        rel = r.unsqueeze(1).long()
        sub = h.unsqueeze(1).long()
        tail = t.unsqueeze(1).long()
        c = torch.nn.functional.softplus(self.c(rel))
        head_real = self.ent_embeddings_real(sub)
        head_img = self.ent_embeddings_img(sub)
        rel_real = self.rel_embeddings_real(rel)
        rel_img = self.rel_embeddings_img(rel)
        tail_real = self.ent_embeddings_real(tail)
        tail_img = self.ent_embeddings_img(tail)
        # 这里当然是正常的[64,400]
        # 为了构造复数，分成了两边
        # rank = head[..., :self.p.dim//2]
        # t1 = head[..., self.p.dim//2:]
        # 可能是这个torch就是不行。。
        # head = head[..., :self.p.dim//2] + 1j * head[..., self.p.dim//2:]
        # head = torch.irfft(head, signal_ndim=1,normalized=True)  # 这里又做了傅里叶
        h_emb_real = self.expmap0(head_real, c).squeeze(1)   # hyperbolic
        h_emb_img = self.expmap0(head_img, c).squeeze(1)   # hyperbolic
        r_emb_real = self.expmap0(rel_real, c).squeeze(1)  # hyperbolic
        r_emb_img = self.expmap0(rel_img, c).squeeze(1)  # hyperbolic
        t_emb_real = self.expmap0(tail_real, c).squeeze(1)  # hyperbolic
        t_emb_img = self.expmap0(tail_img, c).squeeze(1)  # hyperbolic

        # 这就是实现了双曲？！
        # h_emb_real = self.ent_embeddings_real(h)
        # h_emb_img = self.ent_embeddings_img(h)
        # r_emb_real = self.rel_embeddings_real(r)
        # r_emb_img = self.rel_embeddings_img(r)
        #
        # t_emb_real = self.ent_embeddings_real(t)
        # t_emb_img = self.ent_embeddings_img(t)

        return h_emb_real, h_emb_img, r_emb_real, r_emb_img, t_emb_real, t_emb_img

    def forward(self, split, sub, rel, obj, neg_hn_batch, neg_rel_hn_batch, neg_t_batch, neg_h_batch, neg_rel_tn_batch,
                neg_tn_batch, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors,
                new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type, ambiguity):
        # 进入的是这个forward函数,这里我们就可以用上sub, rel,obj这三个部分
        # 但是输入的是一个batch的所有relation和entity，也就是说这个图并不是完整的
        # 输入的sub和rel是完全打乱了的，既包含正向relation 也包含负向relation
        # sub: torch.LongTensor(batch_size)
        # rel: torch.LongTensor(batch_size)
        sub_ambiguity = torch.FloatTensor([i[0] for i in ambiguity]).to(self.device)
        rel_ambiguity = torch.FloatTensor([i[1] for i in ambiguity]).to(self.device)
        obj_ambiguity = torch.FloatTensor([i[2] for i in ambiguity]).to(self.device)
        # sub = sub*sub_ambiguity
        if split == 'train':
            obj = torch.LongTensor(obj).to(self.device)
            h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed(sub, rel, obj)
            # rel1 = rel.unsqueeze(1)
            # sub1 = sub.unsqueeze(1)
            # c = torch.nn.functional.softplus(self.c(rel1))
            # head = self.entity(sub1)
            # 这里当然是正常的[64,400]
            # 为了构造复数，分成了两边
            # rank = head[..., :self.p.dim//2]
            # t1 = head[..., self.p.dim//2:]
            #可能是这个torch就是不行。。
            # head = head[..., :self.p.dim//2] + 1j * head[..., self.p.dim//2:]
            # head = torch.irfft(head, signal_ndim=1,normalized=True)  # 这里又做了傅里叶
            # head = self.expmap0(head, c)  # hyperbolic
            # head2 = head.squeeze(1)#这就是实现了双曲？！
            n_hn_e_real, n_hn_e_img, n_rel_hn_e_real, n_rel_hn_e_img, n_t_e_real, n_t_e_img = self.embed(neg_hn_batch,
                                                                                                         neg_rel_hn_batch,
                                                                                                         neg_t_batch)
            n_h_e_real, n_h_e_img, n_rel_tn_e_real, n_rel_tn_e_img, n_tn_e_real, n_tn_e_img = self.embed(neg_h_batch,
                                                                                                         neg_rel_tn_batch,
                                                                                                         neg_tn_batch)
            try:
                f_prob_hn = self.liner(torch.unsqueeze(torch.sum(
                    n_hn_e_real * n_t_e_real * n_rel_hn_e_real + n_hn_e_img * n_t_e_img * n_rel_hn_e_real + n_hn_e_real * n_t_e_img * n_rel_hn_e_img - n_hn_e_img * n_t_e_real * n_rel_hn_e_img,
                    dim=2), dim=-1))
            except:
                print('123')
            f_prob_tn = self.liner(torch.unsqueeze(torch.sum(
                n_h_e_real * n_tn_e_real * n_rel_tn_e_real + n_h_e_img * n_tn_e_img * n_rel_tn_e_real + n_h_e_real * n_tn_e_img * n_rel_tn_e_img - n_h_e_img * n_tn_e_real * n_rel_tn_e_img,
                dim=2), dim=-1))
            f_prob_hn = torch.squeeze(f_prob_hn, dim=-1)
            f_score_hn = torch.mean(torch.mul(f_prob_hn, f_prob_hn), dim=1)
            f_prob_tn = torch.squeeze(f_prob_tn, dim=-1)
            f_score_tn = torch.mean(torch.mul(f_prob_tn, f_prob_tn), dim=1)
            loss_neg = torch.sum((f_score_tn + f_score_hn) / 2.0) / self.p.batch_size

            # 这一步就是加上了模糊度，乘上对应元素的模糊度，效果就好了？
            # 元素的模糊度和三元组的模糊度应该不冲突吧，就是没有泄露信息吧
            h_e_real = torch.FloatTensor(
                np.asarray([(i * j).detach().cpu().numpy() for i, j in zip(h_e_real, sub_ambiguity)])).to(self.device)
            h_e_img = torch.FloatTensor(
                np.asarray([(i * j).detach().cpu().numpy() for i, j in zip(h_e_img, sub_ambiguity)])).to(self.device)
            r_e_real = torch.FloatTensor(
                np.asarray([(i * j).detach().cpu().numpy() for i, j in zip(r_e_real, rel_ambiguity)])).to(self.device)
            r_e_img = torch.FloatTensor(
                np.asarray([(i * j).detach().cpu().numpy() for i, j in zip(r_e_img, rel_ambiguity)])).to(self.device)
            t_e_real = torch.FloatTensor(
                np.asarray([(i * j).detach().cpu().numpy() for i, j in zip(t_e_real, obj_ambiguity)])).to(self.device)
            t_e_img = torch.FloatTensor(
                np.asarray([(i * j).detach().cpu().numpy() for i, j in zip(t_e_img, obj_ambiguity)])).to(self.device)

            # for i,j in zip(h_e_real,sub_ambiguity):
            # 		kt = (i*j).detach().cpu().numpy()
            # 		# kt = kt.numpy()
            # 		h_e_real_tmp.append(kt)
            # h_e_real_tmp =np.asarray(h_e_real_tmp)
            # h_e_real = torch.FloatTensor(h_e_real_tmp).to(self.device)

            htr = torch.unsqueeze(torch.sum(h_e_real * t_e_real * r_e_real + h_e_img * t_e_img * r_e_real +
                                            h_e_real * t_e_img * r_e_img - h_e_img * t_e_real * r_e_img, dim=1), dim=-1)
            f_prob = self.liner(htr)
            f_prob = torch.squeeze(f_prob, dim=-1)
            w = torch.FloatTensor([i[3] for i in ambiguity]).to(self.device)
            # f_score = torch.mul(f_prob-w,f_prob-w)
            # loss_comp_mse = torch.sum(f_score) / self.p.batch_size
            loss_comp_mse = F.mse_loss(f_prob, w)

        sub_emb, rel_emb, all_ent = self.forward_for_kg_completion(sub, rel, sub_index_list, init_ent_list,
                                                                   final_ent_list, ent_neighbors, rel_evt_idxs,
                                                                   evt_neighbors, new_event_index,
                                                                   new_entity_event_index, new_entity_mask,
                                                                   new_entity_list, new_entity_type)
        # sub_emb, rel_emb: torch.FloatTensor(batch_size, gcn_dim)
        # all_ent: torch.FloatTensor(ent_num, gcn_dim)
        # print(sub_emb.shape)
        # print(rel_emb.shape)
        # print(all_ent.shape)
        stk_inp = self.concat(sub_emb, rel_emb)#反正gcn就是得到了头实体和关系的嵌入
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)  # 反正本来是[64,200]，嵌入的是200的维度，但是后面特征提取之后就是165的嵌入维数
        # print(x.shape)
        # print(all_ent.shape)
        # 其实头实体和关系在此之前就一直都是[batchsize,200]
        # 这个200到底是什参数，意味着什么
        # 之后得到的X也是[batchsize,embeeding]
        # 尾实体的embeding是[采样数,embedding]
        # 尾实体是采样的，其实不用管？
        x = torch.mm(x, all_ent.transpose(1, 0))
        # x就是头和关系的分数，all_ent就是尾的分数，然后相乘得到分数？？，不太对
        # 应该是采样了一部分实体，和真实的进行比较，得到分数，这个应该不是对三元组进行打分
        # print(self.bias.shape)
        # x += self.bias.expand_as(x)

        score = torch.sigmoid(x)  # [64,165]我觉的这里就可以是三元组的评分
        # print(torch.isnan(score).any())#判断有没有nan，True就是有nan
        # print(torch.isnan(x).any())
        if split == 'train':
            return score, loss_comp_mse, loss_neg
        else:
            return score

    def valid_all_batch(self, sub, rel, obj):
        # obj = torch.LongTensor(obj).to(self.device)
        # 这一步感觉确实存在问题，还嵌入，不应该是直接得到对应的权重？
        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed(sub, rel, obj)
        htr = torch.unsqueeze(torch.sum(h_e_real * t_e_real * r_e_real + h_e_img * t_e_img * r_e_real +
                                        h_e_real * t_e_img * r_e_img - h_e_img * t_e_real * r_e_img, dim=1), dim=-1)

        f_prob = self.liner(htr)
        f_prob = torch.squeeze(f_prob, dim=-1)
        # w = torch.FloatTensor([i[3] for i in ambiguity]).to(self.device)
        return f_prob

    def predict(self, sub, rel):
        # 但是输入的是一个batch的所有relation和entity，也就是说这个图并不是完整的
        # 输入的sub和rel是完全打乱了的，既包含正向relation 也包含负向relation
        # sub: torch.LongTensor(batch_size)
        # rel: torch.LongTensor(batch_size)
        # sub_emb, rel_emb, all_ent	= self.forward_for_kg_completion(sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type)

        sub_emb, rel_emb, all_ent = self.predict_for_kg_completion(sub, rel)
        # sub_emb, rel_emb: torch.FloatTensor(batch_size, gcn_dim)
        # all_ent: torch.FloatTensor(ent_num, gcn_dim)
        # print(sub_emb.shape)
        # print(rel_emb.shape)
        # print(all_ent.shape)
        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # print(x.shape)
        # print(all_ent.shape)
        x = torch.mm(x, all_ent.transpose(1, 0))
        # print(self.bias.shape)
        # x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        # print(score)
        return score
