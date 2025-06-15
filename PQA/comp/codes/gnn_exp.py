import argparse
from utils import Runner
parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#换数据集的时候这个也要记得改参数
parser.add_argument('-name', default='testrun', help='Set run name for saving/restoring models')
parser.add_argument('-data', dest='dataset', default='D:\\Fuzzy_multi-hop_reasoning\\MFEGR-main\\PQA\\data\\event_kg_rules\\event_graph', help='Dataset to use, default: FB15k-237')
parser.add_argument('-model', dest='model', default='compgcn', help='Model Name')
parser.add_argument('-score_func', dest='score_func', default='conve', help='Score Function for Link prediction')
parser.add_argument('-opn', dest='opn', default='corr', help='Composition Operation to be used in CompGCN')
#128的batch
parser.add_argument('-batch', dest='batch_size', default=128, type=int, help='Batch size')
# dest才是调用的参数 2560的batch
parser.add_argument('-rel_batch', dest='rel_batch_size', default=2560, type=int,
                    help='Relation Classification Batch size')
parser.add_argument('-gamma', type=float, default=40.0, help='Margin')
parser.add_argument('-gamma1', type=float, default=0.1, help='Event-event propagation hyper-param')
parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
parser.add_argument('-epoch', dest='max_epochs', type=int, default=500, help='Number of epochs')
parser.add_argument('-l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
parser.add_argument('-lr', type=float, default=0.001, help='Starting Learning Rate')#就是 0.001
parser.add_argument('-lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
parser.add_argument('-num_workers', type=int, default=0, help='Number of processes to construct batches')
parser.add_argument('-seed', dest='seed', default=41504, type=int, help='Seed for randomization')
parser.add_argument('-neg_per_positive', dest='neg_per_positive', default=10, type=int, help='Seed for randomization')
parser.add_argument('-restore', dest='restore', action='store_true', help='Restore from the previously saved model')
parser.add_argument('-bias', dest='bias', action='store_true', help='Whether to use bias in the model')

parser.add_argument('-num_bases', dest='num_bases', default=-1, type=int,
                    help='Number of basis relation vectors to use')
parser.add_argument('-init_dim', dest='init_dim', default=200, type=int,
                    help='Initial dimension size for entities and relations')#之前是768
parser.add_argument('-gcn_dim', dest='gcn_dim', default=200, type=int, help='Number of hidden units in GCN')#之前是768
parser.add_argument('-role_type_dim', dest='role_type_dim', default=200, type=int,
                    help='Role type embedding dimension size')
#200这个数就是被embed_dim控制的
parser.add_argument('-embed_dim', dest='embed_dim', default=200, type=int,
                    help='Embedding dimension to give as input to score function')
parser.add_argument('-entity_type_num', dest='entity_type_num', default=25, type=int,
                    help='Number of entity types.')
parser.add_argument('-gcn_layer', dest='gcn_layer', default=1, type=int, help='Number of GCN Layers to use')
parser.add_argument('-gcn_drop', dest='dropout', default=0.1, type=float, help='Dropout to use in GCN Layer')
parser.add_argument('-hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')

# ConvE specific hyperparameters
parser.add_argument('-hid_drop2', dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
parser.add_argument('-feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
parser.add_argument('-k_w', dest='k_w', default=10, type=int, help='ConvE: k_w')
parser.add_argument('-k_h', dest='k_h', default=20, type=int, help='ConvE: k_h')
parser.add_argument('-use_event', dest='use_event', default=1, type=int, help='Whether to use event')
parser.add_argument('-use_temporal', dest='use_temporal', default=0, type=int, help='Whether to use use_temporal')
parser.add_argument('-num_filt', dest='num_filt', default=200, type=int,
                    help='ConvE: Number of filters in convolution')
parser.add_argument('-ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')
parser.add_argument('-alpha', dest='alpha', default=0.01, type=float, help='event_proportion')

parser.add_argument('-logdir', dest='log_dir', default='./log/', help='Log directory')
parser.add_argument('-config', dest='config_dir', default='./config/', help='Config directory')
parser.add_argument('-event_sample_num', dest='event_sample_num', default=10, help='Max Event Sample Num')
parser.add_argument('-entity_sample_num', dest='entity_sample_num', default=10, help='Max Event Sample Num')
parser.add_argument('-eval', dest='eval', default="selected", help='evaluate on gpu or cpu.')
parser.add_argument('-train_all', dest='train_all', default=1, help='whether to train all in one shot')
#128
parser.add_argument('-dim', dest='dim', default=200, type=int, help='实体嵌入的维度')
args_event_graph = parser.parse_args()



model_event_graph = Runner(args_event_graph)

# model.load_data()
# model_event_graph.load_events()





# init_ent_list = []
# sub_index_list = []
# obj_index_list = []
#
# node_pos_dict = {}
#
# sub_list = [i for i in range(10)]
#
# obj_list = [i for i in range(1,11)]
#
#
# for j, idx in enumerate(sub_list):
#     if idx not in node_pos_dict:
#         init_ent_list.append(idx)
#         node_pos_dict.update({idx: len(init_ent_list) - 1})
#         sub_index_list.append(len(init_ent_list) - 1)
#     else:
#         sub_index_list.append(node_pos_dict[idx])
#
#
# for j, idx in enumerate(obj_list):
#     if idx not in node_pos_dict:
#         init_ent_list.append(idx)
#         node_pos_dict.update({idx: len(init_ent_list) - 1})
#         obj_index_list.append(len(init_ent_list) - 1)
#     else:
#         obj_index_list.append(node_pos_dict[idx])
# final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type = model_event_graph.sample_subgraph(
#     init_ent_list)
# print('123')