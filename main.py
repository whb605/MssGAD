# -*- coding: utf-8 -*- loss带权重(OK)
"""
@author: LinFu
"""
import copy
import math

import numpy as np
from sklearn import manifold
from sklearn import preprocessing
from sklearn.metrics import auc, precision_recall_curve, roc_curve
import argparse
import load_data
import torch
import torch.nn as nn
import GCN_embedding
from torch.autograd import Variable
from graph_sampler import GraphSampler
from numpy.random import seed
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

def arg_parse():
    parser = argparse.ArgumentParser(description='GLADD Arguments.')
    parser.add_argument('--datadir', dest='datadir', default='dataset', help='Directory where benchmark is located')
    parser.add_argument('--DS', dest='DS', default='BZR', help='dataset name')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int, default=0,
                        help='Maximum number of nodes (ignore gr1aghs with nodes exceeding the number.')
    parser.add_argument('--clip', dest='clip', default=0.1, type=float, help='Gradient clipping.')
    parser.add_argument('--lr', dest='lr', default=0.001, type=float, help='Learning Rate.')
    parser.add_argument('--num_epochs', dest='num_epochs', default=100, type=int, help='total epoch number')
    parser.add_argument('--batch-size', dest='batch_size', default=800, type=int, help='Batch size.')
    parser.add_argument('--hidden-dim', dest='hidden_dim', default=256, type=int, help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', default=128, type=int, help='Output dimension')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', default=5, type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const', const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', default=0.5, type=float, help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const', const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--seed', dest='seed', type=int, default=1, help='seed')
    parser.add_argument('--sign', dest='sign', type=int, default=1, help='sign of graph anomaly')
    parser.add_argument('--feature', dest='feature', default='default', help='use what node feature')
    parser.add_argument('--includingTest', dest='includingTest', action="store_true",help='add test set into training set')
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(dataset_p, dataset_n, dataset_lp, dataset_ln, dataset_t, model, args):
    optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.5)
    alpha = 0
    beta = 0
    max_AUC = 0
    auroc_final = 0


    for epoch in range(args.num_epochs):
        model.train()

        for batch_idx, data in enumerate(dataset_p):
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            model_embeddings = model(h0, adj)
            landmark_embeddings_p = torch.tensor([])
            landmark_embeddings_n = torch.tensor([])
            for batch_idx, data in enumerate(dataset_lp):
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
                h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
                embed_node_lp = model(h0, adj)
                landmark_embeddings_p = embed_node_lp
            for batch_idx, data in enumerate(dataset_ln):
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
                h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
                embed_node_ln = model(h0, adj)
                landmark_embeddings_n = embed_node_ln

            model_embed = model_embeddings.mean(dim=1)
            landmark_embed_p = landmark_embeddings_p.mean(dim=1)
            landmark_embed_n = landmark_embeddings_n.mean(dim=1)
            model_node = model_embeddings.max(dim=1).values
            landmark_node_p = landmark_embeddings_p.max(dim=1).values
            landmark_node_n = landmark_embeddings_n.max(dim=1).values
            if alpha==0 and beta==0:
                dp1 = torch.cdist(model_embed, landmark_embed_p).mean(dim=1).mean(dim=0)
                dp2 = torch.cdist(model_node, landmark_node_p).mean(dim=1).mean(dim=0)
                dn1 = torch.cdist(model_embed, landmark_embed_n).mean(dim=1).mean(dim=0)
                dn2 = torch.cdist(model_node, landmark_node_n).mean(dim=1).mean(dim=0)
                print('epoch:{}, dn1-dp1:{}, dn2-dp2:{}'.format(epoch, dn1-dp1, dn2-dp2))
                if torch.log(torch.abs(dn1-dp1)) - torch.log(torch.abs(dn2-dp2)) > 1:#choose graph loss
                    alpha = 1
                elif torch.log(torch.abs(dn2-dp2)) - torch.log(torch.abs(dn1-dp1)) > 1:#choose node loss
                    beta = 1
                else:#choose both
                    alpha = beta = 1/2
                print('epoch:{}, alpha:{}, beta:{}'.format(epoch, alpha, beta))
            graph_distances1 = alpha*torch.cdist(model_embed, landmark_embed_p) + beta*torch.cdist(model_node, landmark_node_p)
            graph_distances2 = alpha*torch.cdist(model_embed, landmark_embed_n) + beta*torch.cdist(model_node, landmark_node_n)
            graph_distances3 = alpha*torch.cdist(landmark_embed_p, landmark_embed_n) + beta*torch.cdist(landmark_node_p, landmark_node_n)


            loss1 = graph_distances1.mean(dim=1).mean(dim=0)
            loss2 = graph_distances2.mean(dim=1).mean(dim=0)
            loss3 = graph_distances3.mean(dim=1).mean(dim=0)
            loss = torch.exp(loss1 - loss2 - loss3)
            # print('epoch:{}, loss1:{}, loss2:{}, loss3:{}'.format(epoch, loss1, loss2, loss3))

            loss.backward(loss.clone().detach())
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer1.step()
            scheduler1.step()


        for batch_idx, data in enumerate(dataset_n):
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            model_embeddings = model(h0, adj)
            landmark_embeddings_p = torch.tensor([])
            landmark_embeddings_n = torch.tensor([])
            for batch_idx, data in enumerate(dataset_lp):
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
                h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
                embed_node_lp = model(h0, adj)
                landmark_embeddings_p = embed_node_lp
            for batch_idx, data in enumerate(dataset_ln):
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
                h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
                embed_node_ln = model(h0, adj)
                landmark_embeddings_n = embed_node_ln

            model_embed = model_embeddings.mean(dim=1)
            landmark_embed_p = landmark_embeddings_p.mean(dim=1)
            landmark_embed_n = landmark_embeddings_n.mean(dim=1)
            model_node = model_embeddings.max(dim=1).values
            landmark_node_p = landmark_embeddings_p.max(dim=1).values
            landmark_node_n = landmark_embeddings_n.max(dim=1).values
            graph_distances1 = alpha*torch.cdist(model_embed, landmark_embed_p) + beta*torch.cdist(model_node, landmark_node_p)
            graph_distances2 = alpha*torch.cdist(model_embed, landmark_embed_n) + beta*torch.cdist(model_node, landmark_node_n)
            graph_distances3 = alpha*torch.cdist(landmark_embed_p, landmark_embed_n) + beta*torch.cdist(landmark_node_p, landmark_node_n)

            loss1 = graph_distances1.mean(dim=1).mean(dim=0)
            loss2 = graph_distances2.mean(dim=1).mean(dim=0)
            loss3 = graph_distances3.mean(dim=1).mean(dim=0)
            loss = torch.exp(loss2 - loss1 - loss3)
            # print('epoch:{}, loss1:{}, loss2:{}, loss3:{}'.format(epoch, loss1, loss2, loss3))

            loss.backward(loss.clone().detach())
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer1.step()
            scheduler1.step()


        if (epoch + 1) % 10 == 0 and epoch > 0:
            model.eval()
            loss = []
            y = []

            landmark_embeddings_p = torch.tensor([])
            landmark_embeddings_n = torch.tensor([])
            for batch_idx, data in enumerate(dataset_lp):
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
                h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
                embed_node_lp = model(h0, adj)
                landmark_embeddings_p = embed_node_lp
            for batch_idx, data in enumerate(dataset_ln):
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
                h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
                embed_node_ln = model(h0, adj)
                landmark_embeddings_n = embed_node_ln

            for batch_idx, data in enumerate(dataset_t):
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
                h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
                model_embeddings = model(h0, adj)
                model_embed = model_embeddings.mean(dim=1)
                landmark_embed_p = landmark_embeddings_p.mean(dim=1)
                landmark_embed_n = landmark_embeddings_n.mean(dim=1)
                model_node = model_embeddings.max(dim=1).values
                landmark_node_p = landmark_embeddings_p.max(dim=1).values
                landmark_node_n = landmark_embeddings_n.max(dim=1).values
                graph_distances1 = alpha*torch.cdist(model_embed, landmark_embed_p) + beta*torch.cdist(model_node, landmark_node_p)
                graph_distances2 = alpha*torch.cdist(model_embed, landmark_embed_n) + beta*torch.cdist(model_node, landmark_node_n)

                loss1 = graph_distances1.mean(dim=1).mean(dim=0)
                loss2 = graph_distances2.mean(dim=1).mean(dim=0)
                loss_ = loss1 - loss2
                loss_ = np.array(loss_.cpu().detach())
                loss.append(loss_)
                if data['label'] == args.sign:
                    y.append(1)
                else:
                    y.append(0)

            label_test = []
            for loss_ in loss:
                label_test.append(loss_)
            label_test = np.array(label_test)

            fpr_ab, tpr_ab, thr_ = roc_curve(y, label_test)
            test_roc_ab = auc(fpr_ab, tpr_ab)
            print('abnormal detection: auroc_ab: {}'.format(test_roc_ab))
            if test_roc_ab > max_AUC:
                max_AUC = test_roc_ab
        if epoch == (args.num_epochs - 1):
            auroc_final = max_AUC
    return auroc_final

if __name__ == '__main__':
    args = arg_parse()
    DS = args.DS
    setup_seed(args.seed)

    if args.DS == "Tox21_MMP_training":
        DSST = "Tox21_MMP_testing"
    if args.DS == "Tox21_p53_training":
        DSST = "Tox21_p53_testing"
    if args.DS == "Tox21_PPAR-gamma_training":
        DSST = "Tox21_PPAR-gamma_testing"
    if args.DS == "Tox21_HSE_training":
        DSST = "Tox21_HSE_testing"

    graphs = load_data.read_graphfile(args.datadir, args.DS, max_nodes=args.max_nodes)
    datanum = len(graphs)
    if args.max_nodes == 0:
        max_nodes_num = max([G.number_of_nodes() for G in graphs])
    else:
        max_nodes_num = args.max_nodes
    print('GraphNumber: {}'.format(datanum))
    graphs_label = [graph.graph['label'] for graph in graphs]

    if (args.includingTest == True):
        test_datadir = args.datadir.split("\\")[0] + "\\" + args.datadir.split("\\")[1] + "\\" + DSST
        graphs_test = load_data.read_graphfile(test_datadir, DSST, max_nodes=args.max_nodes)
        datanum_test = len(graphs_test)
        if args.max_nodes == 0:
            max_nodes_num = max([G.number_of_nodes() for G in graphs_test])
        else:
            max_nodes_num = args.max_nodes
        print('GraphTestNumber: {}'.format(datanum_test))
        graphs_label_test = [graph.graph['label'] for graph in graphs_test]

        graphs += graphs_test
        graphs_label += graphs_label_test

    kfd = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True)
    result_auc = []
    for k, (train_index, test_index) in enumerate(kfd.split(graphs, graphs_label)):
        graphs_train_ = [graphs[i] for i in train_index]
        graphs_test = [graphs[i] for i in test_index]

        graphs_train_n = []
        graphs_train_p = []
        for graph in graphs_train_:
            if graph.graph['label'] != args.sign:
                graphs_train_p.append(graph)
            else:
                graphs_train_n.append(graph)

        num_train_p = len(graphs_train_p)
        num_train_n = len(graphs_train_n)
        num_test = len(graphs_test)
        print('TrainSize_p: {}, TrainSize_n: {}, TestSize: {}'.format(num_train_p, num_train_n, num_test))

        graphs_landmark_p = random.sample(graphs_train_p, 4**math.ceil(np.log10(num_train_p)))
        graphs_landmark_n = random.sample(graphs_train_n, 4**math.ceil(np.log10(num_train_n)))

        dataset_sampler_train_p = GraphSampler(graphs_train_p, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
        dataset_sampler_train_n = GraphSampler(graphs_train_n, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
        dataset_sampler_landmark_p = GraphSampler(graphs_landmark_p, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
        dataset_sampler_landmark_n = GraphSampler(graphs_landmark_n, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)

        model = GCN_embedding.GcnEncoderGraph(dataset_sampler_train_p.feat_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn, args=args).cuda()

        data_train_loader_p = torch.utils.data.DataLoader(dataset_sampler_train_p, shuffle=True, batch_size=args.batch_size)
        data_train_loader_n = torch.utils.data.DataLoader(dataset_sampler_train_n, shuffle=True, batch_size=args.batch_size)
        data_landmark_loader_p = torch.utils.data.DataLoader(dataset_sampler_landmark_p, shuffle=True, batch_size=args.batch_size)
        data_landmark_loader_n = torch.utils.data.DataLoader(dataset_sampler_landmark_n, shuffle=True, batch_size=args.batch_size)

        dataset_sampler_test = GraphSampler(graphs_test, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
        data_test_loader = torch.utils.data.DataLoader(dataset_sampler_test, shuffle=False, batch_size=1)

        result = train(data_train_loader_p, data_train_loader_n, data_landmark_loader_p, data_landmark_loader_n, data_test_loader, model, args)

        result_auc.append(result)

    result_auc = np.array(result_auc)
    auc_avg = np.mean(result_auc)
    auc_std = np.std(result_auc)
    print('auroc{}, average: {}, std: {}'.format(result_auc, auc_avg, auc_std))