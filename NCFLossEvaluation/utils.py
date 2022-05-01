import argparse
import yaml
import torch
import torch.nn as nn
from models.NCF import NCF
from models.GMF import GMF
from models.NeuMF import NeuMF
from data.process_data import Dataset
import numpy as np
from statistics import mean
import math


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='dataset', dest='dataset')
    parser.add_argument('-m', help='models', dest='models')
    parser.add_argument('-c', help='config files', dest='configs')
    parser.add_argument('-s', help='dataset threshold', dest='dthreshold')
    parser.add_argument('-l', help='loss function', dest='loss_fn')
    parser.add_argument('-o', help='optimizer', dest='opt')
    parser.add_argument('-n', help='model name', dest='model_name')
    parser.add_argument('-a', help='alpha', dest='alpha')
    return parser


def translate_args(args):
    if args.configs:
        with open(args.configs) as f:
            config = yaml.load(f, yaml.Loader)
        for key in config:
            setattr(args, key, config[key])

    # load dataset and extract train & testing instances
    if args.dataset:
        if args.dataset == "movielens":
            dataset = Dataset("../data/ml-1m")
        else:
            dataset = Dataset("../data/pinterest-20")
    else:
        dataset = Dataset("../data/pinterest-20")
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives

    # apply dthreshold to use a subset of entire dataset
    num_users, num_items = train.shape
    dthreshold = 1.0
    if args.dthreshold:
        dthreshold = float(args.dthreshold)
    num_users = int(num_users * dthreshold)

    # load model
    if args.models:
        if args.models == 'NCF':
            model = NCF(num_users, num_items, args.mlp_emb_size, args.layers)
            emb_size = args.mlp_emb_size
            layers =  args.layers
        if args.models == 'GMF':
            model = GMF(num_users, num_items, args.gmf_emb_size)
            emb_size = args.gmf_emb_size
            layers = ''
        if args.models == 'NeuMF':
            model = NeuMF(num_users, num_items, args.mlp_emb_size, args.gmf_emb_size, args.layers)
            emb_size = args.mlp_emb_size
            layers =  args.layers
    else:
        model = NCF(num_users, num_items, args.mlp_emb_size, args.layers)
        emb_size = args.mlp_emb_size
        layers =  args.layers
    print(model)

    # define the loss function
    if args.loss_fn:
        if args.loss_fn == 'Top1':
            criterion = TOP1
        if args.loss_fn == 'BPR':
            criterion = BPR
    else:
        criterion = BPR

    # define the optimizer
    if args.opt:
        if args.opt == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)
        if args.opt == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)
        if args.opt == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)
        if args.opt == 'AdaGrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)

    return dataset, dthreshold, model, emb_size, layers, criterion, optimizer


def validate(epoch, test_ratings, test_negatives, model, device, num_users):
    '''
    Validate each test instances, return average HitRates@10 and average NDCGs@10.
    '''
    hrs, ndcgs = [], []
    for i in range(len(test_ratings)):
        # only validate on the subset used for training
        if test_ratings[i][0] < num_users:
            hr, ndcg = validate_single(i, test_ratings, test_negatives, model, device)
            hrs.append(hr)
            ndcgs.append(ndcg)

            if i % 100 == 0:
                print(('Validation Epoch: {0} {1}/{2} AvgHitRate: {3} AvgNDCG: {4}').format(epoch, i, len(test_ratings), mean(hrs), mean(ndcgs)))

    return mean(hrs), mean(ndcgs)


def validate_single(i, ratings, negatives, model, device):
    '''
    Validate a single test instances, return HitRates@10 and NDCGs@10.
    '''
    user = ratings[i][0]
    item = ratings[i][1]

    # negative items + 1 positive items
    items = negatives[i]
    items.append(item)
    user_in = np.full(len(items), user, dtype='int32')
    user_in = torch.LongTensor(user_in)
    items = torch.LongTensor(items)

    model.eval()
    with torch.no_grad():
        user_in = user_in.to(device)
        items = items.to(device)
        out = model(user_in, items)

    # get prediction score for each items, find the 10 highest
    preds = out.view(-1).tolist()
    highest_pred = sorted(preds, reverse=True)
    highest_item = []
    for i in range(10):
        pred_idx = preds.index(highest_pred[i])
        highest_item.append(items[pred_idx])

    # compute hit rate and ndcg
    hr, ndcg = 0, 0
    if item in highest_item:
        hr = 1
        idx = highest_item.index(item)
        ndcg = math.log(2)/math.log(idx+2)
    return hr, ndcg


def get_train_instances(train, num_negatives, num_users):
    '''
    Returns vector of user, item and label with all positive examples in "train" and num_negatives negative examples
    (i.e. no interaction between user and item) derived from all items.
    Overall instances is limited by dthreshold number of users.
    '''
    p_user_input, p_item_input, p_labels = [], [], []
    n_user_input, n_item_input, n_labels = [], [], []
    _, num_items = train.shape
    for (u, i) in train.keys():
        if u >= num_users:
            break

        for t in range(num_negatives):
            # positive instance
            p_user_input.append(u)
            p_item_input.append(i)
            p_labels.append(1)

        # negative instances: random sampling from items until finding one that user u do not have interaction with
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            n_user_input.append(u)
            n_item_input.append(j)
            n_labels.append(0)

    return p_user_input, p_item_input, p_labels, n_user_input, n_item_input, n_labels


def TOP1(item_i, item_j):
    diff = item_j - item_i
    loss = (torch.sigmoid(diff) + torch.sigmoid(torch.pow(item_j, 2)))
    return torch.mean(loss)


def BPR(item_i, item_j):
    diff = item_i - item_j
    return -torch.mean(torch.logsigmoid(diff))
