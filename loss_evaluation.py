import math
from statistics import mean
import yaml
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.trainloader import MyDataset
from data.process_data import Dataset
from models.NCF import NCF
from models.GMF import GMF
from models.NeuMF import NeuMF
import matplotlib.pyplot as plt
import pickle

from loss_functions import pointwise_loss, bpr_loss


def training(epoch, train_loader, model, optimizer, criterion, device):
    '''
    Multi-batch training, return the average loss value.
    '''
    model.train()
    losses = []
    for idx, (user, item, target) in enumerate(train_loader):
        user = user.to(device)
        item = item.to(device)
        target = target.to(device)
        out  = model(user, item)

        target = target.float()
        loss = criterion(out.view(-1), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if idx % 100 == 0:
            print(('Training Epoch: {0} {1}/{2} Loss: {3} AvgLoss: {4}').format(epoch, idx, len(train_loader), loss, mean(losses)))
    return mean(losses)

def bpr_loss_fn(pos_pred, neg_pred):
    return - (pos_pred - neg_pred).sigmoid().log().sum()

def training_split(epoch, p_train_loader, n_train_loader, model, optimizer, criterion, device):
    '''
    Multi-batch training, return the average loss value.
    '''
    model.train()
    losses = []
    p_samples = []
    n_samples = []
    for idx, (user, item, target) in enumerate(p_train_loader):
        p_samples.append((user, item, target))
    for idx, (user, item, target) in enumerate(n_train_loader):
        n_samples.append((user, item, target))

    for idx, (user, item, target) in enumerate(p_samples):
        user = user.to(device)
        item = item.to(device)
        target = target.to(device)
        out  = model(user, item)
        labels = torch.ones(len(user)).to(device)

        n_user, n_item, n_target = n_samples[idx]
        n_user = n_user.to(device)
        n_item = n_item.to(device)
        n_target = n_target.to(device)
        n_out = model(n_user, n_item)
        res = bpr_loss_fn(out, n_out)
        loss = res
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if idx % 100 == 0:
            print(('Training Epoch: {0} {1}/{2} Loss: {3} AvgLoss: {4}').format(epoch, idx, len(p_train_loader), loss, mean(losses)))
    return mean(losses)

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


def get_train_instances(train, num_negatives, num_users):
    '''
    Returns vector of user, item and label with all positive examples in "train" and num_negatives negative examples
    (i.e. no interaction between user and item) derived from all items.
    Overall instances is limited by dthreshold number of users.
    '''
    user_input, item_input, labels = [],[],[]
    _, num_items = train.shape
    for (u, i) in train.keys():
        if u >= num_users:
            break
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances: random sampling from items until finding one that user u do not have interaction with
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

def get_train_instances_split(train, num_negatives, num_users):
    '''
    Returns vector of user, item and label with all positive examples in "train" and num_negatives negative examples
    (i.e. no interaction between user and item) derived from all items.
    Overall instances is limited by dthreshold number of users.
    '''
    p_user_input, p_item_input, p_labels = [], [], []
    n_user_input, n_item_input, n_labels = [], [], []
    _, num_items = train.shape
    #print('train ', train.keys())
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
    print('size of input', len(p_user_input), len(n_user_input))
    return p_user_input, p_item_input, p_labels, n_user_input, n_item_input, n_labels

def main():
    # parse command line arguments and arguments from config file
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='dataset', dest='dataset')
    parser.add_argument('-m', help='models', dest='models')
    parser.add_argument('-c', help='config files', dest='configs')
    parser.add_argument('-s', help='dataset threshold', dest='dthreshold')
    parser.add_argument('-l', help='loss function', dest='loss_fn')
    parser.add_argument('-o', help='optimizer', dest='opt')
    parser.add_argument('-n', help='model name', dest='model_name')
    args = parser.parse_args()
    if args.configs:
        with open(args.configs) as f:
            config = yaml.load(f, yaml.Loader)
        for key in config:
            setattr(args, key, config[key])

    # load dataset and extract train & testing instances
    if args.dataset:
        if args.dataset == "movielens":
            dataset = Dataset("./data/ml-1m")
        else:
            dataset = Dataset("./data/pinterest-20")
    else:
        dataset = Dataset("./data/pinterest-20")
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives

    # apply dthreshold to use a subset of entire dataset
    num_users, num_items = train.shape
    dthreshold = 1.0
    if args.dthreshold:
        dthreshold = float(args.dthreshold)
    num_users = int(num_users * dthreshold)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("You are using device: %s" % device)

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
    model = model.to(device)
    print(model)

    # define the loss function
    if args.loss_fn:
        if args.loss_fn == 'BCE':
            criterion = nn.BCELoss()
        if args.loss_fn == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss()
        if args.loss_fn == 'KLDiv':
            criterion = nn.KLDivLoss()
        if args.loss_fn == 'MSE':
            criterion = nn.MSELoss()
        loss_fn = args.loss_fn
    else:
        criterion = nn.BCELoss()
        loss_fn = 'BCE'

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
        optimizer_name = args.opt
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)
        optimizer_name = 'Adam'

    # run through epochs, track loss/hr/ndcg history and best value
    loss_history, hr_history, ndcg_history = [], [], []
    best_model = None
    best_hr, best_ndcg, best_epoch = 0.0, 0.0, 0
    for epoch in range(args.epochs):
        # training
        p_user_input, p_item_input, p_labels, n_user_input, n_item_input, n_labels = get_train_instances_split(train, args.num_negatives, num_users)
        p_ds = MyDataset(user_tensor=torch.LongTensor(p_user_input), item_tensor=torch.LongTensor(p_item_input), label_tensor=torch.LongTensor(p_labels))
        n_ds = MyDataset(user_tensor=torch.LongTensor(n_user_input), item_tensor=torch.LongTensor(n_item_input), label_tensor=torch.LongTensor(n_labels))
        p_train_loader = DataLoader(p_ds, batch_size=args.batch_size, shuffle=False)
        n_train_loader = DataLoader(n_ds, batch_size=args.batch_size, shuffle=False)
        loss = training_split(epoch, p_train_loader, n_train_loader, model, optimizer, criterion, device)

        # validation
        hr, ndcg = validate(epoch, testRatings, testNegatives, model, device, num_users)

        # update history and best value
        loss_history.append(loss)
        hr_history.append(hr)
        ndcg_history.append(ndcg)

        print(("Validation Result Epoch: {0} HitRate: {1} Best HitRate: {2} NDCG: {3} Best NDCG: {4}").format(epoch, hr, best_hr, ndcg, best_ndcg))
        if hr > best_hr:
            best_hr = hr
            best_ndcg = ndcg
            best_epoch = epoch
            if args.save_best:
                best_model = copy.deepcopy(model)
                torch.save(best_model.state_dict(), './saved_models/' + str(model.name) + '.pth')

    print("### Finish ### Best Epoch: {0} Best HitRate: {1} Best Ndcg: {2}".format(best_epoch, best_hr, best_ndcg))

    model_name = '{0}_loss_{1}_opt_{2}_lr_{3}_layers_{4}_emb_size_{5}_negatives_{6}_batch_size_{7}_epochs_{8}'.format(model.name, loss_fn, optimizer_name, args.learning_rate, layers, emb_size, args.num_negatives, args.batch_size, args.epochs)
    # save loss, hr, ndcg as pickle, easy to load and compare against multiple models/configs
    save_dict = {}
    save_dict["loss"] = loss_history
    save_dict["hr"] = hr_history
    save_dict["ndcg"] = ndcg_history
    with open("./saved_models/" + str(model_name) + ".pickle", 'wb') as pickle_file:
        pickle.dump(save_dict, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    # generate plots
    plt.plot(loss_history)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('./saved_models/' + str(model_name) + '_loss.png')
    plt.clf()
    plt.plot(hr_history)
    plt.xlabel('epoch')
    plt.ylabel('HR@10')
    plt.savefig('./saved_models/' + str(model_name) + '_HR.png')
    plt.clf()
    plt.plot(ndcg_history)
    plt.xlabel('epoch')
    plt.ylabel('NDCGR@10')
    plt.savefig('./saved_models/' + str(model.name) + '_NDCG.png')
    plt.clf()

if __name__ == '__main__':
    main()