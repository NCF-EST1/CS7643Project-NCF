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

def main():
    # parse command line arguments and arguments from config file
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='dataset', dest='dataset')
    parser.add_argument('-m', help='models', dest='models')
    parser.add_argument('-c', help='config files', dest='configs')
    parser.add_argument('-s', help='dataset threshold', dest='dthreshold')
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
        if args.models == 'GMF':
            model = GMF(num_users, num_items, args.gmf_emb_size)
        if args.models == 'NeuMF':
            model = NeuMF(num_users, num_items, args.mlp_emb_size, args.gmf_emb_size, args.layers)
    model = model.to(device)
    print(model)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)

    # run through epochs, track loss/hr/ndcg history and best value
    loss_history, hr_history, ndcg_history = [], [], []
    best_model = None
    best_hr, best_ndcg, best_epoch = 0.0, 0.0, 0
    for epoch in range(args.epochs):
        # training
        user_input, item_input, labels = get_train_instances(train, args.num_negatives, num_users)
        ds = MyDataset(user_tensor=torch.LongTensor(user_input), item_tensor=torch.LongTensor(item_input), label_tensor=torch.LongTensor(labels))
        train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        loss = training(epoch, train_loader, model, optimizer, criterion, device)

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

    # save loss, hr, ndcg as pickle, easy to load and compare against multiple models/configs
    save_dict = {}
    save_dict["loss"] = loss_history
    save_dict["hr"] = hr_history
    save_dict["ndcg"] = ndcg_history
    with open("./saved_models/" + str(model.name) + ".pickle", 'wb') as pickle_file:
        pickle.dump(save_dict, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    # generate plots
    plt.plot(loss_history)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('./saved_models/' + str(model.name) + '_loss.png')
    plt.clf()
    plt.plot(hr_history)
    plt.xlabel('epoch')
    plt.ylabel('HR@10')
    plt.savefig('./saved_models/' + str(model.name) + '_HR.png')
    plt.clf()
    plt.plot(ndcg_history)
    plt.xlabel('epoch')
    plt.ylabel('NDCGR@10')
    plt.savefig('./saved_models/' + str(model.name) + '_NDCG.png')
    plt.clf()


if __name__ == '__main__':
    main()


