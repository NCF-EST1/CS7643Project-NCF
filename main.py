import heapq
import math
from statistics import mean
import yaml
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import MyDataset
from process_data import Dataset
from models.NCF import NCF
from models.GMF import GMF
from models.NCF_GMF import NCF_GMF

def training(epoch, train_loader, model, optimizer, criterion):
    losses = []
    for idx, (user, item, target) in enumerate(train_loader):
        out  = model(user, item)
        target = target.float()
        loss = criterion(out.view(-1), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
        if idx % 100 == 0:
            print(('Training Epoch: {0} {1}/{2} Loss: {3} AvgLoss: {4}').format(epoch, idx, len(train_loader), loss, mean(losses)))

def validate_single(i, ratings, negatives, model, criterion):
    user = ratings[i][0]
    item = ratings[i][1]
    items = negatives[i]
    items.append(item)
    user_in = np.full(len(items), user, dtype='int32')
    user_in = torch.LongTensor(user_in)
    items = torch.LongTensor(items)

    with torch.no_grad():
        out = model(user_in, items)
    item_pred = {}
    for i in range(len(items)):
        item_pred[items[i]] = out[i]
    ranklist = heapq.nlargest(10, item_pred, key=item_pred.get)
    hr, ndcg = 0, 0
    if item in ranklist:
        hr = 1
        idx = ranklist.index(item)
        ndcg = math.log(2)/math.log(idx+2)
    return hr, ndcg

def validate(epoch, test_ratings, test_negatives, model, criterion):
    hrs, ndcgs = [], []
    for i in range(len(test_ratings)):
        hr, ndcg = validate_single(i, test_ratings, test_negatives, model, criterion)
        hrs.append(hr)
        ndcgs.append(ndcg)

        if i % 100 == 0:
            print(('Validation Epoch: {0} {1}/{2} HitRate: {3} AvgHitRate: {4}').format(epoch, i, len(test_ratings), hr, mean(hrs)))
    
    return hrs, ndcgs

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users, num_items = train.shape
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='dataset', dest='dataset')
    parser.add_argument('-m', help='models', dest='models')
    parser.add_argument('-c', help='config files', dest='configs')

    args = parser.parse_args()
    if args.configs:
        with open(args.configs) as f:
            config = yaml.load(f, yaml.Loader)
        for key in config:
            setattr(args, key, config[key])

    if args.dataset:
        if args.dataset == "movielens":
            dataset = Dataset("./data/ml-1m")
        else:
            dataset = Dataset("./data/pinterest-20")

    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape

    if args.models:
        if args.models == 'NCF':
            model = NCF(num_users, num_items, 8)
    print(model)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_model = None
    best_hr = 0.0
    for epoch in range(args.epochs):
        user_input, item_input, labels = get_train_instances(train, 4)
        ds = MyDataset(user_tensor=torch.LongTensor(user_input), item_tensor=torch.LongTensor(item_input), label_tensor=torch.LongTensor(labels))
        train_loader = DataLoader(ds, batch_size=256, shuffle=True)
        training(epoch, train_loader, model, optimizer, criterion)
        hrs, ndcgs = validate(epoch, testRatings, testNegatives, model, criterion)

        print(("Validation Result Epoch: {0} HitRate: {1} Best HitRate: {2}").format(epoch, mean(hrs), best_hr))

        if mean(hrs) > best_hr:
            best_hr = mean(hrs)
            best_model = copy.deepcopy(model)

    if args.save_best:
        torch.save(best_model.state_dict(), './saved_models/' + str(model.name) + '.pth')

if __name__ == '__main__':
    main()


