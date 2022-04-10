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

def training(epoch, train_loader, model, optimizer, criterion, device):
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

def validate_single(i, ratings, negatives, model, criterion, device):
    user = ratings[i][0]
    item = ratings[i][1]
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

def validate(epoch, test_ratings, test_negatives, model, criterion, device):
    hrs, ndcgs = [], []
    for i in range(len(test_ratings)):
        hr, ndcg = validate_single(i, test_ratings, test_negatives, model, criterion, device)
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("You are using device: %s" % device)

    if args.models:
        if args.models == 'NCF':
            model = NCF(num_users, num_items, args.mlp_emb_size)
        if args.models == 'GMF':
            model = GMF(num_users, num_items, args.gmf_emb_size)
        if args.models == 'NCF_GMF':
            model = NCF_GMF(num_users, num_items, args.mlp_emb_size, args.gmf_emb_size)
    model = model.to(device)
    print(model)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_model = None
    best_hr, best_ndcg, best_epoch = 0.0, 0.0, 0
    for epoch in range(args.epochs):
        user_input, item_input, labels = get_train_instances(train, args.num_negatives)
        ds = MyDataset(user_tensor=torch.LongTensor(user_input), item_tensor=torch.LongTensor(item_input), label_tensor=torch.LongTensor(labels))
        train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        training(epoch, train_loader, model, optimizer, criterion, device)
        hrs, ndcgs = validate(epoch, testRatings, testNegatives, model, criterion, device)

        print(("Validation Result Epoch: {0} HitRate: {1} Best HitRate: {2}").format(epoch, mean(hrs), best_hr))

        if mean(hrs) > best_hr:
            best_hr = mean(hrs)
            best_ndcg = mean(ndcgs)
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            if args.save_best:
                torch.save(best_model.state_dict(), './saved_models/' + str(model.name) + '.pth')

    print("Best Epoch: {0} HitRate: {1} Ndcg: {2}".format(best_epoch, best_hr, best_ndcg))


if __name__ == '__main__':
    main()


