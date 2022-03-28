from statistics import mean
import yaml
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from process_data import load_dataset
from models import NCF, GMF, NCF_GMF

def train(epoch, train_loader, model, optimizer, criterion):
    losses = []
    for idx, (input, target) in enumerate(train_loader):
        out  = model(input)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)
    
        # TODO: compute & print accuracy Hit@10

        if idx % 10:
            print(('Epoch: {0} {1}/{2} Loss: {3} AvgLoss: {4}').format(epoch, idx, len(data_loader), loss, mean(losses)))


def validate(epoch, test_loader, model, criterion):
    losses = []
    accs = []
    for idx, (input, target) in enumerate(test_loader):
        with torch.no_grad():
            out = model(input)
            loss = criterion(out, target)
        losses.append(loss)

        # TODO: compute * print accuracy Hit@10
        acc = 0
        accs.append(acc)

        if idx % 10:
            print(('Epoch: {0} {1}/{2} Loss: {3} AvgLoss: {4}').format(epoch, idx, len(data_loader), loss, mean(losses)))
    
    return mean(accs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='dataset', dest='dataset')
    parser.add_argument('-m', help='models', dest='models')
    parser.add_argument('-c', help='config files', dest='configs')

    args = parser.parse_args()
    if args.config:
        with open(args.config) as f:
            config = yaml.load(f)
        for key in config:
            for k, v in config[key].items():
                setattr(args, k, v)

    if args.dataset:
        # TODO: load dataset
        train_dataset, test_dataset = load_dataset(args.dataset)

    if args.models:
        if args.models == 'NCF':
            model = NCF()
        elif args.model == 'GMF':
            model = GMF()
        elif args.model == 'NCF_GMF':
            model = NCF_GMF()
    
    # TODO: fix loss, optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.adam(model.parameters(), lr=args.learning_rate)

    # TODO: fix data loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    best_model = None
    best_acc = 0.0
    for epoch in range(args.epochs):
        # TODO: adjust learning rate

        train(epoch, train_loader, model, optimizer, criterion)
        acc = validate(epoch, test_loader, model, criterion)

        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)

    if args.save_best:
        torch.save(best_model.state_dict(), './saved_models/' + str(model.name) + '.pth')

if __name__ == '__main__':
    main()

