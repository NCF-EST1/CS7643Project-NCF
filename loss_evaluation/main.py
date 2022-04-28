import configparser
import data
from BPR import BPR
from GMF import GMF
from NCF import NCF
import tqdm
import torch
import os
import json

from torch import nn, optim
import evaluation
from loss import pointwise_loss, bpr_loss, hinge_loss, adaptive_hinge_loss


def default_loss_fn(pos_pred, neg_pred):
    #print(-(pos_pred - neg_pred).sigmoid().log().sum())
    return - (pos_pred - neg_pred).sigmoid().log().sum()


def train(model, opt, data_splitter, validation_data, batch_size, config, loss_fn):
    epoch_data = []
    best_model_state_dict = None
    best_ndcg = 0
    if loss_fn is None:
        loss_fn = default_loss_fn
    for epoch in range(config.getint('MODEL', 'epoch')):
        model.train()
        train_loader = data_splitter.make_train_loader(config.getint('MODEL', 'n_negative'), batch_size)
        total_loss = 0
        for batch in tqdm.tqdm(train_loader):
            users, pos_items, neg_items = batch[0], batch[1], batch[2]
            users = users.to('cuda:0')
            pos_items = pos_items.to('cuda:0')
            neg_items = neg_items.to('cuda:0')
            #print(neg_items)
            opt.zero_grad()
            #print('positive', len(users), len(pos_items))
            pos_pred = model(users, pos_items)
            #print('negative', len(users), len(neg_items))
            neg_pred = model(users, neg_items)
            loss = loss_fn(pos_pred, neg_pred)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        hit_ratio, ndcg = evaluation.evaluate(model, validation_data, config.getint('EVALUATION', 'top_k'))
        epoch_data.append({'epoch': epoch, 'loss': total_loss, 'HR': hit_ratio, 'NDCG': ndcg})
        if ndcg > best_ndcg:
            best_model_state_dict = model.state_dict()
        print('[Epoch {}] Loss = {:.2f}, HR = {:.4f}, NDCG = {:.4f}'.format(epoch, total_loss, hit_ratio, ndcg))
    return epoch_data, best_model_state_dict


def save_train_result(best_model_state_dict, epoch_data, batch_size, lr, latent_dim, l2_reg, config):
    result_dir = "data/train_result/batch_size_{}-lr_{}-latent_dim_{}-l2_reg_{}-epoch_{}-n_negative_{}-top_k_{}".format(
        batch_size, lr, latent_dim, l2_reg, config['MODEL']['epoch'], config['MODEL']['n_negative'], config['EVALUATION']['top_k'])
    os.makedirs(result_dir, exist_ok=True)
    torch.save(best_model_state_dict, os.path.join(result_dir, 'model.pth'))
    with open(os.path.join(result_dir, 'epoch_data.json'), 'w') as f:
        json.dump(epoch_data, f, indent=4)


def main():
    config = configparser.ConfigParser()
    config.read('./config.ini')

    data_splitter = data.DataSplitter()
    validation_data = data_splitter.make_evaluation_data('validation')
    test_data = data_splitter.make_evaluation_data('test')

    batch_size = int(config['MODEL']['batch_size'])
    lr = float(config['MODEL']['lr'])
    latent_dim = int(config['MODEL']['latent_dim'])
    l2_reg = float(config['MODEL']['l2_reg'])

    print(batch_size, lr, latent_dim, l2_reg)

    model_name = config['MODEL']['model_name']

    if model_name == 'BPR':
        model = BPR(data_splitter.n_user, data_splitter.n_item, latent_dim)
    elif model_name == 'GMF':
        model = GMF(data_splitter.n_user, data_splitter.n_item, latent_dim)
    elif model_name == 'NCF':
        model = NCF(data_splitter.n_user, data_splitter.n_item, latent_dim, [16, 64, 32, 16, 8])
    else:
        model = BPR(data_splitter.n_user, data_splitter.n_item, latent_dim)
    model.to('cuda:0')

    optimizer = config['MODEL']['optimizer']

    if optimizer == 'Adam':
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    else:
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

    loss_fn_name = config['MODEL']['loss_fn_name']

    if loss_fn_name == 'pointwise':
        loss_fn = pointwise_loss
    elif loss_fn_name == 'hinge':
        loss_fn = hinge_loss
    else:
        loss_fn = default_loss_fn

    print('batch_size = {}, lr = {}, latent_dim = {}, l2_reg = {}'.format(batch_size, lr, latent_dim, l2_reg))
    epoch_data, best_model_state_dict = train(model, opt, data_splitter, validation_data, batch_size, config, loss_fn)
    save_train_result(best_model_state_dict, epoch_data, batch_size, lr, latent_dim, l2_reg, config)

    hit_ratio, ndcg = evaluation.evaluate(model, test_data, config.getint('EVALUATION', 'top_k'))
    print('---------------------------------\nBest result')
    print('HR = {:.4f}, NDCG = {:.4f}'.format(hit_ratio, ndcg))

if __name__ == '__main__':
    main()
