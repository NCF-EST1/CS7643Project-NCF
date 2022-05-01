import torch
from utils import get_parser, translate_args, get_train_instances, validate, TOP1
from torch.utils.data import DataLoader
import copy
import pickle
from statistics import mean
from data.trainloader import MyDataset


def main():
    parser = get_parser()
    args = parser.parse_args()
    dataset, dthreshold, model, emb_size, layers, criterion, optimizer = translate_args(args)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("You are using device: %s" % device)

    # apply dthreshold to use a subset of entire dataset
    num_users, num_items = train.shape
    num_users = int(num_users * dthreshold)
    model = model.to(device)

    point_loss_function = torch.nn.BCELoss()
    pair_loss_function = criterion
    alpha = float(args.alpha) if args.alpha is not None else 0.2
    print('alpha ', alpha)
    print('criterion ', criterion)

    # run through epochs, track loss/hr/ndcg history and best value
    loss_history, hr_history, ndcg_history, pair_loss_history, point_loss_history = [], [], [], [], []
    best_model = None
    best_hr, best_ndcg, best_epoch = 0.0, 0.0, 0
    for epoch in range(args.epochs):
        model.train()

        # training
        p_user_input, p_item_input, p_labels, n_user_input, n_item_input, n_labels = get_train_instances(train, args.num_negatives, num_users)

        p_ds = MyDataset(user_tensor=torch.LongTensor(p_user_input), item_tensor=torch.LongTensor(p_item_input), label_tensor=torch.LongTensor(p_labels))
        n_ds = MyDataset(user_tensor=torch.LongTensor(n_user_input), item_tensor=torch.LongTensor(n_item_input), label_tensor=torch.LongTensor(n_labels))
        p_train_loader = DataLoader(p_ds, batch_size=args.batch_size, shuffle=False)
        n_train_loader = DataLoader(n_ds, batch_size=args.batch_size, shuffle=False)

        n_samples = []
        for idx, (user, item, target) in enumerate(n_train_loader):
            n_samples.append((user, item, target))

        losses = []
        pair_losses = []
        point_losses = []
        for idx, (user, item, target) in enumerate(p_train_loader):
            p_user = user.to(device)
            p_item = item.to(device)
            p_target = target.to(device)
            p_target = p_target.float()

            n_user, n_item, n_target = n_samples[idx]
            n_user = n_user.to(device)
            n_item = n_item.to(device)
            n_target = n_target.to(device)
            n_target = n_target.float()

            p_out = model(p_user, p_item)
            n_out = model(n_user, n_item)

            positive_point_loss = (1 - alpha) * point_loss_function(p_out.view(-1), p_target)
            negative_point_loss = (1 - alpha) * point_loss_function(n_out.view(-1), n_target)
            point_loss = positive_point_loss + negative_point_loss

            pair_loss = alpha * pair_loss_function(p_out, n_out)

            loss = pair_loss + point_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pair_losses.append(pair_loss.item())
            point_losses.append(point_loss.item())
            losses.append(loss.item())

            if idx % 100 == 0:
                print('Training Epoch: {0} {1}/{2} Loss: {3} AvgLoss: {4}'.format(epoch, idx, len(p_train_loader), loss, mean(losses)))

        epoch_loss = mean(losses)
        epoch_point_loss = mean(point_losses)
        epoch_pair_loss = mean(pair_losses)

        # validation
        hr, ndcg = validate(epoch, testRatings, testNegatives, model, device, num_users)

        # update history and best value
        loss_history.append(epoch_loss)
        hr_history.append(hr)
        ndcg_history.append(ndcg)
        point_loss_history.append(epoch_point_loss)
        pair_loss_history.append(epoch_pair_loss)

        print(("Validation Result Epoch: {0} HitRate: {1} Best HitRate: {2} NDCG: {3} Best NDCG: {4}").format(epoch, hr, best_hr, ndcg, best_ndcg))
        if hr > best_hr:
            best_hr = hr
            best_ndcg = ndcg
            best_epoch = epoch
            if args.save_best:
                best_model = copy.deepcopy(model)
                torch.save(best_model.state_dict(), './saved_models/' + str(model.name) + '.pth')

        print("### Finish ### Best Epoch: {0} Best HitRate: {1} Best Ndcg: {2}".format(best_epoch, best_hr, best_ndcg))

    model_name = '{0}_loss_{1}_opt_{2}_lr_{3}_layers_{4}_emb_size_{5}_negatives_{6}_batch_size_{7}_epochs_{8}_alpha_{9}'.format(model.name, args.loss_fn, 'Adam', args.learning_rate, layers, emb_size, args.num_negatives, args.batch_size, args.epochs, alpha)
    # save loss, hr, ndcg as pickle, easy to load and compare against multiple models/configs
    save_dict = {}
    save_dict["loss"] = loss_history
    save_dict["hr"] = hr_history
    save_dict["ndcg"] = ndcg_history
    save_dict["pointloss"] = point_loss_history
    save_dict["pairloss"] = pair_loss_history
    with open("./saved_models/" + str(model_name) + ".pickle", 'wb') as pickle_file:
        pickle.dump(save_dict, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
