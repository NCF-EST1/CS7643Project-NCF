import pickle
import matplotlib.pyplot as plt

def run_analysis():

    # optimizers
    adagrad = get_data('./analysis/optimizers/', 'NCF_loss_BCE_opt_AdaGrad_lr_0.001_layers_[64, 32, 16, 8]_emb_size_64_negatives_4_batch_size_256_epochs_20')
    adam = get_data('./analysis/optimizers/', 'NCF_loss_BCE_opt_Adam_lr_0.001_layers_[64, 32, 16, 8]_emb_size_64_negatives_4_batch_size_256_epochs_20')
    RMSprop = get_data('./analysis/optimizers/', 'NCF_loss_BCE_opt_RMSprop_lr_0.001_layers_[64, 32, 16, 8]_emb_size_64_negatives_4_batch_size_256_epochs_20')
    sgd = get_data('./analysis/optimizers/', 'NCF_loss_BCE_opt_SGD_lr_0.001_layers_[64, 32, 16, 8]_emb_size_64_negatives_4_batch_size_256_epochs_20')

    figure, axis = plt.subplots(1, 2)
    figure.set_figwidth(10)
    figure.set_figheight(4)
    axis[0].plot(adagrad['loss'], label='Adagrad')
    axis[0].plot(adam['loss'], label='Adam')
    axis[0].plot(RMSprop['loss'], label='RMSprop')
    axis[0].plot(sgd['loss'], label='SGD')
    axis[0].legend(loc="lower left", prop={'size': 8})
    axis[0].set_title('Average Loss per Epoch for Optimizers')
    axis[0].set_ylabel('Loss')
    axis[0].set_xlabel('Epoch')

    axis[1].plot(adagrad['hr'], label='Adagrad')
    axis[1].plot(adam['hr'], label='Adam')
    axis[1].plot(RMSprop['hr'], label='RMSprop')
    axis[1].plot(sgd['hr'], label='SGD')
    axis[1].legend(loc="lower left", prop={'size': 8})
    axis[1].set_title('Average Hit Rate per Epoch for Optimizers')
    axis[1].set_ylabel('HR')
    axis[1].set_xlabel('Epoch')
    plt.savefig('./results/optimizers.png')
    plt.clf()

    #Losses
    bce = get_data('./analysis/losses/', 'NCF_loss_BCE_opt_Adam_lr_0.001_layers_[64, 32, 16, 8]_emb_size_64_negatives_4_batch_size_256_epochs_20_alpha_0')
    bpr05 = get_data('./analysis/losses/', 'NCF_loss_BPR_opt_Adam_lr_0.001_layers_[64, 32, 16, 8]_emb_size_64_negatives_4_batch_size_256_epochs_20_alpha_0.5')
    mse = get_data('./analysis/losses/', 'NCF_loss_MSE_opt_Adam_lr_0.001_layers_[64, 32, 16, 8]_emb_size_64_negatives_4_batch_size_256_epochs_20_alpha_0')
    top105 = get_data('./analysis/losses/', 'NCF_loss_Top1_opt_Adam_lr_0.001_layers_[64, 32, 16, 8]_emb_size_64_negatives_4_batch_size_256_epochs_20_alpha_0.5')
    top100 = get_data('./analysis/losses/', 'NCF_loss_Top1_opt_Adam_lr_0.001_layers_[64, 32, 16, 8]_emb_size_64_negatives_4_batch_size_256_epochs_20_alpha_0.0')
    top1025 = get_data('./analysis/losses/', 'NCF_loss_Top1_opt_Adam_lr_0.001_layers_[64, 32, 16, 8]_emb_size_64_negatives_4_batch_size_256_epochs_20_alpha_0.25')
    top1075 = get_data('./analysis/losses/', 'NCF_loss_Top1_opt_Adam_lr_0.001_layers_[64, 32, 16, 8]_emb_size_64_negatives_4_batch_size_256_epochs_20_alpha_0.75')
    top11 = get_data('./analysis/losses/', 'NCF_loss_Top1_opt_Adam_lr_0.001_layers_[64, 32, 16, 8]_emb_size_64_negatives_4_batch_size_256_epochs_20_alpha_1.0')

    figure, axis = plt.subplots(1, 2)
    figure.set_figwidth(10)
    figure.set_figheight(4)
    axis[0].plot(bce['loss'], label='BCE')
    axis[0].plot(bpr05['loss'], label='BPR (α = 0.5)')
    axis[0].plot(mse['loss'], label='MSE')
    axis[0].plot(top105['loss'], label='Top1 (α = 0.5)')
    axis[0].legend(loc="lower left", prop={'size': 8})
    axis[0].set_title('Average Loss per Epoch for Loss Functions')
    axis[0].set_ylabel('Loss')
    axis[0].set_xlabel('Epoch')

    axis[1].plot(bce['hr'], label='BCE')
    axis[1].plot(bpr05['hr'], label='BPR (α = 0.5)')
    axis[1].plot(mse['hr'], label='MSE')
    axis[1].plot(top105['hr'], label='Top1 (α = 0.5)')
    axis[1].legend(loc="lower right", prop={'size': 8})
    axis[1].set_title('Average Hit Rate per Epoch for Loss Functions')
    axis[1].set_ylabel('HR')
    axis[1].set_xlabel('Epoch')
    plt.savefig('./results/loss_functions.png')
    plt.clf()

    figure, axis = plt.subplots(1, 2)
    figure.set_figwidth(10)
    figure.set_figheight(4)
    axis[0].plot(top100['loss'], label='α = 0.0')
    axis[0].plot(top1025['loss'], label='α = 0.2')
    axis[0].plot(top105['loss'], label='α = 0.5')
    axis[0].plot(top1075['loss'], label='α = 0.75')
    axis[0].plot(top11['loss'], label='α = 1.0')
    axis[0].legend(loc="lower left", prop={'size': 8})
    axis[0].set_title('Average Loss per Epoch for Top1 α')
    axis[0].set_ylabel('Loss')
    axis[0].set_xlabel('Epoch')

    axis[1].plot(top100['hr'], label='α = 0.0')
    axis[1].plot(top1025['hr'], label='α = 0.2')
    axis[1].plot(top105['hr'], label='α = 0.5')
    axis[1].plot(top1075['hr'], label='α = 0.75')
    axis[1].plot(top11['hr'], label='α = 1.0')
    axis[1].legend(loc="lower right", prop={'size': 8})
    axis[1].set_title('Average Hit Rate per Epoch for Top1 α')
    axis[1].set_ylabel('HR')
    axis[1].set_xlabel('Epoch')
    plt.savefig('./results/loss_alpha_function.png')
    plt.clf()



def get_data(path, filename):
    file = open(path + filename + '.pickle', 'rb')
    saved_dict = pickle.load(file)
    file.close()
    return saved_dict

if __name__ == "__main__":
    run_analysis()
