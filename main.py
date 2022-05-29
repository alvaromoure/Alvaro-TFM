import argparse
import time
import os
import torch
import numpy as np
import utils.util as util
from trainer.train import initialize, train, validation, initialize_from_saved_model
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import plotly.express as px

def main():
    torch.cuda.empty_cache()
    args = get_arguments()
    wandb.init(project='Alvaro-TFM-KAGGLE',entity='alvaromoureupm',config=args.wandb)
    wandb.run.name = args.name
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if (args.cuda):
        torch.cuda.manual_seed(SEED)
    if args.new_training:
        model, optimizer, training_generator, val_generator, class_weight, Last_epoch = initialize_from_saved_model(args)
    else:
        model, optimizer, training_generator, val_generator, class_weight = initialize(args)
        Last_epoch = 0

    best_pred_loss = 0 #lo cambie por balanced accuracy
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-5, verbose=True)
    print('Checkpoint folder ', args.save)
    for epoch in range(1, args.nEpochs + 1):
        start_time = time.time()
        train_metrics = train(args, model, training_generator, optimizer, Last_epoch+epoch, class_weight)
        elapsed_time = time.time()-start_time
        wandb.log({'epoch': epoch, 'train accuracy': train_metrics.avg_acc(),
                   'train loss': train_metrics.avg_loss(), 'elaped_time': elapsed_time})
        print('Performing validation...')
        start_time = time.time()
        val_metrics, confusion_matrix = validation(args, model, val_generator, Last_epoch+epoch, class_weight)
        elapsed_time = time.time()-start_time
        BACC = BalancedAccuray(confusion_matrix.numpy())
        val_metrics.replace({'bacc': BACC})
        wandb.log({'epoch': epoch, 'validation accuracy': val_metrics.avg_acc(),
                   'val loss': val_metrics.avg_loss(), 'val balanced accuracy': BACC, 'elaped_time': elapsed_time})

        wandb.log({f'Confusion Matrix Epoch {epoch}': px.imshow(confusion_matrix, text_auto=True,
                        x=['pneumonia', 'normal', 'COVID-19'],
                        y=['pneumonia', 'normal', 'COVID-19'])})

        print('Saving this epochs model...')
        best_pred_loss = util.save_model(model, optimizer, args,
                                         val_metrics, Last_epoch+epoch,
                                         best_pred_loss, confusion_matrix)
        print(confusion_matrix)
        scheduler.step(val_metrics.avg_loss())
    print('Saving model to wandb')
    wandb.save(os.path.join(args.save,args.model + '_best'+'_checkpoint.pth.tar'))

def BalancedAccuray(CM):
    Nc = CM.shape[0]
    BACC = np.zeros(Nc)
    for i in range(Nc):
        BACC[i] = CM[i,i]/np.sum(CM[i,:])
    return np.mean(BACC)



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_training', action='store_true', default=False,
                        help='load saved_model as initial model')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--dataset_name', type=str, default="COVIDx")
    parser.add_argument('--nEpochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--lr', default=2e-5, type=float,
                        help='learning rate (default: 2e-5)')
    parser.add_argument('--weight_decay', default=1e-7, type=float,
                        help='weight decay (default: 1e-6)')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='COVIDNet',
                        choices=('COVIDNet','CovidNet_ResNet50', 'CovidNet_DenseNet', 'CovidNet_Grad_CAM','CovidNet_DE'))
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--dataset', type=str, default='/content/covid-chestxray-dataset/data',
                        help='path to dataset ')
    parser.add_argument('--saved_model', type=str, default='/mnt/Data/AlvaroTFM/Alvaro-TFM/saved_model/COVIDNet_best_checkpoint.pth.tar',
                        help='path to save_model ')
    parser.add_argument('--save', type=str, default='/mnt/Data/AlvaroTFM/Alvaro-TFM/ModelSavedCovidNet/COVIDNet' + util.datestr(),
                        help='path to checkpoint ')
    parser.add_argument('--train_split_file',type=str,default='train_split_alvaro.txt',help='path to train split file')
    parser.add_argument('--test_split_file',type=str,default='test_split_alvaro.txt',help='path to train split file')
    parser.add_argument('--num_workers',type=int,default=2,help='used to specify the number of workers')
    parser.add_argument('--name', type=str, default='', help='Used to specify a run name for wandb platform')
    parser.add_argument('--wandb',type=dict,default='',help='Include wandb configuration here')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
