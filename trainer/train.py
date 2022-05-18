import time

import torch
import os
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from model.loss import focal_loss, crossentropy_loss
from utils.util import Metrics, print_stats, print_summary, select_model, select_optimizer, load_model
from model.metric import accuracy, top_k_acc
from COVIDXDataset.dataset import COVIDxDataset
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score
from warnings import simplefilter
import wandb


def initialize(args):
    simplefilter(action='ignore', category=FutureWarning)
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    model = select_model(args)
    optimizer = select_optimizer(args, model)

    if (args.cuda):
        model.cuda()

    train_loader = COVIDxDataset(mode='train', n_classes=args.classes, dataset_path=args.dataset,
                                 dim=(224, 224), train_file=args.train_split_file, test_file=args.test_split_file)
    # print(train_loader.)
    # ------ Class weigths for sampling and for loss function -----------------------------------
    labels = np.unique(train_loader.labels)
    print(labels)
    class_weight = compute_class_weight('balanced', labels, train_loader.labels)


    # ---------- Alphabetical order in labels does not correspond to class order in COVIDxDataset-----
    class_weight = class_weight[::-1]
    # ---------------------------------------------------------------------------------

    if (args.cuda):
        class_weight = torch.from_numpy(class_weight.astype(float)).cuda()
    else:
        class_weight = torch.from_numpy(class_weight.astype(float))
    # print(class_weight.shape)
    # -------------------------------------------
    val_loader = COVIDxDataset(mode='test', n_classes=args.classes, dataset_path=args.dataset,
                               dim=(224, 224), train_file=args.train_split_file, test_file=args.test_split_file)
    # ------------------------------------------------------------------------------------
    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': args.num_workers}

    test_params = {'batch_size': args.batch_size,
                   'shuffle': True,
                   'num_workers': args.num_workers}
    # ------------------------------------------------------------------------------------------
    training_generator = DataLoader(train_loader, **train_params)
    val_generator = DataLoader(val_loader, **test_params)
    return model, optimizer, training_generator, val_generator, class_weight


def train(args, model, trainloader, optimizer, epoch, class_weight):
    model.train()
    criterion = nn.CrossEntropyLoss(weight=class_weight, reduction='mean')

    metrics = Metrics('')
    metrics.reset()
    # -------------------------------------------------------
    # Esto es para congelar las capas de la red preentrenada
    # for m in model.modules():
    #    if isinstance(m, nn.BatchNorm2d):
    #        m.train()
    #        m.weight.requires_grad = False
    #        m.bias.requires_grad = False
    # -----------------------------------------------------
    start_time = time.time()
    for batch_idx, input_tensors in enumerate(trainloader):
        optimizer.zero_grad()
        input_data, target = input_tensors
        if (args.cuda):
            input_data = input_data.cuda()
            target = target.cuda()
        output = model(input_data)
        # loss = focal_loss(output, target)
        if args.model == 'CovidNet_DenseNet':
            output = output[-1]
        loss = crossentropy_loss(output, target, weight=class_weight)
        loss.backward()
        optimizer.step()
        correct, total, acc = accuracy(output, target)
        num_samples = batch_idx * args.batch_size + 1
        _, output_class = output.max(1)
        bacc = balanced_accuracy_score(target.cpu().detach().numpy(), output_class.cpu().detach().numpy())
        metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc, 'bacc': bacc})
        print_stats(args, epoch, num_samples, trainloader, metrics)
    elapsed_time = time.time()-start_time
    wandb.log({'epoch':epoch,'accuracy':metrics.avg_acc(),'loss':metrics.avg_loss(),'elaped_time':elapsed_time})
    print_summary(args, epoch, num_samples, metrics, mode="Training",elapsed_time=elapsed_time)
    return metrics


def validation(args, model, testloader, epoch, class_weight):
    model.eval()

    # -------------------------------------------------------
    # Esto es para congelar las capas de la red preentrenada
    # for m in model.modules():
    #    if isinstance(m, nn.BatchNorm2d):
    #        m.train()
    #        m.weight.requires_grad = False
    #        m.bias.requires_grad = False
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss(size_average='mean')
    metrics = Metrics('')
    metrics.reset()
    confusion_matrix = torch.zeros(args.classes, args.classes)
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, input_tensors in enumerate(testloader):
            input_data, target = input_tensors
            if (args.cuda):
                input_data = input_data.cuda()
                target = target.cuda()
            # print(input_data.shape)
            output = model(input_data)
            if args.model == 'CovidNet_DenseNet':
                output = output[-1]
            # loss = focal_loss(output, target)
            loss = crossentropy_loss(output, target, weight=class_weight)

            correct, total, acc = accuracy(output, target)
            num_samples = batch_idx * args.batch_size + 1
            _, preds = torch.max(output, 1)
            bacc = balanced_accuracy_score(target.cpu().detach().numpy(), preds.cpu().detach().numpy())
            for t, p in zip(target.cpu().view(-1), preds.cpu().view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc, 'bacc': bacc})
            print_stats(args, epoch, num_samples, testloader, metrics)
    elapsed_time = time.time() - start_time
    print_summary(args, epoch, num_samples, metrics,elapsed_time, mode="Validation")
    return metrics, confusion_matrix


def initialize_from_saved_model(args):
    print('Training on saved model')
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    model, optimizer, epoch = load_model(args)

    train_loader = COVIDxDataset(mode='train', n_classes=args.classes, dataset_path=args.dataset,
                                 dim=(224, 224))
    # print(train_loader.)
    # ------ Class weigths for sampling and for loss function -----------------------------------
    labels = np.unique(train_loader.labels)
    # print(labels)
    class_weight = compute_class_weight('balanced', labels, train_loader.labels)
    class_weight = class_weight[::-1]
    # class_weight[2]=50
    # weights = torch.DoubleTensor(class_weight.copy())
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_loader.labels))
    if (args.cuda):
        class_weight = torch.from_numpy(class_weight.astype(float)).cuda()
    else:
        class_weight = torch.from_numpy(class_weight.astype(float))
    # print(class_weight.shape)
    # -------------------------------------------
    val_loader = COVIDxDataset(mode='test', n_classes=args.classes, dataset_path=args.dataset,
                               dim=(224, 224))
    # ------------------------------------------------------------------------------------
    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': 4}  # 'sampler' : sampler
    test_params = {'batch_size': args.batch_size,
                   'shuffle': False,
                   'num_workers': 4}
    # ------------------------------------------------------------------------------------------
    training_generator = DataLoader(train_loader, **train_params)
    val_generator = DataLoader(val_loader, **test_params)
    return model, optimizer, training_generator, val_generator, class_weight, epoch
