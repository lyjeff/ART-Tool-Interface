"""
Created on Tue Mar  2 15:38:15 2021

@author: csyu, jefflin

The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
# -*- coding: utf-8 -*-
# Pytorch 1.6 and CUDA 10.2


import os

import torch
import torch.nn as nn
import torch.optim as optim
from art.data_generators import PyTorchDataGenerator
from torch.utils.data import DataLoader

import ART_Function as AF
import ART_Tool
import Attack_Function as Att_F
import Data_Load as DL
import Prediction
import Train_classifier as TC
from argument_generator import Argument_Generator


def select_paramters(args):

    # Interface
    if args.interface == 1:
        args.predict_model, args.attack_model, args.attack_func, args.eps, args.dataset_path = ART_Tool.vp_start_gui()
        # parameter = ['LeNet5', 'LeNet5', 'FGSM', '0.5', './mnist/']
        args.predict_model, args.attack_model, args.attack_func = AF.model_recon(
            args.predict_model), AF.model_recon(args.attack_model), AF.attack_recon(args.attack_func)

    elif args.interface == 2:
        args.predict_model = int(input(
            "select the Predict model (1:LeNet5, 2:CNN, 3:AlexNet, 4:GoogLeNet, 5:VGG19, 6:ResNeXt101): "))
        args.attack_model = int(input(
            "select attack model (1:LeNet5, 2:CNN, 3:AlexNet, 4:GoogLeNet, 5:VGG19, 6:ResNeXt101): "))
        args.attack_func = int(
            input("select Attack function (1:FGSM, 2:BIM, 3:PGD, 4:C&W L2, 5:C&W Linf): "))
        args.eps = float(input(
            "input the eps or confidence (0.0 <= eps <= 1.0, 0 <= confidence <= 16): "))

        if args.dataset_path is None:
            args.dataset_path = input("input the dataset path: ")

    return args


def art_rgb(args):

    # Basic parameter
    torch.backends.cudnn.benchmark = False
    epoch_num = args.epochs
    batch_size = args.batch_size
    max_iter = args.max_iter
    eps = float(args.eps) if (args.attack_func not in [4, 5]
                              or args.conf is None) else args.conf
    clip_values = None

    # check cuda
    if not torch.cuda.is_available() or args.cuda == -1:
        device_type = 'cpu'
    else:
        device_type = f'cuda:{args.cuda}'

    # Load Dataset
    dataset_root = args.dataset_path
    print(f"Loading dataset from {dataset_root} ...")

    train_set = DL.MNISTDataset(os.path.join(dataset_root, "./train/"))
    test_set = DL.MNISTDataset(os.path.join(dataset_root, "./test/"))

    train_data_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False)

    # Create the Predict model
    model, text, select = AF.model_select(
        args.predict_model, train_set.nb_classes)

    # Select the Attack model
    Att_F.model_select(args.attack_model, train_set.nb_classes)

    # Select attack function
    attack_func, _ = AF.attack_select(args.attack_func)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    if args.optim == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr, momentum=args.momentum)

    # load model, optimizer
    model, optimizer, isload = AF.Load_Model(
        text, model, optimizer, device_type)

    data_Generator = PyTorchDataGenerator(
        iterator=train_data_loader, size=int(len(train_data_loader.dataset)), batch_size=batch_size)

    # Training classifier
    classifier = TC.Fit_classifier(
        model=model,
        clip_values=clip_values,
        criterion=criterion,
        optimizer=optimizer,
        data_Generator=data_Generator,
        nb_classes=train_set.nb_classes,
        isload=isload,
        epoch_num=epoch_num,
        device_type=device_type,
        text=text,
    )
    print('Finish training ...')

    print('Start testing the original test dataset ...')
    Prediction.classifier_predict(
        classifier=classifier,
        dataloader=test_data_loader,
        text=text,
        select=select
    )
    print('Finish testing the original test dataset ...')

    # Main
    att_classifier = Att_F.attack_classifier(
        epoch_num=epoch_num,
        clip_values=clip_values,
        data_Generator=data_Generator, nb_classes=train_set.nb_classes,
        select=args.attack_model,
        device_type=device_type,
        lr=args.lr, momentum=args.momentum,
    )
    out_accuracy = Att_F.attack(
        eps=eps, max_iter=max_iter, batch_size=batch_size,
        attack_func=attack_func, dataloader=test_data_loader,
        classifier=classifier, att_classifier=att_classifier,
        attack_select=args.attack_model, select=select
    )

    # Save Data
    AF.Save_data(attack_func, text, out_accuracy, args.attack_model, eps)


if __name__ == '__main__':

    argument_generator = Argument_Generator()
    args = argument_generator.generator()

    args = select_paramters(args)
    art_rgb(args)
