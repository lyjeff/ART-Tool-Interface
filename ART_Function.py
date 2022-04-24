# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 16:19:31 2021

@author: csyu
"""

import torch
import os
import sys
import time
import pandas as pd

# import model
import ResNet
import googlenet
import VGG
import model as modl

# Load Model and Optimizer function


def Load_Model(filename, model, optimizer, device_type):
    PATH = os.path.dirname(os.path.abspath(sys.argv[0])) + "/"
    PATH_model = PATH + filename + ".model"
    PATH_optim = PATH + filename + ".optimizer"
    isload = False

    if os.path.isfile(PATH_model) and os.path.isfile(PATH_optim):
        model.load_state_dict(torch.load(
            PATH_model, map_location=torch.device(device_type)))
        optimizer.load_state_dict(torch.load(
            PATH_optim, map_location=torch.device(device_type)))
        model.eval()
        isload = True

    return model, optimizer, isload


def model_recon(select):

    if select == 'LeNet5':
        model = 1
    elif select == 'CNN':
        model = 2
    elif select == 'AlexNet':
        model = 3
    elif select == 'GoogleNet':
        model = 4
    elif select == 'VGG19':
        model = 5
    elif select == 'ResNext101':
        model = 6

    return model


def attack_recon(select):

    if select == 'FGSM':
        attack = 1
    elif select == 'BIM':
        attack = 2
    elif select == 'PGD':
        attack = 3
    elif select == 'C&W L2':
        attack = 4
    elif select == 'C&W Linf':
        attack = 5

    return attack

# select model to be attack


def model_select(select, num_classes: int = 10):

    #select = int(input('Please select Predict model (1:LeNet5, 2:CNN, 3:AlexNet, 4:GoogLeNet, 5:VGG19, 6:ResNeXt101):'))

    if select == 1:
        model = modl.LeNet5(num_classes=num_classes)
        text = 'LeNet5'
        print('Use LeNet5')
    elif select == 2:
        model = modl.CNN(num_classes=num_classes)
        text = 'CNN'
        print('Use CNN')
    elif select == 3:
        model = modl.AlexNet(num_classes=num_classes)
        text = 'AlexNet'
        print('Use AlexNet')
    elif select == 4:
        model = googlenet.GoogLeNet(num_classes=num_classes)
        text = 'GoogLeNet'
        print('Use GoogLeNet')
    elif select == 5:
        model = VGG.vgg19(num_classes=num_classes)
        text = 'VGG19'
        print('Use VGG19')
    elif select == 6:
        model = ResNet.resnext101_32x8d(num_classes=num_classes)
        text = 'ResNeXt101'
        print('Use ResNeXt101')

    return model, text, select

# select attack function


def attack_select(select):

    #select = int(input("Please select Attack function (1:FGSM, 2:BIM, 3:PGD, 4:C&W L2, 5:C&W Linf):"))

    if select == 1:
        attack_func = [True, False, False, False, False]
        print('Use FGSM')
        att_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    elif select == 2:
        attack_func = [False, True, False, False, False]
        print('Use BIM')
        att_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    elif select == 3:
        attack_func = [False, False, True, False, False]
        print('Use PGD')
        att_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    elif select == 4:
        attack_func = [False, False, False, True, False]
        print('Use C&W L2')
        att_range = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    elif select == 5:
        attack_func = [False, False, False, False, True]
        print('Use C&W Linf')
        att_range = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    else:
        # attack_func = [FGSM, BIM, PGD, C&W L2, C&W Linf]
        attack_func = [False, False, False, False, False]
        att_range = []

    return attack_func, att_range

# Save Data


def Save_data(attack_func, text, out_accuracy, attack_select, eps):
    if attack_func[0] == True:
        att_text = 'FGSM'
    if attack_func[1] == True:
        att_text = 'BIM'
    if attack_func[2] == True:
        att_text = 'PGD'
    if attack_func[3] == True:
        att_text = 'C&W_L2'
    if attack_func[4] == True:
        att_text = 'C&W_Linf'

    if attack_select == 1:
        att_mdl_text = 'LeNet5'
    elif attack_select == 2:
        att_mdl_text = 'CNN'
    elif attack_select == 3:
        att_mdl_text = 'AlexNet'
    elif attack_select == 4:
        att_mdl_text = 'GoogleNet'
    elif attack_select == 5:
        att_mdl_text = 'VGG19'
    elif attack_select == 6:
        att_mdl_text = 'ResNeXt101'

    Now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    context = [att_mdl_text, att_text, out_accuracy[0], eps, Now_time]

    try:
        Accuracy_file = pd.read_csv(text + '_' + att_text + '_Accuracy.csv')

    except:
        Accuracy_file = pd.DataFrame(columns=['Attack_Model',
                                              'Attack_Function', 'Accuracy', 'eps/conf', 'Time'])

    Accuracy_file = pd.concat(
        [Accuracy_file, pd.DataFrame([context], columns=Accuracy_file.columns)], ignore_index=True)
    Accuracy_file.to_csv(text + '_' + att_text +
                         '_Accuracy.csv', header=True, index=False)
