# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:03:05 2021

@author: csyu
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from tqdm import tqdm

from torchvision import transforms
import art.attacks.evasion as art_att

import ART_Function as AF
import Train_classifier as TC

# import model
import ResNet
import googlenet
import VGG
import model as modl


def attack_function(dataloader, classifier, attack_Func, attack_select, select):

    # predictions = []

    # Data transform
    # data_transform_Resize = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.ToTensor(),
    # ])

    # data_transform = transforms.Compose([
    #     transforms.Resize(28),
    #     transforms.ToTensor(),
    # ])

    # data_transform_PIL = transforms.Compose([
    #     transforms.ToPILImage(),
    # ])

    correct = 0

    for x_imgs, labels in tqdm(dataloader):
        # Transform Image
        # x_img = Image.fromarray(x_test_one[0]*255).convert('RGB')

        '''
        if attack_select < 3:
            x_img = np.array(data_transform(x_img))
        else:
            x_img = np.array(data_transform_Resize(x_img))
        '''

        # x_img = np.array(data_transform_Resize(x_img))

        # Generate Attack Image
        x_test_att = attack_Func.generate(x=np.array(x_imgs))

        '''
        if select < 3 and attack_select >= 3:
            x_test_att = np.swapaxes(np.swapaxes(np.array(x_test_att[0]),0,1),1,2)
            x_test_att = data_transform_PIL(np.uint8(x_test_att*255))
            x_test_att = [np.array(data_transform(x_test_att))]
        elif select >= 3 and attack_select < 3:
            x_test_att = np.swapaxes(np.swapaxes(np.array(x_test_att[0]),0,1),1,2)
            x_test_att = data_transform_PIL(np.uint8(x_test_att*255))
            x_test_att = [np.array(data_transform_Resize(x_test_att))]
        '''

        # Predict Attack Image
        # predictions_test = classifier.predict(x_test_att)
        # predictions.append(predictions_test[0])
        predictions_test = classifier.predict(x_test_att)
        _, preds = torch.from_numpy(predictions_test).max(1)
        _, labels = labels.max(1)
        correct += preds.eq(labels).sum()

    # predictions = np.array(predictions)
    # accuracy = np.sum(np.argmax(predictions, axis=1) ==
    #                   np.argmax(y_test, axis=1)) / len(y_test)

    accuracy = float(correct) / len(dataloader.dataset)
    return accuracy

# Generate adversarial test examples


def attack(eps, max_iter, batch_size, attack_func, dataloader, classifier, att_classifier, attack_select, select):

    out_accuracy = []
    verbose = False

    # FGSM
    if attack_func[0] == True:
        print('Running FGSM, eps = ', eps)
        attack_FGSM = art_att.FastGradientMethod(
            estimator=att_classifier, eps=eps, batch_size=batch_size)
        accuracy = attack_function(
            dataloader, classifier, attack_FGSM, attack_select, select)
        print("\nAccuracy on FGSM adversarial test examples: {}%".format(accuracy * 100))
        out_accuracy.append(accuracy * 100)

    # BIM
    if attack_func[1] == True:
        print('Running BIM, eps = ', eps)
        attack_BIM = art_att.BasicIterativeMethod(
            estimator=att_classifier, eps=eps, max_iter=max_iter, batch_size=batch_size, verbose=verbose)
        accuracy = attack_function(
            dataloader, classifier, attack_BIM, attack_select, select)
        print("\nAccuracy on BIM adversarial test examples: {}%".format(accuracy * 100))
        out_accuracy.append(accuracy * 100)

    # PGD
    if attack_func[2] == True:
        print('Running PGD, eps = ', eps)
        attack_PGD = art_att.ProjectedGradientDescentPyTorch(
            estimator=att_classifier, eps=eps, max_iter=max_iter, batch_size=batch_size, verbose=verbose)
        accuracy = attack_function(
            dataloader, classifier, attack_PGD, attack_select, select)
        print("\nAccuracy on PGD adversarial test examples: {}%".format(accuracy * 100))
        out_accuracy.append(accuracy * 100)

    # C&W L2
    if attack_func[3] == True:
        print('Running C&W L2, conf = ', eps)
        attack_CW_L2 = art_att.CarliniL2Method(
            classifier=att_classifier, confidence=eps, max_iter=max_iter, batch_size=batch_size, verbose=verbose)
        accuracy = attack_function(
            dataloader, classifier, attack_CW_L2, attack_select, select)
        print("\nAccuracy on C&W L2 adversarial test examples: {}%".format(
            accuracy * 100))
        out_accuracy.append(accuracy * 100)

    # C&W Linf
    if attack_func[4] == True:
        print('Running C&W Linf, conf = ', eps)
        attack_CW_Linf = art_att.CarliniLInfMethod(
            classifier=att_classifier, confidence=eps, max_iter=max_iter, batch_size=batch_size, verbose=verbose)
        accuracy = attack_function(
            dataloader, classifier, attack_CW_Linf, attack_select, select)
        print("\nAccuracy on C&W Linf adversarial test examples: {}%".format(
            accuracy * 100))
        out_accuracy.append(accuracy * 100)

    return out_accuracy


def attack_classifier(epoch_num, clip_values, data_Generator, nb_classes, select, device_type, lr, momentum):

    model, text = model_select(select, num_classes=nb_classes)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # load model, optimizer
    model, optimizer, isload = AF.Load_Model(
        text, model, optimizer, device_type)

    classifier = TC.Fit_classifier(
        model=model,
        clip_values=clip_values,
        criterion=criterion,
        optimizer=optimizer,
        data_Generator=data_Generator,
        nb_classes=nb_classes,
        isload=isload,
        epoch_num=epoch_num,
        device_type=device_type,
        text=text,
    )

    return classifier

# select model to be attack


def model_select(select, num_classes: int = 10):

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

    return model, text
