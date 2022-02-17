# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:46:05 2021

@author: csyu
"""

from art.estimators.classification import PyTorchClassifier
import os


def Fit_classifier(model, clip_values, criterion, optimizer, data_Generator, nb_classes, isload, epoch_num,  device_type, text):

    # Create the ART classifier RGB
    classifier = PyTorchClassifier(
        model=model,
        clip_values=clip_values,
        loss=criterion,
        # optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=nb_classes,
        device_type=device_type
    )

    # Train the ART classifier
    if isload == False:
        print('Start training ...')
        classifier.fit_generator(data_Generator, nb_epochs=epoch_num)
        classifier.save(filename=text, path=os.path.dirname(
            os.path.abspath(__file__)))

    return classifier
