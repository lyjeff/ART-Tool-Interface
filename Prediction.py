# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:32:44 2021

@author: csyu, jefflin
"""

import numpy as np
import torch
from tqdm import tqdm

# import Data_Load as DL

# Evaluate the ART classifier on benign test examples


def classifier_predict(classifier, dataloader, text, select):

    # predictions = []
    # for x_test_one in tqdm(x_test):
    #     x_test_one = DL.TestDataset(x_test_one[0], select)
    #     predictions_test = classifier.predict([x_test_one])
    #     predictions.append(predictions_test[0])

    correct = 0

    for x_imgs, labels in tqdm(dataloader):
        predictions_test = classifier.predict(x_imgs)
        _, preds = torch.from_numpy(predictions_test).max(1)
        _, labels = labels.max(1)
        correct += preds.eq(labels).sum()

    # predictions = np.array(predictions)
    # accuracy = np.sum(np.argmax(predictions, axis=1) ==
    #                   np.argmax(y_test, axis=1)) / len(y_test)

    accuracy = float(correct) / len(dataloader.dataset)

    print("\nAccuracy on benign test examples: {}%".format(accuracy * 100))
    np.savetxt(text + '_origin_Accuracy.csv', [accuracy*100], delimiter=",")
