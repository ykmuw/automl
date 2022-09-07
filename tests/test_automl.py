# -*- coding: utf-8 -*-

import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(__file__) + "/../automl/")

from model import Dataset
from automl.automl import prepare_data, automl


def test_prepare_data_select_correct_features():

    dataset_id = 1
    target = "Survived"
    categories = ["Embarked", "Name", "Sex"]
    unused = ["PassengerId", "Cabin", "Name", "Ticket"]

    d = Dataset.query.get(dataset_id)
    if d:
        dataset = d.__dict__
        df = pd.read_csv(dataset["filepath"])
        cols = set(df.columns.values) - set([target]) - set(unused)
        
        X_train, X_test, Y_train, Y_test, features = prepare_data(dataset_id, target, categories, unused)
        assert set(features) == cols, "Correct columns seem not be selected."
    else:
        assert False, "Not found the dataset_id. Register a dataset via AutoML EMR UI."


def test_prepare_data_train_data_is_number():

    dataset_id = 1
    target = "Survived"
    categories = ["Embarked", "Name", "Sex"]
    unused = ["PassengerId", "Cabin", "Name", "Ticket"]

    d = Dataset.query.get(dataset_id)
    if d:
        X_train, X_test, Y_train, Y_test, features = prepare_data(dataset_id, target, categories, unused)

        err = 0
        for col_name, item in X_train.iteritems():
            if not (item.dtype == int or item.dtype == float or item.dtype == "uint8"):
                err += 1
        assert err == 0, "There are some items which are not numeric."
    else:
        assert False, "Not found the dataset_id. Register a dataset via AutoML EMR UI."


def test_automl():

    experiment_name = "test"
    dataset_id = 1
    target = "Survived"
    categories = ["Embarked", "Name", "Sex"]
    unused = ["Cabin", "Name", "Ticket"]
    # unused = ["Cabin", "Ticket"] failed

    d = Dataset.query.get(dataset_id)
    if d:
        assert isinstance(automl(experiment_name, dataset_id, target, categories, unused), int), "Don't return new experiement id. AutoML seems to fail."
    else:
        assert False, "Not found the dataset_id. Register a dataset via AutoML EMR UI."
 