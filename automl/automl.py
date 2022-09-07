# -*- coding: utf-8 -*-

from datetime import datetime
from importlib import import_module
import json
import pytz
import pandas as pd
import shap

from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score

from model import db, Dataset, Experiment, Experiment_detail, Algo


def prepare_data(dataset_id, target, categories, unused):
    """
    prepare train/test dataset

    Parameters
    ----------
    dataset_id : int
        dataset_id to read csv
    target : str
        target name for prediction
    categories : list
        category variable name list for one hot encoding
    unused : list
        unused variable name list for prediction model

    Returns
    -------
    X_train : pandas.core.frame.DataFrame
        train X data
    X_test : pandas.core.frame.DataFrame
        test X data
    Y_train : pandas.core.frame.DataFrame
        train Y data
    Y_test : pandas.core.frame.DataFrame
        test Y data
    features : list
        features name list for prediction model
    """

    dataset = Dataset.query.get(dataset_id).__dict__
    df = pd.read_csv(dataset["filepath"])

    Y = df[target]
    for u in unused:
        df = df.drop(u, axis=1)
        if u in set(categories):
            categories.remove(u)

    features = df[df.columns[df.columns != target]].columns.values
    if categories:
        df = pd.get_dummies(df, columns=categories, drop_first=True)
    X = df[df.columns[df.columns != target]]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0)
    return X_train, X_test, Y_train, Y_test, features


def impute_df(X_train, X_test):
    """
    Complement missing values of pandas dataframe when column type is numeric and # of missing values > 0

    Parameters
    ----------
    X_train : pandas.core.frame.DataFrame
        train data with missing values
    X_test : pandas.core.frame.DataFrame
        test data with missing values

    Returns
    -------
    X_train : pandas.core.frame.DataFrame
        complemented train data
    X_test : pandas.core.frame.DataFrame
        complemented test data
    """

    impute_columns = []
    for col in X_train.columns.values:
        if X_train[col].dtype in ('int', 'float') and (
                sum(X_train[col].isnull()) > 0 or sum(X_test[col].isnull())) > 0:
            impute_columns.append(col)

    if impute_columns:
        imputer = KNNImputer(n_neighbors=2)
        df_filled = pd.DataFrame(
            imputer.fit_transform(
                X_train[impute_columns]),
            columns=impute_columns,
            index=X_train.index)
        X_train.update(df_filled)

        test_df_filled = pd.DataFrame(
            imputer.transform(
                X_test[impute_columns]),
            columns=impute_columns,
            index=X_test.index)
        X_test.update(test_df_filled)

    return X_train, X_test


def ml(algo, X_train, Y_train, X_test, Y_test):
    """
    Train a specific machine learning model and evaluate statistical scores

    Parameters
    ----------
    algo : dict
        machine learning model
    X_train : pandas.core.frame.DataFrame
        train X data
    X_test : pandas.core.frame.DataFrame
        test X data
    Y_train : pandas.core.frame.DataFrame
        train Y data
    Y_test : pandas.core.frame.DataFrame
        test Y data

    Returns
    -------
    metrics : dict
        metrics to evaluate ML model
    """
    mod = import_module(algo["import_pkg"])
    clf = getattr(mod, algo["algo_class"])()
    if algo["params"]:
        clf.set_params(**json.loads(algo["params"]))

    clf.set_params(**{"random_state": 0})

    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)
    Y_score = clf.predict_proba(X_test)[:, 1]

    out_acc = accuracy_score(y_true=Y_test, y_pred=Y_pred)
    out_precision = precision_score(y_true=Y_test, y_pred=Y_pred)
    out_recall = recall_score(y_true=Y_test, y_pred=Y_pred)
    out_auc = roc_auc_score(y_true=Y_test, y_score=Y_score)

    if algo["algo_type"] == 1:
        explainer = shap.LinearExplainer(
            clf, masker=shap.maskers.Impute(
                data=X_train), algorithm="linear")
    elif algo["algo_type"] == 2:
        explainer = shap.TreeExplainer(clf)
    elif algo["algo_type"] == 3:
        explainer = shap.KernelExplainer(clf.predict, X_train)

    clf_shap_values = explainer.shap_values(X_test)
    if isinstance(clf_shap_values, list):
        vs = abs(clf_shap_values[0]).sum(axis=0) / len(X_test)
    else:
        vs = abs(clf_shap_values).sum(axis=0) / len(X_test)
    out_shap = {
        key: val for key, val in zip(
            X_test.columns.values, vs)}

    metrics = {
        "accuracy": out_acc,
        "precision": out_precision,
        "recall": out_recall,
        "auc": out_auc,
        "shap": json.dumps(out_shap)}

    return metrics


def automl(experiment_name, dataset_id, target, categories, unused):
    """
    Run multiple ML models

    Parameters
    ----------
    experiment_name : str
        experimental name
    dataset_id : int
        dataset_id to prepare data
    target : str
        target name to prepare data
    categories : list
        category variable name list to prepare data
    unused : list
        unused variable name list to prepare data

    Returns
    -------
    new_experiment_id : int
        this exprimental id
    """

    X_train, X_test, Y_train, Y_test, features = prepare_data(
        dataset_id, target, categories, unused)
    X_train, X_test = impute_df(X_train, X_test)

    current_time = datetime.now(pytz.timezone('US/Pacific'))
    new_experiment = Experiment(
        experiment_name=experiment_name,
        target_name=target,
        features=", ".join(features),
        dataset_id=dataset_id,
        created_date=current_time
    )
    db.session.add(new_experiment)
    db.session.commit()
    new_experiment_id = new_experiment.id

    algos = Algo.query.filter(
        Algo.impl_type == 0, Algo.problem_type == 0).order_by(Algo.id).all()

    for a in algos:
        algo = a.__dict__
        metrics = ml(algo, X_train, Y_train, X_test, Y_test)

        new_experiment_detail = Experiment_detail(
            experiment_id=new_experiment_id,
            algo_id=algo["id"],
            auc=metrics["auc"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            accuracy=metrics["accuracy"],
            shap=metrics["shap"],
            created_date=current_time
        )

        db.session.add(new_experiment_detail)
    db.session.commit()

    return new_experiment_id
