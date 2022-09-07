# -*- coding: utf-8 -*-

from datetime import datetime
from importlib import import_module
import json
import pytz
import pandas as pd
import shap

from sklearn.impute import KNNImputer

from model import db, Dataset, Experiment, Experiment_detail, Algo

import pyspark.pandas as ps
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

from spark.spark_util import get_spark_session


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
    train : pyspark.sql.dataframe.DataFrame
        train data
    test : pyspark.sql.dataframe.DataFrame
        test data
    target :str
        target name for prediction 
    features : list
        features name list for prediction model
    """

    dataset = Dataset.query.get(dataset_id).__dict__
    filepath = dataset["filepath"].replace('s3:', 's3a:')
    spark = get_spark_session()
    df = spark.read.parquet(filepath)
    train, test = df.randomSplit([0.8, 0.2], seed=4000)
    df_train = ps.DataFrame(train)
    df_test = ps.DataFrame(test)
    
    for u in unused:
        df_train = df_train.drop(u)
        df_test = df_test.drop(u)
        if u in set(categories):
            categories.remove(u)
    
    features = list(df_train.columns.values)
    features.remove(target)

    if categories:
        df_train = ps.get_dummies(df_train, columns=categories, drop_first=True)
        df_test = ps.get_dummies(df_test, columns=categories, drop_first=True)
        for c in set(df_train.columns.values)^set(df_test.columns.values):
            if c in df_train.columns.values:
                df_train = df_train.drop(c)
            if c in df_test.columns.values:
                df_test = df_test.drop(c)

    required_features = list(df_train.columns.values)
    if target in required_features:
        required_features.remove(target)
    assembler = VectorAssembler(inputCols=required_features, outputCol='features')
    transformed_train = assembler.transform(df_train.to_spark())
    transformed_test = assembler.transform(df_test.to_spark())

    return transformed_train, transformed_test, target, features


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


def ml(algo, train, test, target):
    """
    Train a specific machine learning model and evaluate statistical scores

    Parameters
    ----------
    algo : dict
        machine learning model
    train : pyspark.sql.dataframe.DataFrame
        train X data
    test : pyspark.sql.dataframe.DataFrame
        test X data

    Returns
    -------
    metrics : dict
        metrics to evaluate ML model
    """
    mod = import_module(algo["import_pkg"])
    clf = getattr(mod, algo["algo_class"])(featuresCol = "features", labelCol=target)
    if algo["params"]:
        clf.setParams(**json.loads(algo["params"]))

    model = clf.fit(train)

    evaluatorMulti = MulticlassClassificationEvaluator(labelCol=target, predictionCol="prediction")
    evaluator = BinaryClassificationEvaluator(labelCol=target, rawPredictionCol="prediction", metricName='areaUnderROC')

    predictionAndTarget = model.transform(test).select(target, "prediction")
    acc = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "accuracy"})
    weightedPrecision = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedPrecision"})
    weightedRecall = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedRecall"})
    auc = evaluator.evaluate(predictionAndTarget)

    # if algo["algo_type"] == 1:
    #     explainer = shap.LinearExplainer(
    #         clf, masker=shap.maskers.Impute(
    #             data=X_train), algorithm="linear")
    # elif algo["algo_type"] == 2:
    #     explainer = shap.TreeExplainer(clf)
    # elif algo["algo_type"] == 3:
    #     explainer = shap.KernelExplainer(clf.predict, X_train)

    # clf_shap_values = explainer.shap_values(X_test)
    # if isinstance(clf_shap_values, list):
    #     vs = abs(clf_shap_values[0]).sum(axis=0) / len(X_test)
    # else:
    #     vs = abs(clf_shap_values).sum(axis=0) / len(X_test)
    # out_shap = {
    #     key: val for key, val in zip(
    #         X_test.columns.values, vs)}

    metrics = {
        "accuracy": acc,
        "precision": weightedPrecision,
        "recall": weightedRecall,
        "auc": auc,
        #"shap": json.dumps(out_shap)
        }

    return metrics


def automl_spark(experiment_name, dataset_id, target, categories, unused):
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

    train, test, target, features = prepare_data(
        dataset_id, target, categories, unused)
    # X_train, X_test = impute_df(X_train, X_test)

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
        Algo.impl_type == 1, Algo.problem_type == 0).order_by(Algo.id).all()

    for a in algos:
        algo = a.__dict__
        metrics = ml(algo, train, test, target)

        new_experiment_detail = Experiment_detail(
            experiment_id=new_experiment_id,
            algo_id=algo["id"],
            auc=metrics["auc"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            accuracy=metrics["accuracy"],
            shap=0,
            #shap=metrics["shap"],
            created_date=current_time
        )

        db.session.add(new_experiment_detail)
    db.session.commit()

    return new_experiment_id
