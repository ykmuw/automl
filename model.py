# -*- coding: utf-8 -*-

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config.from_object('settings.Development')
# connect to DB by retrieving app values
db = SQLAlchemy(app)


class Experiment(db.Model):
    __tablename__ = 'experiment'
    id = db.Column(db.Integer, primary_key=True)
    experiment_name = db.Column(db.String(1000))
    target_name = db.Column(db.String(1000))
    features = db.Column(db.String(1000))
    dataset_id = db.Column(db.Integer, nullable=False)
    created_date = db.Column(db.DateTime, nullable=False)


class Experiment_detail(db.Model):
    __tablename__ = 'experiment_detail'
    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(db.Integer)
    algo_id = db.Column(db.Integer)
    auc = db.Column(db.Float, nullable=False)
    precision = db.Column(db.Float, nullable=False)
    recall = db.Column(db.Float, nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    shap = db.Column(db.String(1000))
    created_date = db.Column(db.DateTime, nullable=False)


class Algo(db.Model):
    __tablename__ = 'algo'
    id = db.Column(db.Integer, primary_key=True)
    impl_type = db.Column(db.Integer)  # 0: python 1: spark
    problem_type = db.Column(db.Integer)  # 0: classification 1: regression
    algo_type = db.Column(db.Integer)
    algo_name = db.Column(db.String(1000))
    import_pkg = db.Column(db.String(1000))
    algo_class = db.Column(db.String(100))
    params = db.Column(db.String(1000))
    created_date = db.Column(db.DateTime, nullable=False)
    last_update = db.Column(db.DateTime, nullable=False)


class Dataset(db.Model):
    __tablename__ = 'dataset'
    id = db.Column(db.Integer, primary_key=True)
    dataset_name = db.Column(db.String(1000), nullable=False)
    filepath = db.Column(db.String(1000))
    stored = db.Column(db.Integer)
    row_number = db.Column(db.Integer)
    column_number = db.Column(db.Integer)
    created_date = db.Column(db.DateTime, nullable=False)
    last_update = db.Column(db.DateTime, nullable=False)


class Dataset_summary(db.Model):
    __tablename__ = 'dataset_summary'
    dataset_id = db.Column(db.Integer, primary_key=True)
    var_id = db.Column(db.String(1000), primary_key=True)
    var_name = db.Column(db.String(1000))
    var_type = db.Column(db.Integer)  # 1:numeric, 2:string
    ds_count = db.Column(db.Integer)
    ds_unique = db.Column(db.Integer)
    ds_mean = db.Column(db.Float)
    ds_std = db.Column(db.Float)
    ds_min = db.Column(db.Float)
    ds_25 = db.Column(db.Float)
    ds_50 = db.Column(db.Float)
    ds_75 = db.Column(db.Float)
    ds_max = db.Column(db.Float)
