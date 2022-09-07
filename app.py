# -*- coding: utf-8 -*-

##local machine
from flask import render_template, request, redirect, abort
from werkzeug.utils import secure_filename

from datetime import datetime
import json
import pytz
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import plotly
import plotly.express as px


from model import app, db, Dataset, Dataset_summary, Experiment, Experiment_detail, Algo
from automl.automl import automl
from automl.stat_util import smirnov_grubbs

##spark version
from flask import jsonify
import boto3
import io
import uuid

import awswrangler as wr
from boto3.session import Session

from pyspark.sql.functions import col, countDistinct
from spark.spark_util import get_spark_session
from automl.automl_spark import automl_spark


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        
        datasets = Dataset.query.order_by(Dataset.id).all()

        return render_template('index.html', datasets=datasets)

    elif request.method == 'POST':

        file = request.files['inputFile']
        dataset_name = secure_filename(file.filename)
        file.filename = secure_filename(file.filename) 
        current_time = datetime.now(pytz.timezone('US/Pacific'))

        if request.form.get('stored') == "python":

            tmpdir = tempfile.mkdtemp()
            filepath = os.path.join(tmpdir, dataset_name)
            file.save(filepath)
            df = pd.read_csv(filepath)
            summary = df.describe(include='all')
            new_dataset = Dataset(dataset_name=dataset_name,
                                row_number=len(df),
                                stored=0,
                                column_number=len(df.columns),
                                created_date=current_time,
                                last_update=current_time
                                )

            db.session.add(new_dataset)
            db.session.commit()
            filepath = 'data/%s.csv' % (new_dataset.id)
            db.session.query(Dataset).filter(Dataset.id == new_dataset.id).update({"filepath": filepath})
            db.session.commit()
            df.to_csv(filepath, index=False)
            shutil.rmtree(tmpdir)
            for i, v in enumerate(summary.columns.values):
                new_ds_summary = Dataset_summary(
                    dataset_id=new_dataset.id,
                    var_id=i,
                    var_name=v,
                    var_type=1 if df.dtypes[v] in (
                        int,
                        float) else 2,
                    ds_count=int(
                        summary[v]["count"]),
                    ds_unique=len(df[v].unique()),
                    ds_mean=summary[v]["mean"],
                    ds_std=summary[v]["std"],
                    ds_min=summary[v]["min"],
                    ds_25=summary[v]["25%"],
                    ds_50=summary[v]["50%"],
                    ds_75=summary[v]["75%"],
                    ds_max=summary[v]["max"])
                db.session.add(new_ds_summary)
                db.session.commit()
        
        elif request.form.get('stored') == "spark":

            tmp = str(uuid.uuid4())

            session = Session(
                aws_access_key_id=app.config['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=app.config['AWS_SECRET_ACCESS_KEY'],
                region_name=app.config['AWS_REGION'],
            )

            tmpdir = tempfile.mkdtemp()
            filepath = os.path.join(tmpdir, dataset_name)
            file.save(filepath)
            
            df = pd.read_csv(filepath)
            s3_bucket = app.config['S3_BUCKET']
            s3_key = f'data/{tmp}/{file.filename}.parquet'
            wr.s3.to_parquet(
                df=df,
                path=f's3://{s3_bucket}/{s3_key}',
                boto3_session=session
            )
            del df
            shutil.rmtree(tmpdir)

#            s3 = boto3.client(
#                "s3",
#                aws_access_key_id=app.config['AWS_ACCESS_KEY_ID'],
#                aws_secret_access_key=app.config['AWS_SECRET_ACCESS_KEY'],
#                region_name=app.config['AWS_REGION']
#            )
            
#            s3_bucket = app.config['S3_BUCKET']
#            s3_key = f'data/{tmp}/{file.filename}'
#            response = s3.put_object(
#                Body=io.BufferedReader(file).read(),
#                Bucket=s3_bucket,
#                Key=s3_key
#            )
#
#            if response['ResponseMetadata']['HTTPStatusCode'] != 200:
#                return jsonify(message='uploading error occured'), 500
#
            spark = get_spark_session()
#            df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(f"s3a://{s3_bucket}/{s3_key}")
            df = spark.read.parquet(f"s3a://{s3_bucket}/{s3_key}")
            row_number = df.count()
            col_number = len(df.columns)

            new_dataset = Dataset(dataset_name=dataset_name,
                                row_number=row_number,
                                stored=1,
                                filepath=f's3://{s3_bucket}/{s3_key}',
                                column_number=col_number,
                                created_date=current_time,
                                last_update=current_time
                                )

            db.session.add(new_dataset)
            db.session.commit()
            
            summary = df.summary().toPandas()
            summary.index = summary["summary"]
            summary = summary.drop(columns=['summary'], axis=1)

            col_countdistinct = df.agg(*(countDistinct(col(c)).alias(c) for c in df.columns)).collect()[0].asDict()
            for i, v in enumerate(summary.columns.values):
                col_type = df.dtypes[i][1]
                new_ds_summary = Dataset_summary(
                    dataset_id=new_dataset.id,
                    var_id=i,
                    var_name=v,
                    var_type=1 if col_type in ('int', 'double') else 2,
                    ds_count=int(
                        summary[v]["count"]),
                    ds_unique=col_countdistinct[v],
                    ds_mean=summary[v]["mean"],
                    ds_std=summary[v]["stddev"],
                    ds_min=summary[v]["min"] if col_type in ('int', 'double') else None,
                    ds_25=summary[v]["25%"],
                    ds_50=summary[v]["50%"],
                    ds_75=summary[v]["75%"],
                    ds_max=summary[v]["max"] if col_type in ('int', 'double') else None)
                db.session.add(new_ds_summary)
                db.session.commit()
            
        return redirect('/detail/%s' % (new_dataset.id))


@app.route('/detail/<int:id>', methods=['GET', 'POST'])
def detail(id):

    dataset = Dataset.query.get(id)

    if not dataset:
        abort(404)
    
    row_number = dataset.__dict__["row_number"]
    
    dataset_summary = Dataset_summary.query.filter(
        Dataset_summary.dataset_id == id).order_by(
        Dataset_summary.var_id).all()
    
    experiments = db.session.query(
        Experiment,
        Experiment_detail,
        Algo) .filter(
        Experiment.dataset_id == id,
        Experiment.id == Experiment_detail.experiment_id,
        Algo.id == Experiment_detail.algo_id) .order_by(
            Experiment.id)
    row_number_column = db.func.row_number().over(
        partition_by=Experiment_detail.experiment_id,
        order_by=Experiment_detail.auc.desc()).label('row_number')
    experiments = experiments.add_columns(row_number_column)
    experiments = experiments.from_self().filter(row_number_column == 1).all()

    if request.method == 'GET':

        if dataset.__dict__["stored"] == 0:

            summaries = []
            select_vars = []
            for summary in dataset_summary:
                d = summary.__dict__
                
                quality = []
                missing = row_number - d['ds_count']
                if missing > 0:
                    quality.append("missing values(%s)" % (missing))
                if d['ds_unique'] > d['ds_count'] / 2 and d['var_type'] == 2:
                    quality.append("too many unique")
                
                # check outlier
                #if d['var_type'] == 1 and d['ds_unique'] > 2:
                #    df = pd.read_csv('data/%s.csv' % (dataset.id))
                #    if sum(
                #            (~np.isnan(smirnov_grubbs(df[d['var_name']], 0.01)))) > 0:
                #        quality.append("might be outliers")
                
                d_dict = {
                    "variable": '<a href="%s/variable/%s">%s</a>' %
                    (request.path,
                    d['var_name'],
                        d['var_name']),
                    "type": 'Numeric' if d['var_type'] == 1 else 'String',
                    "quality": ", ".join(quality),
                    "count": d['ds_count'],
                    "unique": d['ds_unique'],
                    "mean": '-' if d['ds_mean'] is None else d['ds_mean'],
                    "std": '-' if d['ds_std'] is None else d['ds_std'],
                    "min": '-' if d['ds_min'] is None else d['ds_min'],
                    "25%": '-' if d['ds_25'] is None else d['ds_25'],
                    "50%": '-' if d['ds_50'] is None else d['ds_75'],
                    "75%": '-' if d['ds_75'] is None else d['ds_75'],
                    "max": '-' if d['ds_max'] is None else d['ds_max']}

                select_var = {
                    "var_name": d['var_name'],
                    "var_type": 'Numeric' if d['var_type'] == 1 else 'String',
                    "count": d['ds_count'],
                    "unique": d['ds_unique'],
                }

                summaries.append(d_dict)
                select_vars.append(select_var)

            df = pd.DataFrame.from_records(summaries)

            experiment_best = []
            for e, d, a, _ in experiments:
                experiment = e.__dict__
                exp_detail = d.__dict__
                algo = a.__dict__
                d_dict = {
                    "id": '<a href="%s/experiment/%s">%s</a>' %
                    (request.path,
                    experiment["id"],
                        experiment["id"]),
                    "name": experiment["experiment_name"],
                    "target": experiment["target_name"],
                    "features": experiment["features"],
                    "best model": algo["algo_name"],
                    "best auc": exp_detail["auc"],
                    "precision": exp_detail["precision"],
                    "recall": exp_detail["recall"],
                    "accuracy": exp_detail["accuracy"]}
                experiment_best.append(d_dict)

            df0 = pd.DataFrame.from_records(experiment_best)

            dataset = dataset.__dict__
            df_sample = pd.read_csv('data/%s.csv' % (dataset["id"])).head(10)

            return render_template(
                'detail.html',
                table=df.to_html(
                    index=False,
                    classes='table table-sm table-bordered table-hover',
                    header='true',
                    escape=False),
                sample=df_sample.to_html(
                    index=False,
                    classes='table table-sm table-bordered table-hover',
                    header='true'),
                experiment=df0.to_html(
                    index=False,
                    classes='table table-sm table-bordered table-hover',
                    header='true',
                    escape=False),
                var_name=select_vars)

        elif dataset.__dict__["stored"] == 1:

            summaries = []
            select_vars = []
            for summary in dataset_summary:
                d = summary.__dict__

                quality = []
                missing = row_number - d['ds_count']
                if missing > 0:
                    quality.append("missing values(%s)" % (missing))
                if d['ds_unique'] > d['ds_count'] / 2 and d['var_type'] == 2:
                    quality.append("too many unique")

                # check outlier
                #if d['var_type'] == 1:
                #    df = pd.read_csv('data/%s.csv' % (dataset.id))
                #    if sum(
                #            (~np.isnan(smirnov_grubbs(df[d['var_name']], 0.01)))) > 0:
                #        quality.append("might be outliers")

                d_dict = {
                    "variable": '<a href="%s/variable/%s">%s</a>' %
                    (request.path,
                    d['var_name'],
                        d['var_name']),
                    "type": 'Numeric' if d['var_type'] == 1 else 'String',
                    "quality": ", ".join(quality),
                    "count": d['ds_count'],
                    "unique": d['ds_unique'],
                    "mean": '-' if d['ds_mean'] is None else d['ds_mean'],
                    "std": '-' if d['ds_std'] is None else d['ds_std'],
                    "min": '-' if d['ds_min'] is None else d['ds_min'],
                    "25%": '-' if d['ds_25'] is None else d['ds_25'],
                    "50%": '-' if d['ds_50'] is None else d['ds_75'],
                    "75%": '-' if d['ds_75'] is None else d['ds_75'],
                    "max": '-' if d['ds_max'] is None else d['ds_max']}

                select_var = {
                    "var_name": d['var_name'],
                    "var_type": 'Numeric' if d['var_type'] == 1 else 'String',
                    "count": d['ds_count'],
                    "unique": d['ds_unique'],
                }

                summaries.append(d_dict)
                select_vars.append(select_var)

            df = pd.DataFrame.from_records(summaries)

            experiment_best = []
            for e, d, a, _ in experiments:
                experiment = e.__dict__
                exp_detail = d.__dict__
                algo = a.__dict__
                d_dict = {
                    "id": '<a href="%s/experiment/%s">%s</a>' %
                    (request.path,
                    experiment["id"],
                        experiment["id"]),
                    "name": experiment["experiment_name"],
                    "target": experiment["target_name"],
                    "features": experiment["features"],
                    "best model": algo["algo_name"],
                    "best auc": exp_detail["auc"],
                    "precision": exp_detail["precision"],
                    "recall": exp_detail["recall"],
                    "accuracy": exp_detail["accuracy"]}
                experiment_best.append(d_dict)

            df0 = pd.DataFrame.from_records(experiment_best)

            dataset = dataset.__dict__
            filepath = dataset["filepath"].replace('s3:', 's3a:')
            spark = get_spark_session()

            df_sample = spark.read.parquet(filepath).sample(fraction=min(1.0, 10 / dataset["row_number"])).toPandas()

            # df_sample = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(filepath) \
            #    .sample(fraction=min(1.0, 10 / dataset["row_number"])).toPandas()

            return render_template(
                'detail.html',
                table=df.to_html(
                    index=False,
                    classes='table table-sm table-bordered table-hover',
                    header='true',
                    escape=False),
                sample=df_sample.to_html(
                    index=False,
                    classes='table table-sm table-bordered table-hover',
                    header='true'),
                experiment=df0.to_html(
                    index=False,
                    classes='table table-sm table-bordered table-hover',
                    header='true',
                    escape=False),
                var_name=select_vars)

    elif request.method == 'POST':

        experiment_name = request.form.get('experiment_name')
        target = request.form.get('target')
        categories = request.form.getlist('category')
        unused = request.form.getlist('unused')
        
        if dataset.__dict__["stored"] == 0:
            experiment_id = automl(experiment_name, id, target, categories, unused)
        elif dataset.__dict__["stored"] == 1:
            experiment_id = automl_spark(experiment_name, id, target, categories, unused)
        
        return redirect('/detail/%s/experiment/%s' % (id, experiment_id))


@app.route('/detail/<int:dataset_id>/variable/<string:var_name>')
def variable(dataset_id, var_name):

    d = Dataset.query.get(dataset_id)

    if not d:
        abort(404)

    dataset = d.__dict__

    if dataset["stored"] == 0:
        df = pd.read_csv(dataset["filepath"])
        fig = px.histogram(df, x=var_name, marginal="box")
        fig.update_layout(bargap=0.2)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    elif dataset["stored"] == 1:
        spark = get_spark_session()
        filepath = dataset["filepath"].replace('s3:', 's3a:')
        df = spark.read.parquet(filepath)
        df = df.sample(fraction=min(1.0, 100000/dataset["row_number"])).toPandas()
        fig = px.histogram(df, x=var_name, marginal="box")
        fig.update_layout(bargap=0.2)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('variable.html', var_name=var_name, graphJSON=graphJSON)


@app.route('/detail/<int:dataset_id>/experiment/<int:exp_id>')
def experiment(dataset_id, exp_id):

    dataset = Dataset.query.get(dataset_id)

    if not dataset:
        abort(404)

    if request.method == 'GET':
        experiment_name = Experiment.query.get(
            exp_id).__dict__['experiment_name']
        experiments = Experiment_detail.query.filter(
            Experiment_detail.experiment_id == exp_id).order_by(
            Experiment_detail.auc.desc()).all()
        results = []

        algos = Algo.query.all()
        algo_list = {}
        for a in algos:
            algo = a.__dict__
            algo_list[algo["id"]] = {
                "name": algo["algo_name"], "params": algo["params"]}

        for experiment in experiments:
            e = experiment.__dict__
            results.append({"model": '<a href="%s/model/%s">%s</a>' % (exp_id,
                                                                       e["id"],
                                                                       algo_list[e["algo_id"]]["name"]),
                            "parameters": algo_list[e["algo_id"]]["params"],
                            "auc": e["auc"],
                            "precision": e["precision"],
                            "recall": e["recall"],
                            "accuracy": e["accuracy"]})

        df = pd.DataFrame.from_records(results)

        return render_template(
            'experiment.html',
            name=experiment_name,
            result=df.to_html(
                index=False,
                classes='table table-sm thead-light table-bordered table-hover',
                header='true',
                escape=False))


@app.route('/detail/<int:dataset_id>/experiment/<int:exp_id>/model/<int:model_id>')
def model_result(dataset_id, exp_id, model_id):

    dataset = Dataset.query.get(dataset_id)

    if not dataset:
        abort(404)

    impl_type = Dataset.query.get(dataset_id).__dict__["stored"]

    if request.method == 'GET':
        experiment_name = Experiment.query.get(
            exp_id).__dict__['experiment_name']
        experiments = Experiment_detail.query.filter(
            Experiment_detail.experiment_id == exp_id).order_by(
            Experiment_detail.auc.desc()).all()
        results = []


        algos = Algo.query.filter(
        Algo.impl_type == impl_type, Algo.problem_type == 0).order_by(Algo.id).all()
        #algos = Algo.query.all()
        algo_list = {}
        for a in algos:
            algo = a.__dict__
            algo_list[algo["id"]] = {
                "name": algo["algo_name"], "params": algo["params"]}

        for experiment in experiments:
            e = experiment.__dict__
            results.append({"model": '<a href="%s">%s</a>' % (e["id"],
                                                              algo_list[e["algo_id"]]["name"]),
                            "parameters": algo_list[e["algo_id"]]["params"],
                            "auc": e["auc"],
                            "precision": e["precision"],
                            "recall": e["recall"],
                            "accuracy": e["accuracy"]})

        df = pd.DataFrame.from_records(results)

        shaps = pd.DataFrame.from_dict(json.loads(Experiment_detail.query.get(model_id).__dict__["shap"]), orient='index', columns=["shap"])
        fig = px.bar(shaps.sort_values("shap", ascending=False))
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template(
            'experiment.html',
            name=experiment_name,
            result=df.to_html(
                index=False,
                classes='table table-sm thead-light table-bordered table-hover',
                header='true',
                escape=False),
            graphJSON=graphJSON)


@app.route('/delete/<int:id>')
def delete(id):
    dataset = Dataset.query.get(id)
    db.session.delete(dataset)
    db.session.commit()
    return redirect('/')


if __name__ == "__main__":

    app.run(debug=app.config['DEBUG'])
