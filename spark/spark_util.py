
from flask import current_app as app

import findspark
findspark.init()
findspark.add_packages(["org.apache.hadoop:hadoop-aws:3.2.0"])

from pyspark import SparkConf
from pyspark.sql import SparkSession


def get_spark_session():

    conf = SparkConf()

    conf.set('spark.driver.memory', app.config['SPARK_DRIVER_MEMORY'])
    conf.set('spark.executor.memory', app.config['SPARK_EXECUTOR_MEMORY'])

    conf.set('fs.s3a.aws.credentials.provider', "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    conf.set('fs.s3a.access.key', app.config['AWS_ACCESS_KEY_ID'])
    conf.set('fs.s3a.secret.key', app.config['AWS_SECRET_ACCESS_KEY'])

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark
