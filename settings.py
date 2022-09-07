import os

base_dir = os.path.dirname(__file__)


class Development(object):

    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(base_dir, 'input.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # for spark on amazon emr
    AWS_ACCESS_KEY_ID = ''
    AWS_SECRET_ACCESS_KEY = ''
    AWS_REGION = 'us-west-2'
    S3_BUCKET = 'automlinput'

    # for spark setting
    SPARK_DRIVER_MEMORY = '8g'
    SPARK_EXECUTOR_MEMORY = '4g'
