
from datetime import datetime
import pytz

from model import db, Algo


if __name__ == "__main__":

    db.create_all()

    current_time = datetime.now(pytz.timezone('US/Pacific'))

    algo = Algo(
        id=0,
        impl_type=0,
        problem_type=0,
        algo_type=1,
        algo_name='Lasso Logistic Regression',
        import_pkg='sklearn.linear_model',
        algo_class='LogisticRegression',
        params='{"penalty":"l1", "C": 1, "solver": "liblinear", "max_iter": 10000}',
        created_date=current_time,
        last_update=current_time)
    db.session.add(algo)

    algo = Algo(
        id=1,
        impl_type=0,
        problem_type=0,
        algo_type=2,
        algo_name='Random Forest',
        import_pkg='sklearn.ensemble',
        algo_class='RandomForestClassifier',
        params='{"max_depth": 2}',
        created_date=current_time,
        last_update=current_time)
    db.session.add(algo)

    algo = Algo(
        id=2,
        impl_type=0,
        problem_type=0,
        algo_type=2,
        algo_name='Gradient Boosting',
        import_pkg='sklearn.ensemble',
        algo_class='GradientBoostingClassifier',
        params='{"n_estimators" : 100, "learning_rate":1.0, "max_depth":1}',
        created_date=current_time,
        last_update=current_time)
    db.session.add(algo)

    algo = Algo(
        id=3,
        impl_type=0,
        problem_type=0,
        algo_type=2,
        algo_name='LightGBM',
        import_pkg='lightgbm',
        algo_class='LGBMClassifier',
        params='',
        created_date=current_time,
        last_update=current_time)
    db.session.add(algo)

    algo = Algo(
        id=4,
        impl_type=0,
        problem_type=0,
        algo_type=1,
        algo_name='Logistic Regression',
        import_pkg='sklearn.linear_model',
        algo_class='LogisticRegression',
        params='{"max_iter": 10000}',
        created_date=current_time,
        last_update=current_time)
    db.session.add(algo)

    algo = Algo(
        id=5,
        impl_type=0,
        problem_type=0,
        algo_type=1,
        algo_name='Ridge Logistic Regression',
        import_pkg='sklearn.linear_model',
        algo_class='LogisticRegression',
        params='{"penalty":"l2", "C": 1, "solver": "liblinear", "max_iter": 10000}',
        created_date=current_time,
        last_update=current_time)
    db.session.add(algo)

    algo = Algo(
        id=6,
        impl_type=1,
        problem_type=0,
        algo_type=1,
        algo_name='Lasso Logistic Regression',
        import_pkg='pyspark.ml.classification',
        algo_class='LogisticRegression',
        params='{"maxIter": 1000, "regParam": 0.01, "elasticNetParam": 1}',
        created_date=current_time,
        last_update=current_time)
    db.session.add(algo)

    algo = Algo(
        id=7,
        impl_type=1,
        problem_type=0,
        algo_type=1,
        algo_name='Ridge Logistic Regression',
        import_pkg='pyspark.ml.classification',
        algo_class='LogisticRegression',
        params='{"maxIter": 1000, "regParam": 0.01, "elasticNetParam": 0}',
        created_date=current_time,
        last_update=current_time)
    db.session.add(algo)

    algo = Algo(
        id=8,
        impl_type=1,
        problem_type=0,
        algo_type=1,
        algo_name='Random Forest',
        import_pkg='pyspark.ml.classification',
        algo_class='RandomForestClassifier',
        params='{"numTrees": 10}',
        created_date=current_time,
        last_update=current_time)
    db.session.add(algo)

    algo = Algo(
        id=9,
        impl_type=1,
        problem_type=0,
        algo_type=1,
        algo_name='Gradient Boosting',
        import_pkg='pyspark.ml.classification',
        algo_class='GBTClassifier',
        params='',
        created_date=current_time,
        last_update=current_time)
    db.session.add(algo)

    db.session.commit()
