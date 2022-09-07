
![autoMLimage](images/automl_image.png)

# What is AutoML?
* The lack of comparison views for the output metrics that are generated by multiple machine learning (ML) models renders the determination of insights from datasets difficult for users. 
Generating ML models also requires iterative processes, covering the understanding of data to the tuning of model parameters.

* AutoML implements functions and storages that can accelerate the development of ML models. 
The selection of ML modeling, the modeling process itself, the user’s input, and ML modeling output are automated. 
Users can evaluate the output metrics generated by various ML models against one another in one summary view.
Output metrics help users select the most and least successful models for a given business scenario.
<p align="center">
<img src = "images/automl_discussion.png" width="400">
</p>


# AutoML
* While a machine learning(ML) project requires iterative proceses from understanding data to tuning model parameters in order to develop a optimal model for data, some tasks in the analysis process can be automated. AutoML implements functions to accelerate the process of developing the ML model.
<p align="center">
<img src = "images/automl_diagram.jpg" width="800">
</p>


# Output view and metrics
* Users can review their uploaded CSV files in the summary view. 
* Quality columns show them how input data look.
* After ML modeling, area under the curve (AUC), precision, recall, and accuracy values are displayed.
* For Spark in Local, SHapley Additive exPlanations (SHAP) is generated in histgram by clicking the ML model names, so that users can figire which category values impact the most in the ML modeling. SHAP is a method of explaining individual predictions. A positive SHAP value means a positive impact on prediction. In AutoML, SHAP values are converted to absolute values, and displayed in a histgram.
![shap](images/automl_shap.png)


# Users profile
* Our target users are those who want to analyze data and have the results represented in multiple ML models.
* To run the AutoML application, users should download required libraries and run Python modules (see `Install on your mac` section).


# Data sources
IMDb (an abbreviation of Internet Movie Database) is an online database of information related to films, television series, home videos, video games, and streaming content online – including cast, production crew and personal biographies, plot summaries, trivia, ratings, and fan and critical reviews. The earliest movie or TV show start year is 1888 and continue to be provided by 2024. IMDb is owned and operated by IMDb.com, Inc., a subsidiary of Amazon.
###
* IMDb dataset definitions are here: https://www.imdb.com/interfaces/ 
* IMDb files can be downloaded from here: https://datasets.imdbws.com/ 
For AutoML, title.basics.tsv.gz and title.ratings.tsv.gz are applied, then joined and modified in R for the AutoML input.
###
For testing AutoML, movies as a `titleType` and `startYear` between 2010 and 2020 are filtered and used from the movie dataset, and rating dataset is used as it is.
Each `primaryTitle` is categorized as one or more genres. New columns such as are added such as rateHighLow, voteHighLow, and rateGraterThan8 are added for the AutoML inference.

# R in dataset preparation
```
install.packages("tidyverse")
install.packages("readr")
install.packages("dplyr")
install.packages("scales")
install.packages("ggsci")
library(tidyverse)
library(ggplot2)
library(dplyr)
library(readr)
library(scales)
library(ggsci)

#################################################################################
# Prepare dataset to join basic_movie and rating between 2010 and 2022
#################################################################################

# Read dataset
movie_df <- read.csv("data/title.basics_movie.csv") 

# Unpivot movie dataset, and join rating dataset
movie_unpivot_df <- movie_df %>%
  filter(startYear != "\\N" & runtimeMinutes != "\\N") %>%
  separate_rows(genres, sep = ",") %>%
  filter(genres != "\\N") %>%
  filter(startYear >= 2010 & startYear <= 2022) %>%
  mutate(startYear = as.character(startYear)) %>%
  select(tconst, primaryTitle, isAdult, startYear, runtimeMinutes, genres)
#View(movie_unpivot_df)

# Remove unnecessary object to save RAM spaces
rm(list = c("movie_df"))

# Read dataset
rating_df <- read_tsv("data/title.ratings.tsv") 

# Join movie and ranking by tconst
movie_rating_df <- 
  left_join(movie_unpivot_df, rating_df, by = "tconst") %>%
  na.omit(numVotes) %>%
  na.omit(averageRating)
#View(movie_rating_df)

# Remove unnecessary object to save RAM spaces
rm(list = c("rating_df"))
rm(list = c("movie_unpivot_df"))

# Add greater than equal average rate as 1 and less than average rate as 0, and 
# greater than equal average number of vote as 1 ad less than average as 0, and
movie_rate_binary_df <- movie_rating_df %>%
  mutate(rateHighLow = ifelse(averageRating >= sum(averageRating * numVotes)/sum(numVotes), 1, 0)) %>% #[1] 6.860613
  mutate(voteHighLow = ifelse(numVotes >= mean(numVotes), 1, 0)) %>% #[1] 6028.553
  mutate(rateGreaterThan8 = ifelse(averageRating >= sum(8 * numVotes)/sum(numVotes), 1, 0))
#View(movie_rate_binary_df)

# Save df as csv
write.csv(movie_rate_binary_df, "data/imdb_movie_rating_2010_2022.csv", row.names = FALSE)

# Remove unnecessary object to save RAM spaces
rm(list = c("movie_rating_df"))
```

# How the dataset is applied in AutoML
Any comma-separated value (CSV) files can be uploaded onto AutoML. Sample dataset is prepared and stored as sample/imdb_movie_rating_2010_2022.csv.
A user’s uploaded files are stored in the data folder as 1.csv, 2.csv, and so on in Spark in Local.

# Variables for ML modeling
### Target variables
* The variable that AutoML recommends from a list of prospects is the target (predictor) variable selected for ML modeling.

### Category variables
* As a default, AutoML detects variables that cannot be used for ML modeling because they are non-numerical expressions.
* In addition to default detection, user's selected variables in `Category Variables` should be assigned numerical expressions 1 or 0, and one of them is dropped by get_dummies function with drop_first = True.
<img src = "images/automl_data_clearning_for_category_variables.png" width="800">

### Unused variables 
* By default, AutoML identifies variables that are inappropriate for use in ML modeling, such as those with an excessive number of missing values, too many unique values, and so on. Aside from carrying out default detection, users are required to select more unused variables on the basis of their decisions.

### Variables in evaluation metrics
TP: True Positive, TN: True Negative, FP: False Positive, and FN: False Negative
* AUC score: Receiver Operating Characteristic (ROC) = FPR*chi + TPR
    -> AUC score is below 0.5, the output prediction is not trusted 
* Precision score: TP/(TP+FP)
* Recall(Sensitivity) score: TP/(TP+FN)
* Accuracy score: (TP+TN)/(TP+TN+FP+FN)


# Install on your mac

```
git clone git@github.com:ykmuw/automl.git
cd automl
python3 -m venv .
source ./bin/activate
pip3 install --upgrade pip

# the following might be needed to fix pip install lightgbm on mac
# brew install libomp

pip3 install -r requirements.txt

python3.9 init_db.py (once input.db has been created, you don't need to run this again)
python3.9 app.py
# open 127.0.0.1:5000 on your browser
```

# App structure

```
.
├── data (destination to save csv)
├── images (images to refine README)
├── sample (sample data to try AutoML)
├── static (HTML components)
├── templates (HTML templates)
│   └── base.html
│   └── detail.html
│   └── experiment.html
│   └── index.html
│   └── variable.html
├── automl (func set)
│   └──automl_spark.py (support Spark on AutoML)
│   └──automl.py (support Spark in Local)
│   └──stat_util.py (convert non-numerical to numerical)
├── README.md (instructions how to run AutoML)
├── app.py (run web applicaitons)
├── init_db.py (create data framework for Spark on AutoM and Spark in Local)
├── input.db (generated by init_db.py)
├── model.py
└── requirements.txt
└── LICENSE
```
<img src = "images/application_diagram.png" width="800">


# AutoML Features
* Select sample/imdb_movie_rating_2010_2022.csv from user's local folder through `Choose File` button, and click `Import a new dataset`. 
<p align="center">
  <img src="images/automl_manage_dataset.png" width="800" hight="200">
</p>

* AutoML registers the dataset users uploaded, and display string-type as default in `Category Variables`, as well as string-type or if the value contains too many unique variables as default in `Unused Variables for Modeling`. 
<p align="center">
  <img src="images/automl_summary_file_uploaded.png" width="800" hight="700">
</p>

* AutoML provides users uploaded file data summary in `Summary` tab, and data samples in `Samples` tab. There are also useful comments available to check data quality (missing value, uniques, and outliers).
<p align="center">
  <img src="images/automl_summary_summary.png" width="800" hight="700">
  <img src="images/automl_summary_samples.png" width="800" hight="700">
</p>

* AutoML helps visualize data distribution per variable for users to understand data by clicking each variable link.
![summary](images/automl_summary_links.png)

* Applied category variables - skewed distribution
![histgram](images/automl_variable_link_genres.png)
* Applied category variables - normal distribution
![histgram](images/automl_variable_link_averageRating.png)
![histgram](images/automl_variable_link_startYear.png)
* Not applied category variables - too many unique values
![histgram](images/automl_variable_link_primaryTitle.png)

* Users will select a binary object variable in `Target` which users want to predict, as well as select appropriate values in `Category Variables` which users want to use them as category in ML modeling, and also select appropriate unused values in `Unused Variables for Modeling` like below, then click `Run`.
![histgram](images/automl_summary_with_greaterthan8.png)

# AutoML Evaluation Metrics (Output)
* AutoML fits multiple ML models automatically and provide users comparable statistical scores. Users can click each ML model link to observe output in graph.
![metrics](images/automl_evaluation_metrics.png)

* Display ML models in graph - LightGBM
![model_graph](images/automl_LightGBM.png)

* Display ML models in graph - Random Forest
![model_graph](images/automl_metrics_Ramdom_Forest.png)

* Display ML models in graph - Ridge Logistic Regression
![model_graph](images/automl_Ridge_Logistic_Regression.png)

# Algorithms
Here are some ML model diagrams and algorithms, and sharing the algorithm code sample used in AutoML.
<p align="center">
  <img src="images/automl_random_forest.jpg">
</p>
<p align="center">
  <img src="images/automl_gradient_boosting.jpg">
</p>
<p align="center">
  <img src="images/automl_logistic_regression.jpg">
</p>

# Future enhancement
* Increase the number of comparable ML models.
* Add functions to tune multiple parameters by grid search.
* Replace local to Apache Spark on EMR (Hadoop) on EC2 with S3.