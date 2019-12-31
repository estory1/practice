#%% [markdown]
# # Predict heart disease using:
# 1. sklearn
# 2. Spark + MLlib
# 3. TensorFlow
import os
import sys
import numpy as np
import pandas as pd 
import sklearn

#%%
filepath = os.path.expanduser("~") + "/Documents/syncable/home/dev/data_science/practice/classification/Heart_Disease-UCI/heart.csv"
df = pd.read_csv(filepath)
df.head()

#%%
df.describe()

#%% [markdown]
# ## sklearn

#%%
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
from sklearn.linear_model import LogisticRegressionCV

X = df[ list(set(df.columns) - set(["target"])) ]
y = df["target"]

clf = LogisticRegressionCV( 
        cv=10, 
        random_state=0, 
        multi_class="ovr", 
        n_jobs=8, 
        max_iter=1000000,
        verbose=True
    ).fit( X, y )

#%%
clf.predict(X[:-2])

#%%
clf.predict_proba( X[:-2] ).shape

#%%
clf.score(X, y)

#%%
from sklearn.model_selection import cross_val_score

lr_cv_f1 = cross_val_score( clf, X, y, scoring='f1', cv=5 )
pd.Series( lr_cv_f1 ).describe()

#%%
lr_cv_auc = cross_val_score( clf, X, y, scoring='roc_auc', cv=5 )
pd.Series( lr_cv_auc ).describe()

#%%
from sklearn.metrics import confusion_matrix

confusion_matrix( y, clf.predict(X) )

#%% [markdown]
### What we learned from the above:
###### 1) With n=303, m=13, the dataset is miniscule, so quick training is practically assured.
###### 2) Training is on non-standardized, non-centered data. ==> Model is sensitive to particular value ranges.
###### 3) Accuracy = 0.858; F1 (mean, SD) of 5 models = (0.857, 0.040); AUC (mean, SD) of 5 models = (0.898, 0.045).
###### 3.1) Accuracy only gives a point-estimate of model quality; no understanding of variation w.r.t. out-of-sample data.
###### 3.2) F1 gives equal weight to classification precision & recall. Mean is lowest, but SD is lower than AUC, suggesting at first glance that F1-judged models will perform most reliably (albeit marginally so, given the small SD delta between them).
###### 4) The trace of the confusion matrix shows relatively little model error.


#%% [markdown]
# ## Spark

#%%
# https://spark.apache.org/docs/2.1.0/ml-classification-regression.html#binomial-logistic-regression
# import findspark
# findspark.init()

from pyspark import SparkContext
# spark.stop()
spark = SparkContext(appName="heart")

# dfs = spark.read.format('com.databricks.spark.csv').load(filepath)

# just reuse the existing pandas DF...
from pyspark.sql import SQLContext
sqlContext = SQLContext(spark)
sdf = sqlContext.createDataFrame(df)
sdf.head(10)

#%%
# let's see summary stats on the input frame
sdf.describe().show()

#%%
# Do a random 80:20 train:test hold-out split.
strain, stest = sdf.randomSplit( [0.8, 0.2], seed=0 )
print( [ strain.count(), len(strain.columns), stest.count(), len(stest.columns) ] )

#%%
# What are the distributions of the train and test sets?
strain.groupBy("target").count().show()
stest.groupBy("target").count().show()

#%%
# https://docs.databricks.com/spark/latest/mllib/binary-classification-mllib-pipelines.html
# Spark needs all feature col values in a single vector in col1, and the label/target in col2. Weird, lame data-eng flow; I already have column vecs of my data...
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler

categoricalColumns = strain.columns
stages = [] # stages in our Pipeline
for categoricalCol in categoricalColumns:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    # encoder = OneHotEncoderEstimator(inputCol=categoricalCol + "Index", outputCol=categoricalCol + "classVec")
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]
# Convert label into label indices using the StringIndexer
label_stringIdx = StringIndexer(inputCol="target", outputCol="label")
stages += [label_stringIdx]
# Transform all features into a vector using VectorAssembler
assemblerInputs = list( set(strain.columns) - set(["target"]) )
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]
partialPipeline = Pipeline().setStages( stages )

#%%
# run the transforms pipeline on the training and testing sets
strain_preppedDataDF = partialPipeline.fit( strain ).transform( strain )
strain_preppedDataDF.head(10)
stest_preppedDataDF = partialPipeline.fit( stest ).transform( stest )
stest_preppedDataDF.head(10)

#%%
from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression

# config the LR algo
slr = SparkLogisticRegression( 
    labelCol="label",
    featuresCol="features",
    maxIter=1000000, 
    regParam=0.3, 
    elasticNetParam=0.8
    )

# train model
slrModel = slr.fit( strain_preppedDataDF )

# gen predictions on training and test sets
strain_predict = slrModel.transform( strain_preppedDataDF )
strain_predict.toPandas().head(5)
#%%
stest_predict  = slrModel.transform( stest_preppedDataDF )
stest_predict.toPandas().head(5)
# stest_predict  = slrModel.select( "target", "prediction" ).show(10)

#%% [markdown]
# #### Spark model eval: AUROC

from pyspark.ml.evaluation import BinaryClassificationEvaluator

seval = BinaryClassificationEvaluator( rawPredictionCol = "rawPrediction", labelCol = "target", metricName="areaUnderROC" )
print( "AUROC for training set: {}".format( seval.evaluate( strain_predict ) ) )
print( "AUROC for test set    : {}".format( seval.evaluate( stest_predict ) ) )

stest_predict.select( "target", "rawPrediction", "prediction", "probability" ).toPandas().head(5)     # WARNING: .toPandas() collects the DF to a single executor node - will blow-up on big data! But not for this tiny input file.

#%%
# #### Spark model eval: AUPR

from pyspark.ml.evaluation import BinaryClassificationEvaluator

seval = BinaryClassificationEvaluator( rawPredictionCol = "rawPrediction", labelCol = "target", metricName="areaUnderPR" )
print( "AUPR for training set: {}".format( seval.evaluate( strain_predict ) ) )
print( "AUPR for test set    : {}".format( seval.evaluate( stest_predict ) ) )

stest_predict.select( "target", "rawPrediction", "prediction", "probability" ).toPandas().head(5)     # WARNING: .toPandas() collects the DF to a single executor node - will blow-up on big data! But not for this tiny input file.



#%%
