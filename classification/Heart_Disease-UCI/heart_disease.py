#%% [markdown]
# # Predict heart disease using:
# 1. sklearn
# 2. Spark + MLlib
# 3. PyTorch
#
### Why?
# 1. Illustrate same solution, different tech.
# 2. Surface differences and limitations. Useful as a technologist for implementation capability breadth, but also some disciplines require calculation using multiple technologies.
# 3. Python+pandas+sklearn makes most ML jobs relatively easy.
from functools import reduce
import os
import sys
import pprint
import numpy as np
import pandas as pd 
import sklearn
import pyspark

print( "module versions: " + str([ pyspark.__version__ , sklearn.__version__ , pd.__version__ , np.__version__ ]) )

pp = pprint.PrettyPrinter(indent=2)

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)
pd.set_option("display.expand_frame_repr", True)


#%% [markdown]
### Shared training parameters
#%%
random_seed = 0
cv_folds = 5
threads = 10
max_iter = 1000000
elasticnet_l1l2_regularization_alpha = 0.3  # In both sklearn and Spark, 0 ==> Ridge, 1 ==> LASSO. Given that LASSO shrinks feature coefficients (doing i.e. weighting features; https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c), and I've already few (13) features, I'll bias this term towards Ridge.

#%%
filepath = "heart.csv"
df = pd.read_csv(filepath)
df.head()

#%%
# summary stats
df.describe()

#%%
# 1) what's our class distribution? 2) what are some useful distributional facts about each class?
df.groupby(by=["target"]).describe()





#%% [markdown]
# ## sklearn

#%%
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
from sklearn.linear_model import LogisticRegressionCV
# from sklearn.linear_model import ElasticNetCV

X = df[ list(set(df.columns) - set(["target"])) ]
y = df["target"]

# cfg sklearn's logistic regression algo to use ElasticNet regularization: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html#sklearn.linear_model.ElasticNetCV
clf = LogisticRegressionCV( 
        # training params
        random_state=random_seed, 
        max_iter=max_iter,
        multi_class='ovr',      # do binary classification
        penalty = 'elasticnet',
        solver = 'saga',        # "Elastic-Net penalty is only supported by the saga solver." - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
        l1_ratios = [ elasticnet_l1l2_regularization_alpha ],
        # validation params
        cv=cv_folds, 
        # infrastructure params
        n_jobs=threads, 
        verbose=True
    ).fit( X, y )

#%%
clf.predict(X[:-2])

#%%
clf.predict_proba( X[:-2] ).shape

#%%
clf.score(X, y)

#%%
# score classifier performance using the F1 metric
from sklearn.model_selection import cross_val_score

lr_cv_f1 = cross_val_score( clf, X, y, scoring='f1', cv=cv_folds )
pd.Series( lr_cv_f1 ).describe()

#%%
# score classifier performance using the AUC metric
lr_cv_auc = cross_val_score( clf, X, y, scoring='roc_auc', cv=cv_folds )
pd.Series( lr_cv_auc ).describe()

#%%
from sklearn.metrics import confusion_matrix

confusion_matrix( y, clf.predict(X) )

#%% [markdown]
### What we learned via modeling with sklearn:
###### 1. With n=303, m=13, the dataset is miniscule, so quick training is practically assured.
###### 2. Training is on non-standardized, non-centered data. ==> Model is sensitive to particular value ranges.
###### 3. Accuracy = 0.848; F1 (mean, SD) of 5 models = (0.851, 0.054); AUC (mean, SD) of 5 models = (0.886, 0.048).
###### 3.1. Accuracy only gives a point-estimate of model quality - there's no description of variation w.r.t. out-of-sample data.
###### 3.2. F1 gives equal weight to classification precision & recall. Mean is lower, and SD is higher than AUC, suggesting at first glance that an AUC-judged model will be more reliable than an F1-judged model (albeit marginally so, given the small SD delta between them).
###### 4. The trace of the confusion matrix shows relatively little model error.





#%% [markdown]
# ## Spark logistic regression, round 1
#%%
# import findspark
# findspark.init()

#%%
# https://spark.apache.org/docs/2.1.0/ml-classification-regression.html#binomial-logistic-regression
# from pyspark import SparkContext
from pyspark.sql import SparkSession

if "spark" in dir() and spark is not None:
    spark.stop()
# spark = SparkContext(appName="heart")
spark = SparkSession.builder \
    .master(f"local[{threads}]").appName("heart").getOrCreate()

# dfs = spark.read.format('com.databricks.spark.csv').load(filepath)

# just reuse the existing pandas DF...
# from pyspark.sql import SQLContext
# sqlContext = SQLContext(spark)
# sdf = sqlContext.createDataFrame(df)
sdf = spark.createDataFrame(df)
sdf.head(5)

#%%
# let's see summary stats on the input frame
sdf.describe().show()

#%%
# Do a random 80:20 train:test hold-out split.
# strain, stest = sdf.randomSplit( [0.8, 0.2], seed=random_seed )
# print( [ strain.count(), len(strain.columns), stest.count(), len(stest.columns) ] )

#%%
# What are the distributions of the train and test sets?
# strain.groupBy("target").count().show()
# stest.groupBy("target").count().show()

#%%
# https://docs.databricks.com/spark/latest/mllib/binary-classification-mllib-pipelines.html
# Spark needs all feature col values in a single vector in col1, and the label/target in col2.
# Weird, lame data-eng flow; I already have column vecs of my data in `sdf`, as indicated in my DF shape array after train-test split...
from pyspark.ml import Pipeline
# Spark 2.4: from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

stages = [] # stages in our Pipeline
for categoricalCol in sdf.columns:
    # Category Indexing with StringIndexer: https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/StringIndexer.html
    #
    # Similarly to H2O's categorical Vec implementation (http://docs.h2o.ai/h2o/latest-stable/h2o-core/javadoc/water/fvec/CategoricalWrappedVec.html),
    # this maps n String instances to n int-typed array index values,
    # each associated to the same 1 instance of the String.
    #
    # Various attempts at Spark documentation (even official: https://spark.apache.org/docs/latest/ml-features.html#stringindexer)
    # don't explain WHY they do this, but:
    #
    # 1) it's a generally effective strategy to reduce memory usage.
    #
    # So, given 4 bytes/int and 2 bytes/char (the default in Java (https://docs.oracle.com/javase/tutorial/java/nutsandbolts/datatypes.html), to which Scala is translatable, in which Spark was written),
    # this reduces Spark memory usage for any String composed of > 2 chars (i.e. most of them).
    #
    # 2) The label representation within Spark is thus rendered problem-independent.
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    # encoder = OneHotEncoderEstimator(inputCol=categoricalCol + "Index", outputCol=categoricalCol + "classVec")
    # Spark v2.4: encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    # Spark v3.0: API changed: https://stackoverflow.com/a/59926851
    encoder = OneHotEncoder(dropLast=False, inputCol=stringIndexer.getOutputCol(), outputCol=categoricalCol + "classVec")
    # model = encoder.fit()
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]
# Convert label into label indices using the StringIndexer
stages += [ StringIndexer(inputCol="target", outputCol="label") ]
# Transform all features into a vector using VectorAssembler
assemblerInputs = list( set(sdf.columns) - set(["target"]) )
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]
partialPipeline = Pipeline().setStages( stages )

# #%%
# # run the transforms pipeline on the training and testing sets
# strain_preppedDataDF = partialPipeline.fit( strain ).transform( strain )
# strain_preppedDataDF.head(5)
# #%%
# stest_preppedDataDF = partialPipeline.fit( stest ).transform( stest )
# stest_preppedDataDF.head(5)

#%%
from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# config the LR algo: https://spark.apache.org/docs/2.2.0/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegression
slr = SparkLogisticRegression( 
    # training params
    labelCol = "label",
    featuresCol = "features",
    elasticNetParam = elasticnet_l1l2_regularization_alpha,
    # validation params
    # (none?)
    # infrastructure params
    maxIter=max_iter
    )

# # train model
# slrModel = slr.fit( strain_preppedDataDF )

# # gen predictions on training and test sets
# strain_predict = slrModel.transform( strain_preppedDataDF )
# strain_predict.toPandas().head(5)
# #%%
# stest_predict  = slrModel.transform( stest_preppedDataDF )
# stest_predict.toPandas().head(5)

# #%% [markdown]
# # #### Spark model hold-out eval: AUROC

# from pyspark.ml.evaluation import BinaryClassificationEvaluator

# seval = BinaryClassificationEvaluator( rawPredictionCol = "rawPrediction", labelCol = "target", metricName="areaUnderROC" )
# print( "AUROC for training set: {}".format( seval.evaluate( strain_predict ) ) )
# print( "AUROC for test set    : {}".format( seval.evaluate( stest_predict ) ) )  # TODO: only do this once, else we're prone to overfit...

# stest_predict.select( "target", "rawPrediction", "prediction", "probability" ).toPandas().head(5)     # WARNING: .toPandas() collects the DF to a single executor node - will blow-up on big data! But not for this tiny input file.

# #%%
# # #### Spark model hold-out eval: AUPR

# from pyspark.ml.evaluation import BinaryClassificationEvaluator

# seval = BinaryClassificationEvaluator( rawPredictionCol = "rawPrediction", labelCol = "target", metricName="areaUnderPR" )
# print( "AUPR for training set: {}".format( seval.evaluate( strain_predict ) ) )
# print( "AUPR for test set    : {}".format( seval.evaluate( stest_predict ) ) )      # TODO: only do this once, else we're prone to overfit...

# stest_predict.select( "target", "rawPrediction", "prediction", "probability" ).toPandas().head(5)     # WARNING: .toPandas() collects the DF to a single executor node - will blow-up on big data! But not for this tiny input file.

#%% [markdown]
### Now run cross-validation on the Spark classifier.

#%%
# First, we need to run the pipeline on the whole dataframe to do Spark-specific feature matrix transforms.
sdf_preppedDataDF = partialPipeline.fit( sdf ).transform( sdf )
sdf_preppedDataDF.head(5)


#%%
# example: https://medium.com/@dhiraj.p.rai/logistic-regression-in-spark-ml-8a95b5f5434c
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# https://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=crossvalidator
sParamGrid = ParamGridBuilder().addGrid( slr.maxIter , [0, 1, 5, 10] ).build()

scvModel = CrossValidator( 
    # training params
    estimator=slr,
    estimatorParamMaps=sParamGrid, 
    # validation params
    evaluator=BinaryClassificationEvaluator(
        metricName="areaUnderROC"    # only 2 metric options for Spark BinaryClassificationEvaluator are AUCROC or AUPR; why no F-Beta, etc.?: https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.evaluation.BinaryClassificationEvaluator
    ),
    numFolds=cv_folds,
    # infrastructure params
    parallelism=threads,
    collectSubModels=True    # new in Spark 2.4. Docs warn of OOM if models are large, but that's unlikely here.
    ).fit( sdf_preppedDataDF )   # k-fold crossvalidation is usually done on the whole dataset, not training or hold-out testing...

#%%
# see the CV-generated models ensemble averages
scvModel.avgMetrics[0]  # the mean AUROC across all crossvalidated Spark LR models.

#%%
# Use i.e. model.summary() to get summary metrics: https://stackoverflow.com/a/54936081
# NOTE: The "bestModel" from CrossValidator is defined as "the model with the highest average cross-validation metric across folds": https://spark.apache.org/docs/latest/api/java/index.html?org/apache/spark/ml/tuning/CrossValidatorModel.html
pp.pprint( list( zip( ["accuracy", "AUROC", "totalIterations", "TPR by label", "FPR by label"] , [ 
    scvModel.bestModel.summary.accuracy,
    scvModel.bestModel.summary.areaUnderROC,
    scvModel.bestModel.summary.totalIterations,
    scvModel.bestModel.summary.truePositiveRateByLabel,
    scvModel.bestModel.summary.falsePositiveRateByLabel ]) ) )
#%%
# Print and scatterplot the F1-vs-probability threshold curves. Also useful for proba threshold selection by F1 score, so I sort descending to illustrate this.
scvModel.bestModel.summary.fMeasureByThreshold.orderBy('F-Measure', ascending=[0]).show()
scvModel.bestModel.summary.fMeasureByThreshold.toPandas().plot()
#%%
# Print and scatterplot the precision-recall curves...
scvModel.bestModel.summary.pr.orderBy('recall', ascending=[0]).show()  # prefer recall, since it accounts for FN in denominator, and FN in heart disease is expensive (death).
scvModel.bestModel.summary.pr.toPandas().plot()

#%%
# Print and scatterplot the ROC curves...
scvModel.bestModel.summary.roc.show()
scvModel.bestModel.summary.roc.toPandas().plot()


#%%
# for smiles, what does eval using AUPR return?
scvModel = CrossValidator( 
    # training params
    estimator=slr,
    estimatorParamMaps=sParamGrid, 
    # validation params
    evaluator=BinaryClassificationEvaluator(
        metricName="areaUnderPR"    # only 2 metric options for Spark BinaryClassificationEvaluator are AUCROC or AUPR; why no F-Beta, etc.?: https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.evaluation.BinaryClassificationEvaluator ; Scala source: https://github.com/apache/spark/blob/v2.4.3/mllib/src/main/scala/org/apache/spark/ml/evaluation/BinaryClassificationEvaluator.scala#L86-L89
    ),
    numFolds=cv_folds,
    # infrastructure params
    parallelism=threads,
    collectSubModels=True    # new in Spark 2.4. Docs warn of OOM if models are large, but that's unlikely here.
    ).fit( sdf_preppedDataDF )   # k-fold crossvalidation is usually done on the whole dataset, not training or hold-out testing...
scvModel.avgMetrics[0]      # the mean AUPR across all crossvalidated Spark LR models.

#%%
# Use i.e. model.summary() to get summary metrics: https://stackoverflow.com/a/54936081
# NOTE: The "bestModel" from CrossValidator is defined as "the model with the highest average cross-validation metric across folds": https://spark.apache.org/docs/latest/api/java/index.html?org/apache/spark/ml/tuning/CrossValidatorModel.html
pp.pprint( list( zip( ["accuracy", "AUROC", "totalIterations", "TPR by label", "FPR by label"] , [ 
    scvModel.bestModel.summary.accuracy,
    scvModel.bestModel.summary.areaUnderROC,
    scvModel.bestModel.summary.totalIterations,
    scvModel.bestModel.summary.truePositiveRateByLabel,
    scvModel.bestModel.summary.falsePositiveRateByLabel ]) ) )
#%%
# Print and scatterplot the F1-vs-probability threshold curves. Also useful for proba threshold selection by F1 score, so I sort descending to illustrate this.
scvModel.bestModel.summary.fMeasureByThreshold.orderBy('F-Measure', ascending=[0]).show()
scvModel.bestModel.summary.fMeasureByThreshold.toPandas().plot()
#%%
# Print and scatterplot the precision-recall curves...
scvModel.bestModel.summary.pr.orderBy('recall', ascending=[0]).show()  # prefer recall, since it accounts for FN in denominator, and FN in heart disease is expensive (death).
scvModel.bestModel.summary.pr.toPandas().plot()
#%%
# Print and scatterplot the ROC curves...
scvModel.bestModel.summary.roc.show()
scvModel.bestModel.summary.roc.toPandas().plot()


#%% [markdown]
### What we learned via modeling with Spark + MLLib:
###### 1. Observed example that the train and test sets, if randomly selected, can have label distributions that are biased differently (train having more 1s, test having more 0s). ==> Training set isn't representative of the testing set --> likely model unreliability & relatively lower accuracy on the test set. ==> Altogether arguments for using k-fold crossvalidation, not (only) hold-out testing.
###### 2. AUC in the Spark model is so substantially lower than in the sklearn model that I'm nearly certain I've done something wrong. Let's see if I can get a more similar result?

#%% [markdown]
### Spark logistic regression, round 2

#%%
# try new params
sParamGrid = ParamGridBuilder() \
    .addGrid( slr.maxIter , [0, 1, 5, 10] ) \
    .addGrid( slr.elasticNetParam, [ 0.1, 0.3, 0.5, 0.7, 0.9 ] ) \
    .build()

# redefine the model
slr = SparkLogisticRegression( 
    # training params
    labelCol = "label",
    featuresCol = "features",
    elasticNetParam = elasticnet_l1l2_regularization_alpha,
    # validation params
    # (none?)
    # infrastructure params
    maxIter=max_iter
    )

# retrain and cross-validate the model
scvModel = CrossValidator( 
    # training params
    estimator=slr,
    estimatorParamMaps=sParamGrid, 
    # validation params
    evaluator=BinaryClassificationEvaluator(
        metricName="areaUnderROC"    # only 2 metric options for Spark BinaryClassificationEvaluator are AUCROC or AUPR; why no F-Beta, etc.?: https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.evaluation.BinaryClassificationEvaluator ; Scala source: https://github.com/apache/spark/blob/v2.4.3/mllib/src/main/scala/org/apache/spark/ml/evaluation/BinaryClassificationEvaluator.scala#L86-L89
    ),
    numFolds=cv_folds,
    # infrastructure params
    parallelism=threads,
    collectSubModels=True    # new in Spark 2.4. Docs warn of OOM if models are large, but that's unlikely here.
    ).fit( sdf_preppedDataDF )   # k-fold crossvalidation is usually done on the whole dataset, not training or hold-out testing...
scvModel.avgMetrics[0]      # the mean AUROC across all crossvalidated Spark LR models.

#%%
# Use i.e. model.summary() to get summary metrics: https://stackoverflow.com/a/54936081
# NOTE: The "bestModel" from CrossValidator is defined as "the model with the highest average cross-validation metric across folds": https://spark.apache.org/docs/latest/api/java/index.html?org/apache/spark/ml/tuning/CrossValidatorModel.html
pp.pprint( list( zip( ["accuracy", "AUROC", "totalIterations", "TPR by label", "FPR by label"] , [ 
    scvModel.bestModel.summary.accuracy,
    scvModel.bestModel.summary.areaUnderROC,
    scvModel.bestModel.summary.totalIterations,
    scvModel.bestModel.summary.truePositiveRateByLabel,
    scvModel.bestModel.summary.falsePositiveRateByLabel ]) ) )
#%%
# Print and scatterplot the F1-vs-probability threshold curves. Also useful for proba threshold selection by F1 score, so I sort descending to illustrate this.
scvModel.bestModel.summary.fMeasureByThreshold.orderBy('F-Measure', ascending=[0]).show()
scvModel.bestModel.summary.fMeasureByThreshold.toPandas().plot()
#%%
# Print and scatterplot the precision-recall curves...
scvModel.bestModel.summary.pr.orderBy('recall', ascending=[0]).show()  # prefer recall, since it accounts for FN in denominator, and FN in heart disease is expensive (death).
scvModel.bestModel.summary.pr.toPandas().plot()
#%%
# Print and scatterplot the ROC curves...
scvModel.bestModel.summary.roc.show()
scvModel.bestModel.summary.roc.toPandas().plot()

#%% [markdown]
### What we learned in Spark logistic regression, round 2:
###### 1. AUROC increased, but only by ~0.015.
###### 2. Training time was much higher than in round 1, due to grid-search over a larger parameter space, including a larger number of iterations.
###### 3. Accuracy dramatically improved, from 0.69 to 0.85.
###### 4. Average AUROC in `avgMetrics[0]` increased very dramatically, from 0.5 to 0.89.
###### 5. In short, trying different weights in the ElasticNet alpha (regularization) term, and more training iterations, yielded very substantial improvements.
###### 6. We also see that the downward slope of the `threshold` curve in the F-Measure (F1) chart looks much steeper earlier in the x-axis now.
###### 





# %% [markdown]
### PyTorch logistic regression, round 1

#%%
# starting with this as a guide: https://medium.com/biaslyai/pytorch-linear-and-logistic-regression-models-5c5f0da2cb9

import torch
from torch.autograd import Variable
from torch.nn import functional as F

from sklearn.model_selection import train_test_split

print( "module versions: " + str([ torch.__version__ ]) )


# # TODO: split df rows into a hold-out test set (or better: do k-fold CV)
# torch_trn_X, torch_tst_X, torch_trn_y, torch_tst_y = train_test_split( df[ list(set(df.columns) - set(["target"])) ] , df["target"] , test_size=0.2, random_state = random_seed )
# torch_trn = np.zeros( ( torch_trn_X.shape[0] , torch_trn_X.shape[1] + 1 ) )
# torch_trn[ : , :-1 ] = torch_trn_X
# torch_trn[ : ,  -1 ] = torch_trn_y
# print( torch_trn.shape )
# torch_trn
# #%%
# torch_trn = torch.from_numpy( torch_trn )  # copy from the the dataset already in RAM, loaded by Pandas...
# torch_tst = torch.from_numpy( df.values )  # copy from the the dataset already in RAM, loaded by Pandas...

# torch_trn_n = len( torch_trn[ : , 0 ] ) 
# torch_trn_m = len( torch_trn[ 0 , : ] )
# print( "torch_train shape: {}x{}:".format( torch_trn_n , torch_trn_m ) )
# print( torch_trn )

# # %%
# # explore PyTorch usage a little.
# print( torch_trn[0] )     # first row
# print( torch_trn[:,0] )   # first col (age)


# %% [markdown]
### First step: min-max scale the features...
#%%
from sklearn.preprocessing import MinMaxScaler
df_norm = pd.DataFrame( MinMaxScaler().fit_transform(df) , columns = df.columns )
df_norm.describe()
# %%
df_norm_trn_X, df_norm_tst_X, df_norm_trn_y, df_norm_tst_y = train_test_split( 
    df_norm[ list(set(df_norm.columns) - set(["target"])) ] , 
    df_norm["target"] ,
    test_size=0.2, 
    random_state = random_seed )
torch_trn = np.zeros( ( df_norm_trn_X.shape[0] , df_norm_trn_X.shape[1] + 1 ) )
torch_trn[ : , :-1 ] = df_norm_trn_X
torch_trn[ : ,  -1 ] = df_norm_trn_y
print( torch_trn.shape )
torch_trn
#%%
torch_trn = torch.from_numpy( torch_trn )  # copy from the the dataset already in RAM, loaded by Pandas...
torch_tst = torch.from_numpy( df_norm.values )  # copy from the the dataset already in RAM, loaded by Pandas...

torch_trn_n = len( torch_trn[ : , 0 ] ) 
torch_trn_m = len( torch_trn[ 0 , : ] )
print( "torch_train shape: {}x{}:".format( torch_trn_n , torch_trn_m ) )
print( torch_trn )

# %%
# explore PyTorch usage a little.
print( torch_trn[0] )     # first row
print( torch_trn[:,0] )   # first col (age)


# %%
# PyTorch is a neural net modeling lib, and logistic regression can be modeled using NN constructs. We must define our own class for logistic regression...
class TorchLogisticRegression( torch.nn.Module ):
    def __init__(self):
        super(TorchLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear( torch_trn_m - 2 , 1 )     # matrix must represent the *transposed* dimensions of 1 sample: some row vector x in matrix X, excluding the target column.
    
    def forward( self, x ):
        y_pred = F.sigmoid( self.linear(x) )                 # could use other activation functions here - Tanh, ReLU, etc.
        # print( self.linear(x) )
        # y_pred = F.softmax( self.linear(x) )                 # could use other activation functions here - Tanh, ReLU, etc.
        # y_pred = F.log_softmax(self.linear(x) )
        return y_pred

torch_model = TorchLogisticRegression()

# also need to define a loss function... the tutorial uses Binary Cross-Entropy. Try it for now, but why not some other, more common criterion, like RMSE, AUC, F-Beta, etc.?
torch_criterion = torch.nn.BCELoss( size_average=True, reduce="sum" )  # wrt reduce: "sum" yields smaller loss than "mean" and None.

# need an optimizer, of course... the tutorial chose ordinary SGD.
torch_learning_rate = 0.01
torch_optimizer = torch.optim.SGD( torch_model.parameters() , lr=torch_learning_rate )

# %%
# now we train the model!
torch_n_epochs = 10000
for epoch in range( torch_n_epochs ):
    
    torch_model.train()             # train the model
    torch_optimizer.zero_grad()     # start with gradients at 0

    # we're using feedforward neural net structure, so we compute a forward pass.
    # NOTE: Without .float(), I get 'RuntimeError: Expected object of scalar type Float but got scalar type Double for argument #4 'mat1''.
    y_pred = torch_model( torch_trn[ : , 0 : torch_trn_m - 2 ].float() )  # -1 for 0-based indexing, -1 because the last column contains the target.

    # print( "shape of y_pred: [{}, {}]".format( torch_n , torch_m - 1 ] ) ) )

    # calc loss on this training epoch.
    # loss = torch_criterion( y_pred , torch_trn[ : , torch_trn_m - 1 ].float() )
    loss = torch_criterion( y_pred , torch_trn[ : , torch_trn_m - 1 ].unsqueeze(1).float() )  # PyTorch 1.6 -> 1.7: API changed s.t. my [242] matrix must be "unsqueezed" to [242,1]: https://stackoverflow.com/a/57802321 and https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
    if epoch % (torch_n_epochs/10) == 0: print( "loss (epoch: {}): {}".format( epoch, loss ) )

    # model parameter update, aka backward pass (NN update on errors (backpropagation))
    loss.backward()
    torch_optimizer.step()

# %%
# Now let's try predicting using our trained model.
torch_tst_X = np.zeros( ( 1 , df_norm_tst_X.shape[1] - 1 ) )
torch_tst_X = torch.from_numpy( np.array( df_norm_tst_X.iloc[ 0, : 0:-1 ] ) )    # take first row, exclude target col
torch_tst_y = torch.from_numpy( np.array( df_norm_tst_X.iloc[ 0, -1 ] ) )    # take first row, only target col
print("torch_tst_X.shape: " + str(torch_tst_X.shape))
print("torch_tst_X: " + str(torch_tst_X))
print("torch_tst_y: " + str(torch_tst_y))

print()
torch_model_pred = torch_model( torch_tst_X.float() )  # `.float()` (unless you `.eval()` the model first! Then `.float()` is not needed.) because else "RuntimeError: expected scalar type Double but found Float": https://stackoverflow.com/a/56741419
print( "Given input sample X[0] ({}), model outputs y_pred = {} against sample's y_target class = {}".format( 
    torch_tst_X , torch_model_pred , torch_tst_y ) )
print()

from torchinfo import summary  # https://github.com/TylerYep/torchinfo
print("Model summary:")
print( summary( torch_model , verbose=2 ) )

# %%
# Now let's predict each of the samples in the test set.
torch_tst_X = np.zeros( ( 1 , df_norm_tst_X.shape[1] - 1 ) )
torch_tst_X = torch.from_numpy( np.array( df_norm_tst_X.iloc[ : , : 0:-1 ] ) )    # take first row, exclude target col
torch_tst_y = torch.from_numpy( np.array( df_norm_tst_X.iloc[ : , -1 ] ) )    # take first row, only target col
print("torch_tst_X.shape: " + str(torch_tst_X.shape))
# print("torch_tst_X: " + str(torch_tst_X))
# print("torch_tst_y: " + str(torch_tst_y))

print()
torch_model_pred = torch_model( torch_tst_X.float() )  # `.float()` (unless you `.eval()` the model first! Then `.float()` is not needed.) because else "RuntimeError: expected scalar type Double but found Float": https://stackoverflow.com/a/56741419
print( "Given input sample X[0:5], excluding the rest ({}), model outputs y_pred = {} against sample's y_target class = {}".format( 
    torch_tst_X[0:5] , torch_model_pred , torch_tst_y ) )
# %%
