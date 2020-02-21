#%% [markdown]
# # Predict suicidality using:
# 1. sklearn
# 2. Spark + MLlib
# 3. TensorFlow
#
# source:
#  Suicide Rates Overview 1985 to 2016 | Kaggle
#  https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016


#%%
import os
import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

import pyspark
from pyspark import SparkContext
import pprint

print( "module versions: " + str([ pyspark.__version__ , sklearn.__version__ , pd.__version__ , np.__version__ ]) )

pp = pprint.PrettyPrinter(indent=2)

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)
pd.set_option("display.expand_frame_repr", True)

random_seed = 0
cv_folds = 5
threads = 8
max_iter = 1000000
elasticnet_l1l2_regularization_alpha = 0.3  # In both sklearn and Spark, 0 ==> Ridge, 1 ==> LASSO. Given that LASSO shrinks feature coefficients (doing i.e. weighting features; https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c), and I've already few (13) features, I'll bias this term towards Ridge.
train_split_pct = 0.8

#%%
filepath = os.path.expanduser("~") + "/Documents/syncable/home/dev/data_science/practice/classification/Suicide-Kaggle_russellyates88/suicide-rates-overview-1985-to-2016.zip"
df = pd.read_csv(filepath)
df = df.dropna()
df.head()

#%%
df_cols_numeric = df.describe().columns
print("df_cols_numeric: " + str(df_cols_numeric))
df.describe()

#%% [markdown]
### Correlation matrix...
#%%
dfn = df[ df_cols_numeric ]
dfn.corr()

#%%
# let's make that prettier, with labeled axes...
def plot_square_mat( df, n, cols ):
    f = plt.figure( figsize=(15,15) )
    plt.matshow( df )
    plt.xticks( range( n ), cols, fontsize=8, rotation=45)
    plt.yticks( range( n ), cols, fontsize=8, rotation=45)
    plt.title("Correlation matrix", fontsize=14)
    plt.show()

#%%
plot_square_mat( dfn.corr() , dfn.shape[1] , dfn.columns  )

#%%
# this is kind of a neat at-a-glance viz of feature correlation, but seems old, assumes a Spark DF, and doesn't control for reproducibility in the random sampling, so adapt...: https://towardsdatascience.com/building-a-linear-regression-with-pyspark-and-mllib-d065c3ba246a
def plot_scatter_mat( df: pd.DataFrame, rand_seed: int ):
    numeric_features = [ c for c in df.columns if df.dtypes[ c ] == 'int' or df.dtypes[ c ] == 'double']
    sampled_data = df[ numeric_features ].sample(replace=False, frac=0.8, random_state=rand_seed )
    axs = pd.plotting.scatter_matrix(sampled_data, figsize=(10, 10))
    n = len(sampled_data.columns)
    for i in range(n):
        v = axs[i, 0]
        v.yaxis.label.set_rotation(0)
        v.yaxis.label.set_ha('right')
        v.set_yticks(())
        h = axs[n-1, i]
        h.xaxis.label.set_rotation(90)
        h.set_xticks(())

plot_scatter_mat( df, random_seed )

#%% [markdown]
### Covariance matrix
# Don't use pandas' `DataFrame.cov()`: it calculates covariance over time-series data. This dataset isn't time-series.
#%%
from sklearn.covariance import EmpiricalCovariance

cov = EmpiricalCovariance().fit( dfn )
cov.covariance_   # covariance matrix
#%%
cov.location_  # estimated mean

#%%
pd.DataFrame( cov.covariance_ , columns=dfn.columns )

#%%
print([ len( cov.covariance_[0] ) , len( cov.covariance_ ) ])
plot_square_mat( pd.DataFrame( cov.covariance_ , columns=dfn.columns ) , len( cov.covariance_ ) , dfn.columns )

#%% [markdown]
#### Oh....
# A population rate strongly covaries with population? Stunning find!
#
# Let's throw out `suicides/100k pop` and try again:

#%%
covcols = [ c for c in dfn.columns if "population" not in c and "gdp_per_capita ($)" not in c ]
plot_square_mat( pd.DataFrame( EmpiricalCovariance().fit( dfn[ covcols ] ).covariance_ , columns= covcols ) , len( covcols ) , covcols )


#%% [markdown]
# Not seeing the use of a covariance matrix here... Moving on.



#%% [markdown]
### Make a convenient regressor function...
#%%
def regress( df, y_col, x_cols ):
    X, Y = get_training_set(df, y_col, x_cols)

    #%%
    from sklearn.model_selection import KFold
    from sklearn import linear_model
    from sklearn.metrics import r2_score, mean_squared_error
    import matplotlib.pyplot as plt
    from sklearn.feature_selection import f_regression, mutual_info_regression, SelectPercentile, SelectKBest

    kf = KFold( n_splits=10, shuffle=True, random_state=random_seed )
    scores=[]
    models=[]

    for train_row_idx, test_row_idx in kf.split( X ):
        X_train = X.iloc[ train_row_idx ]
        Y_train = Y.iloc[ train_row_idx ]
        X_test  = X.iloc[ test_row_idx ]
        Y_test  = Y.iloc[ test_row_idx ]
        assert len(X_train) == len(Y_train) and len(X_test) == len(Y_test)
        
        lr = linear_model.LinearRegression()
        lr.fit( X_train, Y_train )
        models.append( lr )

        Y_pred = lr.predict( X_test )
        score = lr.score( X_test, Y_test )  # returns R^2 value
        scores.append( score )

        print()
        print( "Coefs: " + str( dict(zip( X.columns, lr.coef_ ) ) ))
        print([ "R^2: ", r2_score( Y_test, Y_pred ), "MSE: ", mean_squared_error( Y_test, Y_pred ) ])

        # # variables importance analysis
        # mi_arr = mutual_info_regression( X_test, Y_test, random_state = random_seed )
        # print( "Mutual information: " + str( dict(zip( X.columns, mi_arr )) ) )

        (F_scores, p_values) = f_regression( X_test, Y_test )
        print("F scores: " + str( dict(zip( X.columns, F_scores )) ) )
        print("p-values: " + str( dict(zip( X.columns, p_values )) ) )

    print()
    print( "scores (R^2): " + str( scores ) )
    print("scores (R^2) summary:")
    df_scores = pd.DataFrame( scores )
    print( df_scores.describe() )
    return df_scores

def get_training_set(df, y_col, x_cols):
    Y = df[y_col]
    X = df[ list( set(x_cols) - set([y_col]) ) ]
    # print([ np.shape(X), np.shape(Y) , list(X.columns) ])
    return X, Y

#%% [markdown]
### Regress rate on all other numerics, using k-fold crossvalidation.
#%% 
regress( df, "suicides/100k pop", df_cols_numeric )

#%% [markdown]
### Sooooooo... population and # suicides provide the most information about a suicides-per-capita dependent variable? Brilliant deduction, Watson!!

### Let's regress a different attr: "gdp_per_capita ($)"

#%%
regress( df, "gdp_per_capita ($)", df_cols_numeric )

#%% [markdown]
### Much better model. But compute is trivial here; what can we predict best?
#%%
max_mean_score = -1
best_var = None
for c in df_cols_numeric:
    print()
    print()
    print()
    print("*"*80)
    print("***** For y=" + c + " *****")
    print("*"*80)
    df_scores = regress( df, c, df_cols_numeric )
    print( df_scores.describe() )
    mean = df_scores.describe().transpose()["mean"].values
    print("mean:" + str(mean))
    if mean > max_mean_score:
        max_mean_score = mean
        best_var = c

#%%
print()
print("***** DATA FISHING REVEALS THE TARGET WE CAN PREDICT BEST, AND THE MEAN R^2 OF ITS 10-MODEL ENSEMBLE, ARE...: *****")
print([ best_var, max_mean_score ])

#%% [markdown]
### How about regularization? What about parsimony?
#%%
# Run an ElasticNet regression: linear regression with L1 and L2 regularizers as priors.
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
import itertools

y_at_max = None
x_at_max = None
r2_at_max = -1
for y in df_cols_numeric:
    for x in sorted( list( itertools.combinations( list( set(df_cols_numeric) - set([y]) ) , 2 ) ) +\
        list( itertools.combinations( list( set(df_cols_numeric) - set([y]) ) , 3 ) ) +\
        list( itertools.combinations( list( set(df_cols_numeric) - set([y]) ) , 4 ) ) ):
        X, Y = get_training_set( df, y, x )
        # hold out a small test set for scoring
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=random_seed )

        lr = ElasticNetCV( cv=10, random_state=random_seed )
        lr.fit( X_train, Y_train )

        r2 = lr.score( X_test, Y_test )
        print( "R^2 for "+str(y)+"="+str(x)+": " + str( r2 ) )
        if r2 > r2_at_max:
            y_at_max = y
            x_at_max = x
            r2_at_max = r2

#%%
print( "Best of all ElasticNet models: {}={} @ {}".format( y_at_max, x_at_max, r2_at_max ) )

#%% [markdown]
### Okay, running ElasticNet over many combinations of features is sort of humorous...
#
# This is because ElasticNet automatically downweights weaker features; it's i.e. differentiable (continuous, real-valued) feature selection.




#%% [markdown]
# ## Spark linear regression, model 1: data prep, regress suicide rate on raw features.

#%%
if "spark" in dir() and spark is not None:
    spark.stop()
spark = SparkContext(appName="heart")

# just reuse the existing pandas DF...
from pyspark.sql import SQLContext
sqlContext = SQLContext(spark)
sdf = sqlContext.createDataFrame(df)
sdf.printSchema()

#%%
sdf.head(5)

#%%
# let's see summary stats on the input frame
sdf.describe().toPandas().transpose()

#%%
# data prep for linear regression w/ Spark... have to do the bizarre, annoying thing of restructuring data into 2 cols just for Spark.
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from typing import Union

# UDF for converting column type from vector to double type; src: https://stackoverflow.com/a/56953290
unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())

def prep_sdf( 
    sdf : pyspark.sql.dataframe.DataFrame , 
    targetColName : str , 
    scale_col_names : Union[ list, None ] = None, 
    verbose : bool = False, 
    keep_orig_cols : bool = False ):

    sdf_asm = sdf

    # >= 1 column to be scaled.
    if scale_col_names is not None:
        for c in scale_col_names:
            vecAsm = VectorAssembler( inputCols = [c] , outputCol = c + "_raw" )
            scaler = MinMaxScaler( inputCol = c+"_raw", outputCol = c+"_scaled" )
            pipeline = Pipeline( stages=[ vecAsm, scaler ] )
            sdf_asm = pipeline.fit( sdf_asm ).transform( sdf_asm ).withColumn( c+"_scaled", unlist(c+"_scaled") ).drop( c+"_raw" )
            if not keep_orig_cols:
                sdf_asm = sdf_asm.drop(c)
            if verbose: print("prep_sdf: c={}, cols={}".format( c , str( sdf_asm.columns ) ))
    
    # assemble to Spark ML input format
    numeric_cols = [ t[0] for t in sdf_asm.dtypes if t[1] == 'bigint' or t[1] == 'double' ]
    vecAsm = VectorAssembler( inputCols = [ c for c in numeric_cols if targetCol not in c and targetCol + "_scaled" not in c ] , outputCol = 'features' )
    sdf_asm = vecAsm.transform( sdf_asm )
    sdf_asm = sdf_asm.select([ 'features', targetColName ])  # make a Spark DF of 2 cols: features (K:V pair), target

    return sdf_asm

#%%
# First, we need to specify a target column. What to predict? Let's try suicide rate.
targetCol = "suicides/100k pop"

# Second, build the Spark-is-so-special DF, of unnormalized features (same as above)...
sdf_prepped = prep_sdf( sdf, targetCol, None )
sdf_prepped.show(3)

#%%
def get_sdf_shape( sdf : pyspark.sql.dataframe.DataFrame ):
    return [ sdf.count() , len( sdf.columns ) ]

# Finally, select a training set & hold-out test set.
sdf_train, sdf_test = sdf_prepped.randomSplit([ train_split_pct, 1-train_split_pct  ], seed = random_seed)
print([ get_sdf_shape( sdf_train ), get_sdf_shape( sdf_test ) ])

#%%
from pyspark.ml.regression import LinearRegression

# train a LR on unnormalized features.
slrModel = LinearRegression( maxIter = max_iter , labelCol = targetCol ).fit( sdf_train )

#%%
def print_lr_results( model : pyspark.ml.regression.LinearRegression ):
    # Print the coefficients and intercept for linear regression
    print("Coefs: %s" % str(model.coefficients))
    print("Intercept: %s" % str(model.intercept))

    # Summarize the model over the training set and print out some metrics
    print("numIterations: %d" % model.summary.totalIterations)
    print("objectiveHistory: %s" % str(model.summary.objectiveHistory))
    model.summary.residuals.show()
    print("RMSE: %f" % model.summary.rootMeanSquaredError)
    print("r2: %f" % model.summary.r2)

print_lr_results( slrModel )
# %% [markdown]
### What we learned from this first linear regression:
#
# Weak R^2. Not too surprising.
#
# Let's try feature normalization.

#%% [markdown]
### Spark linear regression, model 2: update model 1 to use normalized features.

#%%
sdf_numeric_feature_names = [ t[0] for t in sdf.dtypes if t[1] == 'bigint' or t[1] == 'double' ]
sdf_prepped = prep_sdf( 
    sdf, 
    targetCol + "_scaled", 
    sdf_numeric_feature_names,
    verbose=False, keep_orig_cols=False )
sdf_prepped.limit(3).toPandas()

# %%
# re-split, re-train, print new results
sdf_train, sdf_test = sdf_prepped.randomSplit([ train_split_pct, 1-train_split_pct ], seed = random_seed)
print([ get_sdf_shape( sdf_train ), get_sdf_shape( sdf_test ) ])
slrModel = LinearRegression( maxIter = max_iter , labelCol = targetCol+"_scaled" ).fit( sdf_train )
print_lr_results( slrModel )


# %% [markdown]
### What we learned about modeling normalized features:
#
# Normalizing improved R^2 in the 10-thousandths place. Practically 0 improvement.
# But, the exercise enabled me to add optional normalization to my Spark pipeline construction.

#%% [markdown]
### Spark linear regression, round 3: parsimony - update model 2 to model using fewer features.
#
# Maybe some of the features are adding noise to our model;
# this is simple multiple, straight-line regression through feature space,
# so capturing the variation owing to feature nonlinearities 
# is difficult absent polynomial or interaction terms.
#
# And for now, I want to keep the model simple and linear.
#
# There are multiple ways to do feature selection and re-modeling though:
# * **Manual feature selection, manual regression**: choose features I believe (either from domain knowledge or naive speculation) are most significant to the model. Then use the selected features in another run of LinearRegression.
# * **Automated feature selection, manual regression**: AIC, BIC, etc.. Then use the selected features in another run of LinearRegression.
# * **Automated feature selection, automated regression**: ElasticNet regression, searching over the `elasticNetParam` (canonically, α) which I did earlier using sklearn, which yielded a best model: `HDI for year=('suicides/100k pop', 'suicides_no', 'gdp_per_capita ($)') @ 0.6068950167453945`
## 
# I don't know anything about suicide, don't really have access to domain experts (psychologists or psychiatrists; but even if I did, domain experts likely don't know this dataset), and above all this is a small exercise in regression anyway.
#
# But I'm curious and want to understand the domain a little better - at minimum, I want to know the rank-order of variable influence. So I'll start with the automated + manual approach.
#
# Problem: upon several minutes of googling, it appears there is no way to do such classic inferential stats as feature selection using information criterions.
#
# Further, statisticians/data scientists with more expertise than me suggest that such stepwise selection may not be a great approach anyway: https://towardsdatascience.com/stopping-stepwise-why-stepwise-selection-is-bad-and-what-you-should-use-instead-90818b3f52df
#
# However, It is possible to extract feature importances from some Spark tree-based regressor models (which under-hood use some information criterion, e.g. entropy, to decide feature importance at tree-building time), e.g.: https://towardsdatascience.com/building-a-linear-regression-with-pyspark-and-mllib-d065c3ba246a
#
# I'll try that.

#%%
# Just for curiosity: build a Random Forest Regressor and extract feature importances: https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.RandomForestRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

rfrModel = RandomForestRegressor(featuresCol ='features', labelCol = targetCol+"_scaled").fit( sdf_train )
rfrPreds = rfrModel.transform( sdf_test )
rfrEvaluator = RegressionEvaluator(
    labelCol = targetCol+"_scaled", predictionCol="prediction", metricName="rmse")
rfrRMSE = rfrEvaluator.evaluate( rfrPreds )
print("Root Mean Squared Error (RMSE) on test data = %g" % rfrRMSE)

# %%
print( rfrModel.numFeatures )
rfrModel.featureImportances

# %%
# Spark's featureImportances vec would be a lot more readable if we had feature vec names associated to the values...
pd.DataFrame({
    "names": [ c for c in sdf_numeric_feature_names if c != targetCol ],
    "featureImportance": rfrModel.featureImportances
})

# %% [markdown]
### What we learned from exploring feature importances:
#
# Oh. Suicide count predicts suicide rate? Someone call the NIMH to report this incredible finding!
#
# Frankly, there aren't many features here on which to model... Yet, we can do better.

#%% [markdown]
### How to do better?
#
# 1) Recall that all my modeled features so far are purely numeric. This is convenient, but excludes probably-useful features like age and sex and generation.
#
# Of course, OLS no longer applies; now I must use a model that can take categorical variables, like logistic regression.
#
# 2) Try a different modeling technique. I've already done this to extract feature importance - what were those results?

#%%
rfrEvaluator = RegressionEvaluator(
    labelCol = targetCol+"_scaled", predictionCol="prediction", metricName="r2")
print( "RF model R^2: {}".format( rfrEvaluator.evaluate( rfrPreds ) ) )

# %% [markdown]
### Wow, my best R^2 so far, over just numeric features!
#
# **Why?**
#
# Probably because random forest modeling can capture nonlinearities that linear modeling generally fails to find.
#

#%% [markdown]
# # Keras regression, round 1
#
##### Inspired by: https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

# %%
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# %%
k_X = dfn[ list( set(dfn.columns) - set(["suicides/100k pop"]) ) ].values
k_y = dfn[ "suicides/100k pop" ].values
k_X_trn, k_X_tst, k_y_trn, k_y_tst = train_test_split(k_X, k_y, test_size=0.2, random_state=random_seed)
print([ k_X.shape , k_y.shape , k_X_trn.shape, k_X_tst.shape, k_y_trn.shape, k_y_tst.shape ])

#%%
# R^2 implementation: https://stackoverflow.com/a/46969576
from keras import backend as K
from keras import losses
from sklearn.metrics import accuracy_score

def coeff_determination(y_true, y_pred):
    """Returns the R^2 value of regression predictions."""
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    
# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(5, input_dim=5, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='glorot_uniform'))
    # Compile model
    # model.compile(loss='mean_squared_error', optimizer='adam')
    model.compile(loss=coeff_determination, optimizer='adam')
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=[accuracy, coeff_determination])
    # model.compile( loss=losses.mean_absolute_percentage_error , optimizer='adam')
    return model

k_estimator = KerasRegressor( build_fn=baseline_model, epochs=5, batch_size=5, verbose=0 )
kfold = KFold( n_splits=10 )
results = cross_val_score( k_estimator, k_X_trn, k_y_trn, cv=kfold )
# print("Results: %.2f (%.2f) MSE" % ( results.mean(), results.std()) )
print("Results: %.2f (%.2f) R^2" % ( results.mean(), results.std()) )  # TODO: WTF is wrong w/ R^2 result???!
# print("Results: %.2f (%.2f) MAPE" % ( results.mean(), results.std()) )  # TODO: WTF is wrong w/ MAPE result too???!

#%%
k_estimator.fit( k_X_trn , k_y_trn )
k_pred = k_estimator.predict( k_X_tst )
accuracy_score( k_y_tst , k_pred )

# %%
