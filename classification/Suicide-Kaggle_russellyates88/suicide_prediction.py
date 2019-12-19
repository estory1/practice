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

#%%
randseed = 42

filepath = "/Users/estory/Documents/syncable/home/dev/data_science/practice/suicide/suicide-rates-overview-1985-to-2016.zip"
df = pd.read_csv(filepath)
df = df.dropna()
df.head()

#%%
df_cols_numeric = df.describe().columns
print("df_cols_numeric: " + str(df_cols_numeric))
df.describe()


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

    kf = KFold( n_splits=10, shuffle=True, random_state=randseed )
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
        # mi_arr = mutual_info_regression( X_test, Y_test, random_state = randseed )
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

#%%
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
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=randseed )

        lr = ElasticNetCV( cv=10, random_state=randseed )
        lr.fit( X_train, Y_train )

        r2 = lr.score( X_test, Y_test )
        print( "R^2 for "+str(y)+"="+str(x)+": " + str( r2 ) )
        if r2 > r2_at_max:
            y_at_max = y
            x_at_max = x
            r2_at_max = r2

#%%
print( "Best of all ElasticNet models: {}={} @ {}".format( y_at_max, x_at_max, r2_at_max ) )


#%%
