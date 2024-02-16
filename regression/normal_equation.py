#%% [markdown]
# Normal equation for computing the coefficients (aka Theta, aka model parameters) of a linear regression.
### Created: 20210303

#%%
import numpy as np

#%%
# Make an input data matrix. Note since I use pinv (SVD) instead of an ordinary inverse,
# the matrix need not be square and invertible.
X = [ 
    [ 3 , 4, 7  ] , 
    [ 5, 11, 17 ] ,
    [ 9, 10, 13 ] ]

y = [ 1 , 2 , 3 ]

#%%
X_np = np.array( X )

# The normal equation.
theta_np = np.matmul( np.matmul( np.linalg.pinv( np.matmul( X_np.T , X_np ) ) , X_np.T ) , y )
theta_np

# %%
# 
