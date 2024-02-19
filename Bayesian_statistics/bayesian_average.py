#%%[markdown]
## Bayesian Average: [Bayesian average - Wikipedia](https://en.wikipedia.org/wiki/Bayesian_average)
# Created: 20240218

#%%
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

C = [C for C in range(0,10)] 
X = [ random.gauss() for i in range(0,5) ] 
n = len(X) 
m = 10 
pd.DataFrame([ (c * m + sum(X) ) / (c+n) for c in C ]).plot(title="C=[0,1,...,10], X=5 draws from N(0,1), n=5, m=10", xlabel="C", ylabel="xbar") 
print(X) 
plt.show()

#%% [markdown]
### Below, notice how the shape of of the curve changes in response to increasing weight (C; left-to-right among subplots), increasing sample size (X; top-to-bottom), and both (towards bottom-right).
#
# We know the prior mean is 10. So in absence of present data, the average curve bends towards 10.
#
# We also know that the present data mean is ~0, being drawn from a Gaussian(mean=0, std=1). So in absence of a prior, the curve bends towards 0.
#
# The two "fight" as we have more of both prior and present information.

#%%
# Create a subplot for each range of values for X
nplots = 16
c_end  = nplots // 2
xi_end = nplots // 2
fig, axs = plt.subplots(c_end, xi_end, figsize=(48, 48))

# prior weight loop
for j in range(c_end):

    df = pd.DataFrame()

    # sample size loop
    for i in range(xi_end):

        # priors
        C = [C for C in range(0,j)]   # prior weight
        m = 10                                   # prior mean

        # present
        X = [ random.gauss() for i in range(0,i) ]     # present sample data
        n = len(X)                                             # present sample size

        if i != 0 and j != 0:       # prevent div-by-0 on no 0 weight and no sample data
            if j == 0:              # zero-weight case ==> arithmetic mean
                xbar = [ (c * m + sum(X) ) / (c+n) for c in [0]*i ]
            else:
                xbar = [ (c * m + sum(X) ) / (c+n) for c in C ]

            df[f"n={i},C={j}"] = xbar
    
            # Plot the data for the current subplot
            df.plot(
                title=f"prior mean m={m}, prior weight C={C}, sample X is n={n} draws from N(0,1),", 
                xlabel="prior weight (C)", 
                ylabel="xbar",
                ax=axs[i,j])

plt.show()
