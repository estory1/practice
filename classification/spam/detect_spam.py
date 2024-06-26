#%%[markdown]
## Spam classification
#

#%%
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#%%
%%time
colnames =["label", "text"]
df = pd.read_csv( os.path.join( os.path.expanduser("~") , "Data", "archive.ics.uci.edu", "sms-spam-collection", "archive.ics.uci.edu-sms+spam+collection/SMSSpamCollection" ), sep="\t", names=colnames)
print(df.shape)
df.head()

#%%[markdown]
## Data cleaning / prep

#%%
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Continue + Ollama + starcoder1.5 wrote text_process except for the join during the return statement.
def text_process(text):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in text if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return " ".join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])

#%%
%%time
# remove junk
df_clean = pd.concat([ 
                      df["label"].apply(lambda s: True if s == "spam" else False) , 
                      df['text'].apply(text_process)
                      ], 
                     axis=1, 
                     ignore_index=True)
df_clean.columns = colnames
print(df_clean.shape)
df_clean.head()


#%%[markdown]
## Modeling
### First, try TF-IDF.

random_seed = 101

#%%
%%time
# split the corpus
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_clean["text"], df_clean['label'], test_size=0.15, random_state=random_seed)

#%%
%%time
# TF-IDF vectorize the training & test sets, separately so as not to leak info between the sets via the TF-IDF coefs.
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
X_train_features = tfidf.fit_transform(X_train)
X_test_features  = tfidf.transform(X_test)
print(X_train_features)

#%%
%%time
# summarize the TF-IDF training set features
pd.DataFrame(X_train_features.toarray()).describe()

# %%
%%time
# model: start simple w/ LR
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1e9, solver='lbfgs', multi_class='multinomial')
clf.fit(X_train_features, y_train)

print(classification_report(y_test, clf.predict(X_test_features)))

#%%
%%time
# model: try Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train_features, y_train)
print(classification_report(y_test, clf.predict(X_test_features)))

# %%
%%time
# model: try Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train_features.toarray(), y_train)
print(classification_report(y_test, clf.predict(X_test_features.toarray())))


#%%[markdown]
### Try Cross Validation to see if the result is consistent across different train/test splits.

#%%
%%time
from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(cv=10, 
                           random_state=random_seed, 
                           class_weight="balanced", 
                           scoring="f1")  # balanced class weights because this dataset is imbalanced; balanced re-weights classes inversely-proportional to class probability.
clf.fit(X_train_features, y_train)
print(classification_report(y_test, clf.predict(X_test_features)))

#%%
%%time
# model: try random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=random_seed, 
                           class_weight="balanced")  # balanced class weights because this dataset is imbalanced; balanced re-weights classes inversely-proportional to class probability.
clf.fit(X_train_features, y_train)
print(classification_report(y_test, clf.predict(X_test_features)))

#%%
%%time
# model: try XGBoost, of course
import xgboost as xgb

# search over a small, arbitrary hyperparameter space
for ne, md, lr in [ (2,2,1), (15,15,1), (20,20,1), (35,35,1), 
                   (20,20,0.25), (20,20,0.5), (20,20,2), (20,20,10),
                   (25,25,0.5), (25,25,2), (100,100,0.25) ]:
    
    clf = xgb.XGBClassifier(seed=random_seed, 
                            n_estimators=ne, max_depth=md, learning_rate=lr,
                            objective='binary:logistic')
    clf.fit(X_train_features, y_train)
    print(f"** Given hyperparams: n_estimators={ne}, max_depth={md}, learning_rate={lr}")
    print(classification_report(y_test, clf.predict(X_test_features)))



# %%[markdown]
### **Result analysis**:
# 1. On the particular hold-out test dataset, Logistic Regression outperforms Naive Bayes.
# 2. CV Logistic Regression outperforms hold-out LR.


#%%


#%%[markdown]
## Modeling
### Next, try Bag-of-Words.

#%%

