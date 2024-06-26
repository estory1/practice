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
# df = pd.read_csv('kaggle.com-email-spam-classification-dataset-csv.zip')
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

#%%
# split the corpus
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_clean["text"], df_clean['label'], test_size=0.15, random_state=101)

#%%
# TF-IDF vectorize the training & test sets, separately so as not to leak info between the sets via the TF-IDF coefs.
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
X_train_features = tfidf.fit_transform(X_train)
X_test_features  = tfidf.transform(X_test)
print(X_train_features)

#%%
# summarize the TF-IDF training set features
pd.DataFrame(X_train_features.toarray()).describe()

# %%
# model: start simple w/ LR
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1e9, solver='lbfgs', multi_class='multinomial')
clf.fit(X_train_features, y_train)

print(classification_report(y_test, clf.predict(X_test_features)))

#%%
# model: try Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train_features, y_train)
print(classification_report(y_test, clf.predict(X_test_features)))

# %%
# model: try Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train_features.toarray(), y_train)
print(classification_report(y_test, clf.predict(X_test_features.toarray())))

# %%
