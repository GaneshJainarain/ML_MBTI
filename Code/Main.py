 #MBTI Personality Prediction Machine Learning Model¶
# Data Analysis
import pandas as pd
import numpy as np
'''
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import pickle as pkl
from scipy import sparse

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud, STOPWORDS

# Text Processing
import re
import itertools
import string
import collections
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Machine Learning packages
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn.cluster as cluster
from sklearn.manifold import TSNE

# Model training and evaluation
from sklearn.model_selection import train_test_split

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance

#Metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import classification_report
 '''
# Ignore noise warning
import warnings
warnings.filterwarnings("ignore")

#loading dataset
data_set = pd.read_csv("/Users/richeyjay/Desktop/MBTI_ML/venv/Code/mbti_1.csv")
#print(data_set.tail())

#Checking to see if there are any missing or null values in the dataset 
#print(data_set.isnull().any())

#Getting an idea of the size of our dataset
nRow, nCol = data_set.shape
#print(f'There are {nRow} rows and {nCol} columns')

#print(data_set.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8675 entries, 0 to 8674
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   type    8675 non-null   object
 1   posts   8675 non-null   object
dtypes: object(2)
memory usage: 135.7+ KB
None
'''
#There are only 2 columns in the dataset
#Total no. of rows are 8675
#There are no null values present in the dataset
#One Disadvantage is that all values are textual, 
#hence they have to be converted to numerical form to train the ML model

#print(data_set.describe(include=['object']))
'''
        type                                              posts
count   8675                                               8675
unique    16                                               8675
top     INFP  'Sometimes I ask myself things like this when ...
freq    1832                                                  1
'''
# There are 16 unique personality type indicators in the dataset
#INFP is the most frequently occuring personality type in our dataset
#(no. of occurences is 1832)
#Lastly, there are no repeating posts in the dataset

#Finding the unique values from the data set
types = np.unique(np.array(data_set['type']))
print(types)
'''
'ENFJ' 'ENFP' 'ENTJ' 'ENTP' 'ESFJ' 'ESFP' 'ESTJ' 'ESTP' 'INFJ' 'INFP'
 'INTJ' 'INTP' 'ISFJ' 'ISFP' 'ISTJ' 'ISTP']
'''
#16 different types corresponding to the differnt 16 mbti types

total = data_set.groupby(['type']).count()*50
#Group by allows you to split your data into separate groups 
#to perform computations for better analysis.
print(total)
'''
      posts
type       
ENFJ   9500
ENFP  33750
ENTJ  11550
ENTP  34250
ESFJ   2100
ESFP   2400
ESTJ   1950
ESTP   4450
INFJ  73500
INFP  91600
INTJ  54550
INTP  65200
ISFJ   8300
ISFP  13550
ISTJ  10250
ISTP  16850
'''