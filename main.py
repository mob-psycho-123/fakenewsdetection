import pandas as pd  
import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, classification_report  
import re  
import string  
import joblib  # Library to save and load models

# Importing machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the datasets
print("Loading datasets...")
dataframe_fake = pd.read_csv("data/Fake.csv")  
dataframe_true = pd.read_csv("data/True.csv")  
print("Datasets loaded.")

# Add class labels
print("Adding class labels...")
dataframe_true["class"] = 1 
dataframe_fake["class"] = 0 

# Remove the last 10 records for manual testing
print("Preparing manual testing datasets...")
dataframe_fake_manual_testing = dataframe_fake.tail(10)  
for i in range(23480,23470,-1):  
    dataframe_fake.drop([i], axis = 0, inplace = True)  
     
dataframe_true_manual_testing = dataframe_true.tail(10)  
for i in range(21416,21406,-1):  
    dataframe_true.drop([i], axis = 0, inplace = True)  

# Mark manual testing samples with class labels
dataframe_fake_manual_testing["class"] = 0  
dataframe_true_manual_testing["class"] = 1  

# Merge the datasets and drop unnecessary columns
print("Merging datasets...")
dataframe_merge = pd.concat([dataframe_fake, dataframe_true], axis=0)  
dataframe = dataframe_merge.drop(["title", "subject", "date"], axis=1)  

# Check for null values
print("Checking for null values...")
print(dataframe.isnull().sum())  

# Shuffle the dataset
print("Shuffling dataset...")
dataframe = dataframe.sample(frac=1)  
dataframe.reset_index(inplace=True)  
dataframe.drop(["index"], axis=1, inplace=True)  

# Text preprocessing function
def wordopt(t):  
    t = t.lower()  
    t = re.sub('\[.*?\]', '', t)  
    t = re.sub("\\W", " ", t)  
    t = re.sub('https?://\S+|www\.\S+', '', t)  
    t = re.sub('<.*?>+', '', t)  
    t = re.sub('[%s]' % re.escape(string.punctuation), '', t)  
    t = re.sub('\n', '', t)  
    t = re.sub('\w*\d\w*', '', t)      
    return t  
  
# Apply text preprocessing
print("Applying text preprocessing...")
dataframe["text"] = dataframe["text"].apply(wordopt)  

# Define dependent and independent variables
x = dataframe["text"]  
y = dataframe["class"]  

# Split the dataset into training and testing sets
print("Splitting dataset into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)  

# Vectorize the text data
# print("Vectorizing text data...")
# vectorization = TfidfVectorizer()  
# xv_train = vectorization.fit_transform(x_train)  
# xv_test = vectorization.transform(x_test)  

# Save the vectorizer
# joblib.dump(vectorization, 'vectorizer.pkl')
# print("Vectorizer saved.")

# Train and save models (only do this once)
# print("Training Logistic Regression model...")
# LR = LogisticRegression()  
# LR.fit(xv_train, y_train)  
# joblib.dump(LR, 'logistic_regression_model.pkl')
# print("Logistic Regression model trained and saved.")

# print("Training Decision Tree model...")
# DT = DecisionTreeClassifier()  
# DT.fit(xv_train, y_train)  
# joblib.dump(DT, 'decision_tree_model.pkl')
# print("Decision Tree model trained and saved.")

# print("Training Gradient Boosting model...")
# GBC = GradientBoostingClassifier(random_state=0)  
# GBC.fit(xv_train, y_train)  
# joblib.dump(GBC, 'gradient_boosting_model.pkl')
# print("Gradient Boosting model trained and saved.")

# print("Training Random Forest model...")
# RFC = RandomForestClassifier(random_state=0)  
# RFC.fit(xv_train, y_train)  
# joblib.dump(RFC, 'random_forest_model.pkl')
# print("Random Forest model trained and saved.")

# Load vectorizer and models (skip training and directly load if models are already saved)
print("Loading trained models and vectorizer...")
vectorization = joblib.load('vectorizer.pkl')
LR = joblib.load('logistic_regression_model.pkl')
# DT = joblib.load('decision_tree_model.pkl')
# GBC = joblib.load('gradient_boosting_model.pkl')
RFC = joblib.load('random_forest_model.pkl')
print("Models and vectorizer loaded successfully.")

# Output label function
def output_label(n):  
    return "Fake News" if n == 0 else "Not A Fake News"  
     
# Manual testing function
def manual_testing(news):  
    testing_news = {"text": [news]}  
    new_def_test = pd.DataFrame(testing_news)  
    new_def_test["text"] = new_def_test["text"].apply(wordopt)  
    new_x_test = new_def_test["text"]  
    new_xv_test = vectorization.transform(new_x_test)  
    
    pred_LR = LR.predict(new_xv_test)  
    # pred_DT = DT.predict(new_xv_test)  
    # pred_GBC = GBC.predict(new_xv_test) 
    pred = RFC.predict(new_xv_test) 
    if (pred or pred_LR):
        print("Real news")
    else:
        print("Fake news")
  
# Input news text for manual testing
ch=1
while(ch):
    news = str(input("Enter news text: "))  
    manual_testing(news)
    ch=int(input("enter 0 to exit:"))


