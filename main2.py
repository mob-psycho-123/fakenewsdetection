import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
import re
import string

# Function to preprocess text
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Load the datasets
print("Loading datasets...")
dataframe_fake = pd.read_csv("data/Fake.csv")
dataframe_true = pd.read_csv("data/True.csv")
print("Datasets loaded.")

# Add class labels
dataframe_true["class"] = 1
dataframe_fake["class"] = 0

# Combine datasets
dataframe = pd.concat([dataframe_fake, dataframe_true], axis=0)
dataframe = dataframe.drop(["title", "subject", "date"], axis=1)
dataframe["text"] = dataframe["text"].apply(wordopt)

# Split the dataset into training and testing sets
x = dataframe["text"]
y = dataframe["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Vectorize the text data
print("Vectorizing text data...")
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Save the vectorizer
joblib.dump(vectorization, 'vectorizer.pkl')
print("Vectorizer saved.")

# Train and save models
print("Training and saving models...")
LR = LogisticRegression()
DT = DecisionTreeClassifier()
GBC = GradientBoostingClassifier(random_state=0)
RFC = RandomForestClassifier(random_state=0)

# Train the models
LR.fit(xv_train, y_train)
DT.fit(xv_train, y_train)
GBC.fit(xv_train, y_train)
RFC.fit(xv_train, y_train)

# Save the models
joblib.dump(LR, 'logistic_regression_model.pkl')
joblib.dump(DT, 'decision_tree_model.pkl')
joblib.dump(GBC, 'gradient_boosting_model.pkl')
joblib.dump(RFC, 'random_forest_model.pkl')
print("Models trained and saved.")

# Load vectorizer and models for testing
vectorization = joblib.load('vectorizer.pkl')
LR = joblib.load('logistic_regression_model.pkl')
DT = joblib.load('decision_tree_model.pkl')
GBC = joblib.load('gradient_boosting_model.pkl')
RFC = joblib.load('random_forest_model.pkl')

# Manual testing function
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)
    if (pred_LR or pred_DT or pred_GBC or pred_RFC):
        print("real news")
    else:
        print("fake")

    

# Input news text for manual testing
news = str(input("Enter news text: "))
manual_testing(news)
