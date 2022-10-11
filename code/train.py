import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import xgboost as xgb
import joblib
import gzip
from code.helper import preprocess_kaggle

# Machine learning model: XGBoost 

clean_df = preprocess_kaggle('data/dataset.csv')

# Preprocessing
X_data = clean_df.iloc[:,:-1]
y_data = clean_df.iloc[:,-1]

# Convert y to categorical values
y_data = y_data.astype('category')

# Convert y categories tu numbers with encoder
le = preprocessing.LabelEncoder()
le.fit(y_data)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

# Convert labels to numbers
y_train = le.transform(y_train)
y_test = le.transform(y_test)

# Init classifier
model = xgb.XGBClassifier()

# Fit
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Test accuracy
print(f"The accuracy of the model is {accuracy_score(y_test, preds)}")

# Export model
joblib.dump(model, gzip.open('model/model_binary.dat.gz', "wb"))
model.save_model("model/xgboost_model.json")
