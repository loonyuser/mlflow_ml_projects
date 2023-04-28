import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    
    holiday_package_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'holiday_package_data.csv')
    package_sale_data = pd.read_csv( holiday_package_data_path )
    
    package_sale_data.replace('Fe Male','Female', inplace = True)
    package_sale_data = package_sale_data.drop(columns = ['CustomerID'])
    
    numeric_features = ['Age', 'DurationOfPitch', 'MonthlyIncome']
    numeric_transformer = Pipeline(
    steps=[('imputer', SimpleImputer(strategy = 'median')), ("scaler", StandardScaler())])

    categorical_features = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched','MaritalStatus','Designation']
    categorical_transformer = Pipeline(
    steps=[('imputer', SimpleImputer(strategy = 'most_frequent')),('encoder', OneHotEncoder(handle_unknown = 'ignore'))
    ])
    preprocessor = ColumnTransformer(
    transformers = [
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder = SimpleImputer(strategy = 'most_frequent'))
    
    X = package_sale_data.drop(columns = ['ProdTaken'])

    y = package_sale_data['ProdTaken']
      
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    min_samples_split = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    min_samples_leaf  = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    with mlflow.start_run():          
        model = RandomForestClassifier(n_estimators = n_estimators, min_samples_split = min_samples_split, 
                                       min_samples_leaf = min_samples_leaf)
        clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size= 0.2, random_state = 123)

        clf.fit(X_train, y_train)
        y_pred =  clf.predict(X_test)
        predictions_proba = clf.predict_proba(X_test)
        
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision_score = precision_score(y_test, y_pred)
        test_recall_score = recall_score(y_test, y_pred)
        test_f1_score = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, predictions_proba[:,1])
        metrics = {'Test_accuracy': test_accuracy, 'Test_precision_score': test_precision_score,
                   'Test_recall_score': test_recall_score,'Test_f1_score': test_f1_score, 'AUC_score': auc_score}
    
          
        mlflow.log_metrics(metrics)
    
        mlflow.set_tag('Classifier', 'RF-model')
       
        mlflow.sklearn.log_model(clf, 'RF-model')
        
        

        
        
