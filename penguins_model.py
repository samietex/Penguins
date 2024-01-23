import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, precision_recall_fscore_support
import seaborn as sns
import pickle
import os
import joblib


penguins = pd.read_csv('penguins_cleaned.csv')

penguins['species'] = penguins['species'].replace({'Adelie': 0, 'Gentoo': 1, 'Chinstrap': 2})

lb = LabelEncoder()
def encoder(df):
    encoders = {}  # To store label encoders
    for i in df.columns:
        if df[i].dtype == 'O':
            lb = LabelEncoder()
            df[i] = lb.fit_transform(df[i])
            encoders[i] = lb  # Store the encoder
    return df, encoders


penguins, encoders = encoder(penguins)




class models:
    def __init__(self, data, target, encoders):
        self.encoders = encoders
        x = data.drop(target, axis = 1)
        y = data[target]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


        self.models = {
            'XGBoost': xgb.XGBClassifier(use_label_encoder = False, eval_metrics = 'logloss')
        }

        self.metrics = {}
    
    def fit_models(self):
        for name, model in self.models.items():
            model.fit(self.x_train, self.y_train)
            print(f'{name} has been fitted.')

    def evaluate_models(self):

        for name, model in self.models.items():
            predictions = model.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, predictions, average='weighted')
            report = classification_report(self.y_test, predictions)

            self.metrics[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

            print(f'{name} Metrics:')
            print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}\n')

    def visualize_comparision(self):

        metrics_df = pd.DataFrame(self.metrics)
        for metric in metrics_df.columns:
            sns.barplot(x = metrics_df.index, y = metrics_df[metric])
            plt.title(f'Comparison of {metric}')
            plt.ylabel(metric)
            plt.show()

    def save_model(self, directory='saved_models'):
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save models
        for name, model in self.models.items():
            filename = os.path.join(directory, f'{name}.pkl')
            with open(filename, 'wb') as file:
                pickle.dump(model, file)
            print(f'{name} model saved to {filename}')

        # Save Label Encoders
        for col, lb in self.encoders.items():  # Change here
            joblib.dump(lb, os.path.join(directory, f'{col}_encoder.pkl'))


model = models(penguins, 'species', encoders)
model.fit_models()
model.evaluate_models()
model.visualize_comparision()
model.save_model()