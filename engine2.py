import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, plot_importance
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, roc_curve, auc, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from ML_pipeline import dataset
from ML_pipeline import data_splitting
from ML_pipeline import data_preprocessing
from ML_pipeline import feature_engineering
from ML_pipeline import upsampling_minorityClass
from ML_pipeline import scaling_features
from ML_pipeline import model_params
from ML_pipeline import train_model
from ML_pipeline import predict_model

st.title("LightGBM Model Training and Evaluation")

st.write('### Script started')

# Importing datasets
train, val = dataset.read_data()

# Splitting the train dataset into training and testing
df_test, df_train, y_test, y_train, train, test = data_splitting.training_testing_dataset(train)

# Data preprocessing
train_clean, test_clean, val_clean = data_preprocessing.data_preprocessing(
    train, test, val,
    'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate', 'MonthlyIncome', 'DebtRatio',
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfDependents', 'SeriousDlqin2yrs'
)

# Feature Engineering 
train_df, test_df, val_df = feature_engineering.feature_engineering(
    train_clean, test_clean, val_clean,
    'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate', 'NumberOfOpenCreditLinesAndLoans',
    'NumberRealEstateLoansOrLines', 'MonthlyIncome', 'NumberOfDependents',
    'DebtRatio', 'age'
)

# Data Transformation
train_x_df, train_y, test_x_df, test_y, val_x_df = upsampling_minorityClass.upsampling_class(
    train_df, test_df, val_df, False
)

# Scaling of the features
train_x_scaled, test_x_scaled, val_x_scaled = scaling_features.scaling_features(
    train_x_df, test_x_df, val_x_df, False
)

# Model Object  
classifier = model_params.model_params()

# Training the model 
model = train_model.train_model(classifier, train_x_scaled, train_y, test_x_scaled, test_y)

# Predicting the model 
val_x_scaled = predict_model.predict_model(model, val_x_scaled)

# Model evaluation and visualization
def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    st.pyplot(plt)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

def plot_feature_importance(model):
    plt.figure(figsize=(10, 6))
    plot_importance(model, max_num_features=10, importance_type='gain', height=0.5, title='Feature Importance')
    st.pyplot(plt)

def plot_predictions(y_true, y_proba):
    plt.figure(figsize=(10, 6))
    sns.histplot(y_proba, kde=True, bins=50, color='blue', label='Predicted Probability')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
    plt.title('Distribution of Predicted Probabilities')
    plt.xlabel('Predicted Probability of Delinquency')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    st.pyplot(plt)

# Evaluate on test set
y_pred = model.predict(test_x_scaled)
y_pred_proba = model.predict_proba(test_x_scaled)[:, 1]

# ROC Curve
st.write("### ROC Curve")
plot_roc_curve(test_y, y_pred_proba)

# Confusion Matrix
st.write("### Confusion Matrix")
plot_confusion_matrix(test_y, y_pred)

# Feature Importance
st.write("### Feature Importance")
plot_feature_importance(model)

# Predictions Distribution
st.write("### Predictions Distribution")
plot_predictions(test_y, y_pred_proba)

# Evaluate model performance
accuracy = accuracy_score(test_y, y_pred)
precision = precision_score(test_y, y_pred)
recall = recall_score(test_y, y_pred)
f1 = f1_score(test_y, y_pred)
roc_auc = roc_auc_score(test_y, y_pred_proba)

st.write(f'**Accuracy:** {accuracy:.4f}')
st.write(f'**Precision:** {precision:.4f}')
st.write(f'**Recall:** {recall:.4f}')
st.write(f'**F1 Score:** {f1:.4f}')
st.write(f'**ROC AUC:** {roc_auc:.4f}')

# Save predictions
path = r"D:\G10x"
val_x_scaled.to_csv(path + '/' + 'test.csv', index=False)

st.write("### Output CSV File")
output_df = pd.read_csv(path + '/' + 'test.csv')
st.write(output_df)

st.write('### Script completed successfully')

