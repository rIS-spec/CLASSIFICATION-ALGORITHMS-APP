"""
Streamlit ML Explorer — Robust file handling update

This file includes robust handling for uploaded files including:
- Proper handling of empty uploads (shows friendly Streamlit error instead of crashing)
- Safe detection and reading of CSV and Excel files
- Uses uploaded.seek(0) before multiple reads so pandas doesn't see an empty buffer
- Catches pandas EmptyDataError, UnicodeDecodeError, and suggests user actions
- Provides clear sidebar guidance when uploaded file has no columns or is empty

Author: Updated for Arish Mahammad — fixes EmptyDataError and improves user guidance
"""

import streamlit as st
import pandas as pd
import numpy as np
import traceback
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import os
from pandas.errors import EmptyDataError

st.set_page_config(page_title="ML Explorer — Robust Uploads", layout="wide")

# -------------------------- Helper functions --------------------------
@st.cache_data
def load_builtin_dataset(name):
    if name == "Iris":
        data = datasets.load_iris(as_frame=True)
    elif name == "Wine":
        data = datasets.load_wine(as_frame=True)
    elif name == "Breast Cancer":
        data = datasets.load_breast_cancer(as_frame=True)
    elif name == "Digits":
        data = datasets.load_digits(as_frame=True)
    else:
        return None

    X = data.data
    y = data.target
    feature_names = data.feature_names if hasattr(data, 'feature_names') else list(X.columns)
    return X, y, feature_names


def safe_read_table(uploaded_file):
    """Try to read uploaded file as CSV first; if that fails and filename indicates Excel, try read_excel.
    Returns (df, error_message). If df is None and error_message is set, show the reason.
    """
    if uploaded_file is None:
        return None, 'No file uploaded'

    # reset pointer
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    filename = getattr(uploaded_file, 'name', '')
    # try CSV
    try:
        df = pd.read_csv(uploaded_file)
        # reset pointer for any later reads
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        if df.shape[1] == 0:
            return None, 'File has no columns (empty or malformed CSV)'
        return df, None
    except EmptyDataError:
        # empty file
        return None, 'Uploaded file is empty'
    except UnicodeDecodeError:
        return None, 'Could not decode file. Is it a binary (Excel) file?'
    except Exception as e_csv:
        # if file name suggests Excel, try reading as Excel
        try:
            if filename.lower().endswith(('.xls', '.xlsx')):
                try:
                    uploaded_file.seek(0)
                except Exception:
                    pass
                df = pd.read_excel(uploaded_file)
                try:
                    uploaded_file.seek(0)
                except Exception:
                    pass
                if df.shape[1] == 0:
                    return None, 'Excel file read but no columns found.'
                return df, None
        except EmptyDataError:
            return None, 'Uploaded Excel file is empty.'
        except Exception:
            # fall through to return csv error
            return None, f'Failed to parse file as CSV or Excel: {e_csv}'


def safe_encode_series(s):
    if s.dtype == 'object' or s.dtype.name == 'category':
        return LabelEncoder().fit_transform(s.astype(str))
    return s.values


def load_uploaded_csv(uploaded_file, target_name=None):
    df, err = safe_read_table(uploaded_file)
    if err is not None:
        return None, None, None, err

    st.write("Uploaded dataset preview (first 5 rows):")
    st.dataframe(df.head())

    cols = list(df.columns)
    if target_name is None or target_name == '':
        # return dataframe so user can pick target column later
        return df, None, cols, None

    if target_name not in df.columns:
        return None, None, cols, 'Target column not found in uploaded CSV.'

    X = df.drop(columns=[target_name])
    y = df[target_name]

    # encode non-numeric features
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    return X, y, cols, None


def get_classifier(name, params, problem_type='classification'):
    if problem_type == 'classification':
        if name == 'K-Nearest Neighbors':
            return KNeighborsClassifier(n_neighbors=params.get('n_neighbors', 5))
        if name == 'Logistic Regression':
            return LogisticRegression(max_iter=1000, C=params.get('C', 1.0), solver='lbfgs', multi_class='auto')
        if name == 'SVM':
            return SVC(C=params.get('C', 1.0), probability=True, kernel=params.get('kernel', 'rbf'))
        if name == 'Decision Tree':
            return DecisionTreeClassifier(max_depth=params.get('max_depth', None))
        if name == 'Random Forest':
            return RandomForestClassifier(n_estimators=params.get('n_estimators', 100), max_depth=params.get('max_depth', None))
        if name == 'Naive Bayes':
            return GaussianNB()
    else:
        # regressors
        if name == 'Linear Regression':
            return LinearRegression()
        if name == 'Random Forest Regressor':
            return RandomForestRegressor(n_estimators=params.get('n_estimators', 100), max_depth=params.get('max_depth', None), random_state=42)
    raise ValueError('Unknown estimator')


def evaluate_classification(model, X_test, y_test, average='weighted'):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=average, zero_division=0)
    rec = recall_score(y_test, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, cm=cm, y_pred=y_pred)


def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return dict(mse=mse, mae=mae, r2=r2, y_pred=y_pred)


def plot_confusion(cm, labels):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig


def plot_roc(model, X_test, y_test):
    try:
        y_score = model.predict_proba(X_test)
    except Exception:
        st.info('Classifier does not support probability estimates — ROC not available.')
        return None

    n_classes = y_score.shape[1]
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
        auc = roc_auc_score(y_test, y_score[:, 1])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        ax.plot([0, 1], [0, 1], linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        return fig
    else:
        st.info('ROC curve plotting only implemented for binary classification in this app.')
        return None


def plot_pca_scatter(X, y, feature_names, target_names=None):
    try:
        pca = PCA(n_components=2)
        Xp = pca.fit_transform(X)
    except Exception as e:
        st.warning('PCA failed: ' + str(e))
        return None
    dfp = pd.DataFrame(Xp, columns=['PC1', 'PC2'])
    dfp['target'] = pd.Series(y).astype(str)
    fig, ax = plt.subplots()
    sns.scatterplot(data=dfp, x='PC1', y='PC2', hue='target', palette='tab10', ax=ax)
    ax.set_title('PCA (2 components)')
    return fig

# -------------------------- Streamlit App Layout --------------------------
st.title('📊 ML Explorer — Robust Uploads')
st.markdown('Upload datasets, detect target type, choose estimator (classifier/regressor), tune hyperparameters, and visualize results.')

# Sidebar: dataset selection and upload
st.sidebar.header('Dataset')
dataset_source = st.sidebar.radio('Choose source', ('Built-in (sklearn)', 'Upload CSV / Excel'))

X = y = feature_names = target_names = None
uploaded_df_columns = None

if dataset_source == 'Built-in (sklearn)':
    dataset_name = st.sidebar.selectbox('Select dataset', ('Iris', 'Wine', 'Breast Cancer', 'Digits'))
    res = load_builtin_dataset(dataset_name)
    if res is None:
        st.error('Failed to load built-in dataset.')
        st.stop()
    X, y, feature_names = res
    try:
        target_names = datasets.load_iris().target_names if dataset_name == 'Iris' else None
    except Exception:
        target_names = None
else:
    uploaded = st.sidebar.file_uploader('Upload CSV or Excel', type=['csv', 'xls', 'xlsx'])
    target_col = ''
    if uploaded is not None:
        # safe read header/preview
        df_preview, err = safe_read_table(uploaded)
        if err is not None:
            st.error('Upload error: ' + err)
            st.info('Try opening the file to ensure it has a header row and data. If it is an Excel file, ensure the first sheet has data.')
            st.stop()
        # present columns to choose target from
        cols = list(df_preview.columns)
        target_col = st.sidebar.selectbox('Select target column (if CSV uploaded)', options=[''] + cols)
        if target_col == '':
            st.info('Select the target column from the sidebar dropdown to proceed.')
            st.stop()
        X, y, cols, err2 = load_uploaded_csv(uploaded, target_col)
        if err2 is not None:
            st.error('Error processing uploaded file: ' + err2)
            st.stop()
        feature_names = cols

if X is None or y is None:
    st.info('Please select or upload a dataset and choose the target column.')
    st.stop()

# Debug: inspect target
st.sidebar.header('Debug & Target Info')
y_series = pd.Series(y)
st.sidebar.write('Target dtype:', str(y_series.dtype))
st.sidebar.write('Unique values count:', int(y_series.nunique()))
try:
    st.sidebar.write('Sample unique values:', list(pd.Series(y).unique()[:10]))
except Exception:
    pass

# Let user choose problem type or auto detect
problem_choice = st.sidebar.selectbox('Problem type detection', ('Auto-detect', 'Force Classification', 'Force Regression'))

detected_problem = None
if problem_choice == 'Auto-detect':
    if pd.api.types.is_numeric_dtype(y_series) and y_series.nunique() > 20:
        detected_problem = 'regression'
    else:
        detected_problem = 'classification'
else:
    detected_problem = 'classification' if problem_choice == 'Force Classification' else 'regression'

st.sidebar.write('Detected/problem set to:', detected_problem)

# If user forces classification but target continuous -> offer binning
if detected_problem == 'classification' and pd.api.types.is_numeric_dtype(y_series) and y_series.nunique() > 20:
    st.warning('Target appears continuous but classification was chosen. You can bin the target into classes or switch to regression.')
    action = st.sidebar.radio('Resolve continuous target', ('Bin target into classes', 'Switch to regression (recommended)'), index=0)
    if action == 'Bin target into classes':
        n_bins = st.sidebar.slider('Number of bins (classes)', 2, 10, 3)
        try:
            y_binned = pd.qcut(y_series.astype(float), q=n_bins, labels=False, duplicates='drop')
            st.success(f'Binned into {n_bins} classes.')
            y = y_binned.values
            y_series = pd.Series(y)
            detected_problem = 'classification'
        except Exception as e:
            st.error('Binning failed: ' + str(e))
            st.stop()
    else:
        detected_problem = 'regression'

# ---------------- Prepare data ----------------
st.write('**Dataset shape:**', X.shape)
if isinstance(X, pd.DataFrame):
    st.dataframe(X.head())

# Sidebar: train-test split
st.sidebar.header('Train / Test')
test_size = st.sidebar.slider('Test set size (fraction)', 0.1, 0.5, 0.25, 0.05)
random_state = st.sidebar.number_input('Random seed', value=42, step=1)

# Sidebar: estimator selection
st.sidebar.header('Estimator & Hyperparameters')
if detected_problem == 'classification':
    estimator_name = st.sidebar.selectbox('Choose classifier', ('Logistic Regression', 'K-Nearest Neighbors', 'SVM', 'Decision Tree', 'Random Forest', 'Naive Bayes'))
else:
    estimator_name = st.sidebar.selectbox('Choose regressor', ('Linear Regression', 'Random Forest Regressor'))

params = {}
if estimator_name == 'K-Nearest Neighbors':
    params['n_neighbors'] = st.sidebar.slider('n_neighbors', 1, 30, 5)
if estimator_name == 'Logistic Regression' or estimator_name == 'SVM':
    if estimator_name == 'Logistic Regression':
        params['C'] = st.sidebar.slider('C (inverse reg strength)', 0.01, 10.0, 1.0)
    else:
        params['C'] = st.sidebar.slider('C', 0.01, 10.0, 1.0)
        params['kernel'] = st.sidebar.selectbox('kernel', ('rbf', 'linear', 'poly'))
if estimator_name == 'Decision Tree':
    params['max_depth'] = st.sidebar.slider('max_depth (None = 0)', 0, 20, 0)
    if params['max_depth'] == 0:
        params['max_depth'] = None
if estimator_name in ('Random Forest', 'Random Forest Regressor'):
    params['n_estimators'] = st.sidebar.slider('n_estimators', 10, 500, 100, step=10)
    params['max_depth'] = st.sidebar.slider('max_depth (None = 0)', 0, 50, 0)
    if params['max_depth'] == 0:
        params['max_depth'] = None

# Sidebar: scaling & cross-val
st.sidebar.header('Preprocessing & Eval')
use_scaler = st.sidebar.checkbox('Use StandardScaler', value=True)
cross_val = st.sidebar.checkbox('Run cross-validation (5-fold) for score estimate', value=False)

# Prepare features: if DataFrame, encode object columns
if isinstance(X, pd.DataFrame):
    X_proc = X.copy()
else:
    X_proc = pd.DataFrame(X, columns=feature_names)

for col in X_proc.select_dtypes(include=['object', 'category']).columns:
    X_proc[col] = LabelEncoder().fit_transform(X_proc[col].astype(str))

# If target is object -> label encode
if pd.api.types.is_object_dtype(y_series) or pd.api.types.is_categorical_dtype(y_series):
    y = LabelEncoder().fit_transform(y_series.astype(str))
else:
    # keep numeric as-is (may be binned into integers earlier)
    y = y_series.values

# split
try:
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=test_size, random_state=int(random_state))
except Exception as e:
    st.error('train_test_split failed: ' + str(e))
    st.stop()

# build pipeline
estimator = get_classifier(estimator_name, params, problem_type= 'classification' if detected_problem=='classification' else 'regression')
if use_scaler and detected_problem == 'classification':
    model = make_pipeline(StandardScaler(), estimator)
elif use_scaler and detected_problem == 'regression':
    model = make_pipeline(StandardScaler(), estimator)
else:
    model = make_pipeline(estimator)

# train with try/except for debug
with st.spinner('Training model...'):
    try:
        model.fit(X_train, y_train)
        st.success('Model trained successfully')
    except Exception as e:
        st.error('Model training failed: ' + str(e))
        st.text('Traceback:')
        st.text(traceback.format_exc())
        st.stop()

# cross-val
if cross_val:
    try:
        cv_scores = cross_val_score(model, X_proc, y, cv=5, scoring='r2' if detected_problem=='regression' else 'accuracy')
        st.write(f'Cross-validation score (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')
    except Exception as e:
        st.warning('Cross-validation failed: ' + str(e))

# evaluate
if detected_problem == 'classification':
    results = evaluate_classification(model, X_test, y_test)
    st.subheader('Model performance on test set (classification)')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Accuracy', f"{results['accuracy']:.3f}")
    col2.metric('Precision', f"{results['precision']:.3f}")
    col3.metric('Recall', f"{results['recall']:.3f}")
    col4.metric('F1-score', f"{results['f1']:.3f}")

    # Confusion matrix
    st.subheader('Confusion Matrix')
    labels = list(range(len(np.unique(y))))
    fig_cm = plot_confusion(results['cm'], labels)
    st.pyplot(fig_cm)

    # ROC (if applicable)
    st.subheader('ROC Curve (binary classification only)')
    fig_roc = plot_roc(model, X_test, y_test)
    if fig_roc is not None:
        st.pyplot(fig_roc)
else:
    results = evaluate_regression(model, X_test, y_test)
    st.subheader('Model performance on test set (regression)')
    col1, col2, col3 = st.columns(3)
    col1.metric('MSE', f"{results['mse']:.3f}")
    col2.metric('MAE', f"{results['mae']:.3f}")
    col3.metric('R2', f"{results['r2']:.3f}")

# PCA visualization
st.subheader('PCA Projection (2D)')
fig_pca = plot_pca_scatter(X_proc, y, feature_names, target_names)
if fig_pca is not None:
    st.pyplot(fig_pca)

# Download trained model
st.subheader('Download trained model')
buf = io.BytesIO()
joblib.dump(model, buf)
buf.seek(0)
st.download_button(label='Download model (joblib)', data=buf, file_name='trained_model.joblib')

# Show feature importances where available
if estimator_name in ('Random Forest', 'Decision Tree', 'Random Forest Regressor'):
    try:
        underlying = model.named_steps[list(model.named_steps.keys())[-1]]
        importances = getattr(underlying, 'feature_importances_', None)
        if importances is not None:
            fi = pd.DataFrame({'feature': X_proc.columns, 'importance': importances}).sort_values('importance', ascending=False)
            st.subheader('Feature importances')
            st.dataframe(fi)
    except Exception as e:
        st.warning('Could not show feature importances: ' + str(e))

# Small usage tips
st.markdown('---')
st.write('**Tips:** If you get `Unknown label type: continuous`, it means the target is continuous. Use the sidebar to switch to regression or bin the target into classes.')

# Learning reinforcement summary
st.markdown('## You now fully understand:')
st.write('- How to detect whether a dataset is classification or regression')
st.write('- Options to convert continuous targets into classes (binning)')
st.write('- How to switch between classifiers and regressors in one app')

st.success('App ready — tweak hyperparameters and datasets from the sidebar!')
