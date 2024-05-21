
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import altair as alt
from sklearn.ensemble import RandomForestClassifier

def Grafico_RF_HP(X,y)->alt.Chart:
    test_sizes = [0.15,0.20,0.25, 0.3,0.35, 0.4]  # List of different test sizes
    random_states = [42, 123, 456,89]  # List of different random states
    n_estimators_values = [10, 50, 100, 200, 500]  # Number of trees in the forest
    class_weights = [None, 'balanced']
    results=[]

    for test_size in test_sizes:
        for random_state in random_states:
            for n_estimators in n_estimators_values:
                for class_weight in class_weights:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    rf = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight, random_state=random_state)
                    fit = rf.fit(X_train, y_train)
                    predictions = fit.predict(X_test)
                    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
                    recall = tp / (tp + fn)
                    specificity = tn / (tn + fp)
                    accuracy = accuracy_score(y_test, predictions)
                    precision = precision_score(y_test, predictions)
                    results.append({'Test Size': test_size, 
                                    'Random State': random_state,
                                    'N_Estimators': n_estimators,
                                    'Class Weight': class_weight,
                                    'Accuracy': accuracy,
                                    'Precision': precision,
                                    'Recall': recall,
                                    'Specificity':specificity,})
    results_df = pd.DataFrame(results)
    results_df.to_csv("Grafico_RF_HP_OT_MV.csv", index=False)
    grafico_RG = alt.Chart(results_df).mark_circle().encode(
    x=alt.X('Accuracy:Q', scale=alt.Scale(domain=[0.0,0.7])),
    y=alt.Y('Recall:Q', scale=alt.Scale(domain=[0.0,1.3])),
    color=alt.Color('N_Estimators:Q'),
    size='Test Size:Q',
    tooltip=['Accuracy','Precision','Test Size','Random State','N_Estimators','Class Weight','Recall','Specificity']
    ).interactive().properties(height=800)
    return grafico_RG

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import altair as alt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

def Grafico_RF_CV(X, y):
    test_sizes = [0.15,0.20,0.25, 0.3,0.35, 0.4]  # List of different test sizes
    random_states = [42, 123, 456,89]  # List of different random states
    n_estimators_values = [10, 50, 100, 200, 500]  # Number of trees in the forest
    class_weights = [None, 'balanced']
    results=[]
    for test_size in test_sizes:
        for random_state in random_states:
            for n_estimators in n_estimators_values:
                for class_weight in class_weights:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    rf = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight, random_state=random_state)
                    cv_results = cross_validate(rf, X_train, y_train, cv=5, scoring=('accuracy', 'recall', 'precision'))
                    results.append({'Test Size': test_size, 
                                    'Random State': random_state,
                                    'N_Estimators': n_estimators,
                                    'Class Weight': class_weight,
                                    'Accuracy': cv_results['test_accuracy'].mean(),
                                    'Precision': cv_results['test_precision'].mean(),
                                    'Recall': cv_results['test_recall'].mean()})
    results_df = pd.DataFrame(results)
    results_df.to_csv("Grafico_RF_CV_OT_MV.csv", index=False)
    grafico_RG = alt.Chart(results_df).mark_circle().encode(
    x=alt.X('Accuracy:Q', scale=alt.Scale(domain=[0.0,0.7])),
    y=alt.Y('Recall:Q', scale=alt.Scale(domain=[0.0,1.3])),
    color=alt.Color('N_Estimators:Q'),
    tooltip=['Accuracy','Precision','Test Size','Random State','N_Estimators','Class Weight','Recall']
    ).interactive().properties(height=800)
    return grafico_RG