

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import altair as alt
import streamlit as st


def Grafico_vizinhos(X,y,vizinhos=100)->alt.Chart:
    k_values = [i for i in range (1,vizinhos)]
    accuracy=[]
    for i in k_values:
        knn = KNeighborsClassifier(n_neighbors=i, weights="distance", metric="euclidean") #Parametros Otimizados
        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=89) #Parametros Otimizados
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        fit = knn.fit(X_train, y_train)
        accuracy.append(accuracy_score(y_test, fit.predict(X_test)))
    df_acc = pd.DataFrame({'K Values': k_values, 'Accuracy Score': accuracy})  
    alt_c = alt.Chart(df_acc).mark_circle().encode(
    alt.X('K Values:Q', scale=alt.Scale(zero=False)),
    alt.Y('Accuracy Score:Q', scale=alt.Scale(domain=[0.5, 1]))).interactive()
    return alt_c


def Grafico_KNN_HP(X,y)->alt.Chart:
        test_sizes = [0.15,0.20,0.25, 0.3,0.35, 0.4]  # List of different test sizes
        random_states = [42, 123, 456,14,23,89]  # List of different random states
        n_neighbors= [1,3,5,7,9,11,13,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51]        
        weights= ['uniform', 'distance']
        metric= ['euclidean', 'manhattan']
        results=[]
        for test_size in test_sizes:
            for random_state in random_states:
                for neighbors in n_neighbors:
                    for weight in weights:
                        for m in metric:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                            scaler = StandardScaler()
                            X_train = scaler.fit_transform(X_train)
                            X_test = scaler.transform(X_test)
                            knn = KNeighborsClassifier(n_neighbors=neighbors,metric=m,weights=weight)
                            fit = knn.fit(X_train, y_train)
                            predictions = fit.predict(X_test)
                            tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
                            recall = tp / (tp + fn)
                            specificity = tn / (tn + fp)
                            accuracy = accuracy_score(y_test, predictions)
                            precision = precision_score(y_test, predictions)
                            results.append({'Test Size': test_size, 
                                            'Random State': random_state,
                                            'Neighbors': neighbors,
                                            'Weight': weight,
                                            'Metric': m,
                                            'Accuracy': accuracy,
                                            'Precision': precision,
                                            'Recall': recall,
                                            'Specificity':specificity,})
        results_df = pd.DataFrame(results)
        results_df.to_csv("Grafico_KNN_HP_MV.csv", index=False)
        grafico_RG = alt.Chart(results_df).mark_circle().encode(
        x=alt.X('Accuracy:Q', scale=alt.Scale(domain=[0.0,0.7])),
        y=alt.Y('Recall:Q', scale=alt.Scale(domain=[0.0,1.3])),
        color=alt.Color('Neighbors:Q'),
        size='Test Size:Q',
        tooltip=['Accuracy','Precision','Test Size','Random State','Neighbors','Weight','Metric','Recall','Specificity']
        ).interactive().properties(height=800)
        return grafico_RG


def Grafico_KNN_CV(X, y):
    test_sizes = [0.15,0.20,0.25, 0.3,0.35, 0.4]  # List of different test sizes
    random_states = [42, 123, 456,89, 36, 95, 70, 34, 74]  # List of different random states
    n_neighbors_values = [3, 5, 7, 9, 11]  # Number of neighbors for KNN
    results=[]
    for test_size in test_sizes:
        for random_state in random_states:
            for n_neighbors in n_neighbors_values:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                knn = KNeighborsClassifier(n_neighbors=n_neighbors)
                cv_results = cross_validate(knn, X_train, y_train, cv=5, scoring=('accuracy', 'recall', 'precision'))
                results.append({'Test Size': test_size, 
                                'Random State': random_state,
                                'N_Neighbors': n_neighbors,
                                'Accuracy': cv_results['test_accuracy'].mean(),
                                'Precision': cv_results['test_precision'].mean(),
                                'Recall': cv_results['test_recall'].mean()})
    results_df = pd.DataFrame(results)
    results_df.to_csv("Grafico_KNN_MV.csv", index=False)
    grafico_KNN = alt.Chart(results_df).mark_circle().encode(
    x=alt.X('Accuracy:Q', scale=alt.Scale(domain=[0.0,0.7])),
    y=alt.Y('Recall:Q', scale=alt.Scale(domain=[0.0,1.3])),
    color=alt.Color('N_Neighbors:Q'),
    tooltip=['Accuracy','Precision','Test Size','Random State','N_Neighbors','Recall']
    ).interactive().properties(height=800)
    return grafico_KNN