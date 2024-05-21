

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import altair as alt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import streamlit as st

def Grafico_vizinhos(y_test,y_train,X_test,X_train,vizinhos=100)->alt.Chart:
    k=list(range(1,vizinhos))
    accuracy=[]
    for i in k:
        knn = KNeighborsClassifier(n_neighbors=i)
        fit = knn.fit(X_train, y_train)
        accuracy.append(accuracy_score(y_test, fit.predict(X_test)))
    df_acc = pd.DataFrame({'K Values': k, 'Accuracy Score': accuracy})  
    alt_c = alt.Chart(df_acc).mark_circle().encode(
    alt.X('K Values:Q', scale=alt.Scale(zero=False)),
    alt.Y('Accuracy Score:Q', scale=alt.Scale(domain=[0.2, 0.8]))).interactive()
    return alt_c

def Grafico_vizinhos_CV(X,y,vizinhos=100)->alt.Chart:
    k_values = [i for i in range (1,vizinhos)]
    scores = []

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        scores.append(np.mean(score))
    df_scores = pd.DataFrame({'K Values': k_values, 'Accuracy Score': scores})

    alt_c = alt.Chart(df_scores).mark_circle().encode(
    alt.X('K Values:Q', scale=alt.Scale(zero=False)),
    alt.Y('Accuracy Score:Q', scale=alt.Scale(domain=[0.5, 0.8]))
).interactive().properties(height=800)
    return alt_c

def Grafico_KNN_HP(X,y)->alt.Chart:
        test_sizes = [0.15,0.20,0.25, 0.3,0.35, 0.4]  # List of different test sizes
        random_states = [42, 123, 456,14,23,89]  # List of different random states
        n_neighbors= [1,3,5,7,9,11,13,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51]        
        weights= ['uniform', 'distance']
        metric= ['euclidean', 'manhattan']
        results=[]
        contador=0
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
                            contador+=1
                            st.write('Contador:',contador) 
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