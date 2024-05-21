import altair as alt
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import pandas as pd
import streamlit as st
from sklearn.model_selection import GridSearchCV


# Assume X is your features and y is your target
def SVM_HP(X, y):
    Contador=0
    resultados = [] # Initialize an empty list to store the results
    C_values = [0.1, 1, 10, 100]
    kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
    gamma_values = ['scale', 'auto', 0.1, 1]
    test_sizes = [0.15,0.20,0.25, 0.3,0.35]  # List of different test sizes
    random_states = [42, 456]  # List of different random states
    for c in C_values:
        
        for kernel in kernel_values:
            for gamma in gamma_values:
                for test_size in test_sizes:
                    for random_state in random_states:
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                        
                        # Initialize the SVM model
                        model = svm.SVC(kernel=kernel, C=c, gamma=gamma)  # you can change the kernel as needed
                        Contador+=1
                        # Fit the model to the training data
                        model.fit(X_train, y_train)
                        st.write('Contador:',Contador)
                        # Use the model to make predictions on unseen data
                        predictions = model.predict(X_test)

                        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
                        recall= recall_score(y_test, predictions)
                        specificity = tn / (tn + fp)
                        accuracy = accuracy_score(y_test, predictions)
                        precision = precision_score(y_test, predictions)
                         
                        st.write(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, Specificity: {specificity}')
                        resultados.append({'C': c,
                                            'Kernel': kernel, 
                                            'Gamma': gamma,
                                            'Test Size': test_size, 
                                            'Random State': random_state,
                                            'Accuracy': accuracy,
                                            'Precision': precision,
                                            'Recall': recall,
                                            'Specificity':specificity,})
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("Grafico_SVM_HP_MV.csv", index=False)
    grafico_RG = alt.Chart(df_resultados).mark_circle().encode(
    x=alt.X('Precision:Q', scale=alt.Scale(zero=False)),
    y=alt.Y('Accuracy:Q', scale=alt.Scale(zero=False)),
    color='Test Size:Q',
    size='C:N',
    tooltip=['Accuracy','Precision','Test Size','Random State','C','Kernel','Gamma','Recall','Specificity']
    ).interactive().properties(height=800)
    
    return grafico_RG 

def SVM_Cross_validation(X, target):
    resultados = [] # Initialize an empty list to store the results
    C_values = [0.1, 1, 10, 100]
    kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
    gamma_values = ['scale', 'auto']
    contador=0
    for c in C_values:
        for kernel in kernel_values:
            for gamma in gamma_values:
                st.write('aa')
                model = svm.SVC(kernel=kernel, C=c, gamma=gamma)
                score = cross_val_score(model, X, target, cv=10, scoring='accuracy')
                contador+=1
                st.write('Contador:',contador)
                resultados.append({'C': c,
                               'Kernel': kernel, 
                                   'Gamma': gamma,
                                   'Accuracy': np.mean(score),
                                   'STD': np.std(score)
                                   })
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("Grafico_SVM_CV.csv", index=False)
    grafico_RG = alt.Chart(df_resultados).mark_circle().encode(
    x=alt.X('Accuracy:Q', scale=alt.Scale(zero=False)),
    y=alt.Y('STD:Q', scale=alt.Scale(zero=False)),
    color='Gamma:N',
    size='Test Size',
    tooltip=['Accuracy','Precision','Test Size','Random State','C','Kernel','Gamma','Recall','Specificity']
    ).interactive().properties(height=800)
    return grafico_RG



def SVM_GS(X_train,y_train):


    # Define the parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.1, 1]
    }

    # Initialize the SVM model
    model = svm.SVC()

    # Initialize the grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    # Convert the cv_results_ to a DataFrame and return it
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv("Grafico_SVM_GS.csv", index=False)
    return results_df