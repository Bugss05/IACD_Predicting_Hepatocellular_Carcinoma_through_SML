
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import altair as alt
from sklearn.metrics import recall_score
import streamlit as st

from sklearn.model_selection import cross_validate
def Grafico_DC_HP(X,y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Define the parameter ranges
    depths = range(6,14 )
    min_samples_splits = range(2, 11)
    min_samples_leafs = range(1, 11)
    test_sizes = [ 0.15 , 0.2 , 0.25 , 0.3 , 0.35 , 0.4 ]
    random_states = [42, 123, 456,89]  # List of different random states
    # Initialize a DataFrame to store the results
    results = []
     


    for depth in depths:
        for min_samples_split in min_samples_splits:
            for min_samples_leaf in min_samples_leafs:
                for test_size in test_sizes:
                    for random_state in random_states:
                    # Split the data
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                    # Create and train the classifier
                        clf = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                        clf.fit(X_train, y_train)
                        
                    # Make pre dictions
                        y_pred = clf.predict(X_test)

                    # Calculate accuracy, precision and recall
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                    # Calculate specificity
                        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                        specificity = tn / (tn + fp)
                    
                    # Store the results
                        results.append({
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'Specificity': specificity,
                        'Depth': depth,
                        'Min_Samples_Split': min_samples_split,
                        'Min_Samples_Leaf': min_samples_leaf,
                        'Test_Size': test_size,
                        'Random_State': random_state
                    })
    results= pd.DataFrame(results)
    results.to_csv("Grafico_DC_HP_MV.csv", index=False)
    chart = alt.Chart(results).mark_circle().encode(
    x=alt.X('Precision:Q', scale=alt.Scale(zero=False)),
    y=alt.Y('Recall:Q', scale=alt.Scale(zero=False)),
    color=alt.Color('Test_Size:N'),
    size='Depth:N',
    tooltip=['Depth', 'Min_Samples_Split', 'Min_Samples_Leaf', 'Test_Size','Accuracy', 'Precision', 'Recall','Specificity','Random_State']
    ).interactive().properties(height=800)
    return chart
def Grafico_DC_CV(X,y):


    # Assuming X is your feature set and y is the target variable
    # X, y = load_your_data()

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Define the parameter ranges
    depths = range(6, 14)
    min_samples_splits = range(2, 11)
    min_samples_leafs = range(1, 11)

    # Initialize an empty list to store the results
    results_list = []

    # Loop over the parameter ranges
    for depth in depths:
        for min_samples_split in min_samples_splits:
            for min_samples_leaf in min_samples_leafs:
                # Create the classifier
                clf = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                
                # Perform cross-validation
                cv_results = cross_validate(clf, X, y, cv=5, scoring=('accuracy', 'recall', 'precision'))
                
                # Store the results
                results_list.append({
                        'Accuracy': cv_results['test_accuracy'].mean(),
                        'Precision': cv_results['test_precision'].mean(),
                        'Recall': cv_results['test_recall'].mean(),
                        'Depth': depth,
                        'Min_Samples_Split': min_samples_split,
                        'Min_Samples_Leaf': min_samples_leaf
                        
                        
                        })

    # Convert the list of results into a DataFrame
    resultados_DC_CV = pd.DataFrame(results_list)
    resultados_DC_CV.to_csv("Grafico_DC_OT_MV.csv", index=False)
    grafico_DC_CV = alt.Chart(resultados_DC_CV).mark_circle().encode(
        x=alt.X('Recall:Q', scale=alt.Scale(zero=False)),
        y=alt.Y('Accuracy:Q', scale=alt.Scale(zero=False)),
        color=alt.Color('Depth:N'),
        size=alt.value(600),
        tooltip=['Depth', 'Min_Samples_Split', 'Min_Samples_Leaf', 'Accuracy', 'Precision', 'Recall']
    ).interactive().properties(height=800)
    return grafico_DC_CV
