'''from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.discriminant_analysis import StandardScaler
import altair as alt
import streamlit as st
import pandas as pd 
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.discriminant_analysis import StandardScaler
import altair as alt

data = Dataset.builderData("Tabela_sem_missing_values_3.csv", "?")
data1= data.categorical_to_numerical()
data= Dataset.builderData(data1, "?")
X = data.df.drop(columns=['Class']).dropna()
y = data.df['Class']

test_sizes = [0.1,0.15,0.20,0.25, 0.3,0.35, 0.4]  # List of different test sizes
random_states = [42, 123, 456]  # List of different random states
k_neighbors = range(1, 99)  # Range of k neighbors

results = []  # List to store the results

for test_size in test_sizes:
    for random_state in random_states:
        for k in k_neighbors:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            knn = KNeighborsClassifier(n_neighbors=k)
            fit = knn.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, fit.predict(X_test))
            results.append((accuracy, test_size, random_state, k))

results.sort(reverse=True)  # Sort the results in descending order by accuracy

for accuracy, test_size, random_state, k in results:
    st.write("Test Size:", test_size, "Random State:", random_state, "K Neighbors:", k, "Accuracy:", accuracy)

k_values = [i for i in range (1,165)]
scores = []

scaler = StandardScaler()
X = scaler.fit_transform(X)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=5)
    scores.append(np.mean(score))
df_scores = pd.DataFrame({'K Values': k_values, 'Accuracy Score': scores})

alt_c = alt.Chart(df_scores).mark_circle().encode(
    alt.X('K Values:Q', scale=alt.Scale(zero=False)),
    alt.Y('Accuracy Score:Q', scale=alt.Scale(domain=[0.5, 0.8]))
).interactive()

st.altair_chart(alt_c, use_container_width=True)'''



from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.discriminant_analysis import StandardScaler
import altair as alt
from notebook import *

data = Dataset.builderData("Tabela_sem_missing_values_3.csv", "?")
data1= data.categorical_to_numerical()
data= Dataset.builderData(data1, "?")
X = data.df.drop(columns=['Class']).dropna()
y = data.df['Class']

test_sizes = [0.1,0.15,0.20,0.25, 0.3,0.35, 0.4]  # List of different test sizes
random_states = [42, 123, 456]  # List of different random states
k_neighbors = range(1, 99)  # Range of k neighbors

results = []  # List to store the results

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=456)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=4)
fit = knn.fit(X_train, y_train)
accuracy = accuracy_score(y_test, fit.predict(X_test))
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce the dimensionality of your data to 2 dimensions using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Predict the classes of the data
y_pred = fit.predict(X)

# Create a scatter plot of the data
plt.figure(figsize=(10, 10))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred)
plt.legend(*scatter.legend_elements(), title="Classes")
plt.show()




from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import altair as alt

# Assuming X is your feature set and y is the target variable
# X, y = load_your_data()

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the parameter ranges
depths = range(1, 20)
min_samples_splits = range(2, 11)
min_samples_leafs = range(1, 11)
test_sizes = [0.1, 0.15 , 0.2 , 0.25 , 0.3 , 0.35 , 0.4 , 0.45 , 0.5]

# Initialize a DataFrame to store the results
results = pd.DataFrame(columns=['Accuracy', 'Precision', 'Depth', 'Min_Samples_Split', 'Min_Samples_Leaf', 'Test_Size'])

# Loop over the parameter ranges
for depth in depths:
    for min_samples_split in min_samples_splits:
        for min_samples_leaf in min_samples_leafs:
            for test_size in test_sizes:
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=456)

                # Create and train the classifier
                clf = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                clf.fit(X_train, y_train)

                # Make predictions
                y_pred = clf.predict(X_test)

                # Calculate accuracy and precision
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')

                # Store the results
                results = results.append({
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Depth': depth,
                    'Min_Samples_Split': min_samples_split,
                    'Min_Samples_Leaf': min_samples_leaf,
                    'Test_Size': test_size
                }, ignore_index=True)
results.to_csv("results.csv", index=False)
# Create the Altair chart
chart = alt.Chart(results).mark_circle().encode(
    x=alt.X('Precision:Q', scale=alt.Scale(zero=False)),
    y=alt.Y('Accuracy:Q', scale=alt.Scale(zero=False)),
    color=alt.Color('Test_Size:N'),
    size='Depth:N',
    tooltip=['Depth', 'Min_Samples_Split', 'Min_Samples_Leaf', 'Test_Size','Accuracy', 'Precision']
).interactive().properties(height=800)

# Display the chart
st.header("Gráfico de Decision Tree por Hyperparameter antes de outliers")
st.altair_chart(chart, use_container_width=True, theme=None)
#_______________________________________________________________________________________________________________________