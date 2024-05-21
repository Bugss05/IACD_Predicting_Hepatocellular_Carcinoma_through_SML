
# Reduce the dimensionality of your data to 2 dimensions using PCA
import pandas as pd
import altair as alt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

def Pca(X,target)->alt.Chart:
    pca = PCA(n_components=2)
# Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_pca = pca.fit_transform(X)

# Define the missing variables
    y = [0, 1]  # Replace with your actual data

# Create a DataFrame with the data
    df_pca = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Class': target})
    df_pca['Class'] = df_pca['Class'].map({1: 'Vive', 0: 'Morre'})
    df_pca.to_csv("Grafico_PCA_OT_MV.csv", index=False)
# Create the chart with scaled x values
    alt_c = alt.Chart(df_pca).mark_circle().encode(
    alt.X('PC1:Q', scale=alt.Scale(domain=[-5.5, 7.5])),
    alt.Y('PC2:Q',scale=alt.Scale(domain=[-6, 10])),
    color='Class'
).interactive().properties(height=800)
    return alt_c