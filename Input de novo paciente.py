import streamlit as st
import pandas as pd 
import numpy as np
import altair as alt
from notebook import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.set_page_config(layout="wide")
data = Dataset.builderData("Tabela_sem_missing_values_3.csv", "?")
tabela = data.df
tabela = data.df_num()
num_colunas = tabela.columns

# Load dataset
data = pd.read_csv("hcc_dataset.csv")

# Define the columns to ignore
ignore_columns = ["Class"]

# Get numeric and categorical columns
cat_colunas = [col for col in data.columns if col not in num_colunas and col not in ignore_columns]

# Combine all columns while preserving the original order
all_columns = [col for col in data.columns if col not in ignore_columns]

# Split the columns into four segments while preserving the order
num_columns = len(all_columns)
segment_size = (num_columns + 3) // 4  # ceil division

segments = [all_columns[i * segment_size:(i + 1) * segment_size] for i in range(4)]

st.markdown('''
******
<br><br>
# Imputação de um paciente
<br>
''', unsafe_allow_html=True)

# List to store user inputs
features = []

# Create four columns in Streamlit
col1, col2, col3, col4 = st.columns(4)
columns = [col1, col2, col3, col4]

# Iterate over segments and columns
for i, col in enumerate(columns):
    with col:
        for segment in segments[i]:
            missing = "?"
            if segment in num_colunas or segment in ["Iron", "Sat", "Ferritin"]:
                user_input = st.text_input(f"Introduza um valor para {segment}")
                try:
                    # Attempt to convert input to a number
                    user_input_num = float(user_input)
                    features.append(user_input_num)
                except ValueError:
                    if user_input == missing:
                        features.append(missing)
                    else:
                        #Handle invalid input by appending None or a default value
                        features.append(None)
            else:
                valores = list(data[segment].unique())
                if missing not in valores:
                    valores.append(missing)
                valores = sorted(valores)
                option = st.selectbox(f"Introduza um valor para {segment}", valores)
                features.append(option)

st.markdown('''
<br>
''', unsafe_allow_html=True)

dicionario = {}          
dicionario["paciente_inputado"] = features
dfPacienteNovo = pd.DataFrame(dicionario)
dfPacienteNovo = dfPacienteNovo.transpose()
dfPacienteNovo.columns = all_columns
st.write(dfPacienteNovo)
agree = st.checkbox("Enviar relatório")
if agree:
    st.write("Relatório Enviado! Por favor aguarde pelo diagonóstico...")
    #utilizar método de classisificação
    #INPUTAR MISSING VALUES PELO DATASET
    
    dfPacienteNovo["Class"] = "Find"
    
    data = Dataset.builderData("Tabela_sem_missing_values_3.csv", "?")
    dfNovo_total = pd.concat([dfPacienteNovo, data.df],  ignore_index = True)
    
    #dfNovo_total é um df com todos os pacientes antigos e com o novo introduzido
    #novo paciente = linha 0
    
    dfNovo_total = Dataset.builderData(dfNovo_total, "?")
    dfNovo_total.df = dfNovo_total.fill_missing_values(3)
    


    dfNovo_total.df = dfNovo_total.categorical_to_numerical()
    temp = dfNovo_total.df

    # ------ MODELO DE CLASSIFICAÇÃO ----------
    data = Dataset.builderData("Tabela_OT_antes_MV.csv", "?")

    #data -> DF para dar fit ao modelo
    #temp -> DF com o novo paciente a ser classificado

    model = DecisionTreeClassifier(max_depth=13, min_samples_split=10, min_samples_leaf=9) #Parametros Otimizados
    scaler = StandardScaler()

    X = data.df.drop(columns=['Class']).dropna()
    y = data.df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42) #Parametros Otimizados
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model.fit(X_train, y_train)

    rowSeries = temp.drop(columns=['Class']).iloc[0]
    rowFrame = rowSeries.to_frame().T
    rowFrame = rowFrame.reindex(columns=X.columns, fill_value=0)
    rowScaled = scaler.transform(rowFrame)

    test = list(model.predict(rowScaled))[0]
    if list(model.predict(rowScaled))[0]:
        diagnostico = "sobrevive"
    else:
        diagnostico = "morre"

    st.write("O paciente " + diagnostico)

else:
    st.write("Por favor envie o relatório do paciente clicando na caixa acima.")
    #nao fazer nada
