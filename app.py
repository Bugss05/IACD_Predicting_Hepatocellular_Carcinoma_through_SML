
import streamlit as st
import pandas as pd 
import numpy as np
from notebook import *
st.set_page_config(layout="wide")
df=pd.read_csv("hcc_dataset.csv")# Abrir o data-set



st.markdown('''# <font size="65">Predicting Hepatocellular Carcinoma through Supervised Machine learning</font>

Trabalho realizado por:

* Afonso Coelho (FCUP_IACD:202305085)
* Diogo Amaral (FCUP_IACD:202305187) 
* Miguel Carvalho (FCUP_IACD:202305229)

******

## <font size="40">0.Introdução</font>

Neste projeto, o propósito é abordar um caso real do conjunto de dados do Carcinoma Hepatocelular (HCC). O referido conjunto de dados HCC foi coletado no Centro Hospitalar e Universitário de Coimbra (CHUC) em Portugal, e consiste em dados clínicos reais de pacientes diagnosticados com HCC.<br>

O objetivo primordial deste projeto é desenvolver um algoritmo SML (Supervised Machine Learning) capaz de determinar a possibilidade de sobrevivencia dos pacientes após 1 ano do diagnóstico (por exemplo, "sobrevive" ou "falece"). <br>
Os métodos de Machine learning que iremos utilizar são:
* KNN
* Decision Tree
* Random Forest(debativel)
* SVM(debativel)
* Logistic Regression(debativel)
            
            
            
### 0.1 Código do projeto
A base do código envolve a criação de uma classe ``Dataset`` que contém métodos para a identificação de missing values, outliers etc.
É o ``coração do projeto `` e foi utilizado ao longo de todo seu percurso. 
Em todos os passos irão estar presentes ``snippets de código`` que foram utilizados para a resolução do problema.<br>
            
Para esclarecer como a classe Dataset foi construída, eis o respetivo código inicial (BuilderData e _init_):  
            

''',unsafe_allow_html=True)

st.code('''
        class Dataset:
            def __init__(self, df, missing_values):
                self.df = df
                self.missing_values = missing_values
                
            (...)

            @classmethod #este classmethod funciona como um construtor alternativo e construir um dataframe a partir de um arquivo CSV

            def builderData(cls, df, missing_values):# df= data-set, missing_values= valores que representam missing values
                try:
                    if not isinstance(df, pd.DataFrame):# Confirmar se é um DataFrame
                        df = pd.read_csv(df)
                    df = df.copy()# Evitar copiar o Datarameo original
                    return cls(df, missing_values)
                except (FileNotFoundError, pd.errors.ParserError):
                    # Teve um erro: file not found ou parsing error
                    print(f"Erro: Não conseguiu ler a data de {df}.")
                    raise




''', language="python")

st.markdown('''       
            *******
<br><br>
## <font size="40">1.Data-prep</font>
Conforme a página avança iremos exibir os nossos passos *step-by-step* em como resolvemos este problema e aplicamo-lo
Para uma boa análise de dados, é essencial uma boa preparação dos dados, logo foi decidido que iriamos dividir esta parte do projeto nas seguintes partições:
* Missing values
* Identificação de outliers 
* Análise de variaveis 
* Estatisticas descritivas 

<br>
Em cada tópico iremos extrair as nossas conclusões e explicá-las conformalmente.
            
### 1.1 Missing values
Foi necessário identificar os missing values no data-set, de modo a eliminar certas variáveis com percentagens de missing values muito altas. Assim será mais fácil de prever os valores das variáveis restantes. 
            
        
''',unsafe_allow_html=True)



col1, col2, col3 = st.columns(spec=[0.75,0.05, 0.2])
#construir a tabela de missing values
data = Dataset.builderData(df, "?")
percentagem= data.missing_values_percentagem()#Percentagem de missing values
chart_data = pd.DataFrame(
   {"Variaveis": data.df.columns, "percentagem de missing_values": percentagem}
)

with col1:#grafico de missing values
   col1.header("Gráfico de Missing values")
   st.markdown("<br>", unsafe_allow_html=True)
   col1.bar_chart(chart_data,x="Variaveis" ,y="percentagem de missing_values", color=["#FF0000"])  # Optional)

with col3:#tabelade missing values
   data = Dataset.builderData(df, "?")
   col3.header("Tabela de Missing values")
   col3.dataframe(data.pintarMissingValues(), use_container_width=True)#Tabela de Missing values


st.markdown(''' 
Conforme o gráfico decidimos eliminar as variaveis >35% nomeadamente ``Iron``,``SAT`` e  ``Ferritin`` pois seria difícil prever os valores destas variaveis.<br>

Eis o respetivo código:

''', unsafe_allow_html=True)
st.code('''    
    #para pintar os missing values na tabela
    def pintarMissingValues(self):#pintar a tabela de missing values

        if self.missing_values is not None:#se existirem missing values
            self.df.replace(self.missing_values, "NaN", inplace=True)#substituir missing values por string "NaN" devido a limitação do site 
            return self.df.style.applymap(lambda valor: "color: red;" if valor=="NaN" else "")#pintar missing values a vermelho
        else: return self.df #se não existirem missing values

    #para calcular a percentagem de missing values
    def missing_values_percentagem(self):#Percentagem de missing values

        self.df.replace(self.missing_values, np.nan, inplace=True)#substituir missing values por NaN e nao string "NaN"
        missing_values_percentages = self.df.isnull().mean() * 100#calcular a percentagem de missing values
        return missing_values_percentages.tolist()#retornar a percentagem de missing values''', language="python")

st.markdown('''
<br>

******

### 1.2 Identificação de outliers
Para identificar os outliers, foi necessário calcular o IQR (Interquartile Range) e os limites inferior e superior. Definimos os outliers para depois os saber agrupar usando ``KNN``. 




''',unsafe_allow_html=True)
st.latex(r'''\begin{align*}
\text{IQR} &= Q3 - Q1 \\
\text{Limite Inferior} &= Q1 - 1.5 \times \text{IQR} \\
\text{Limite Superior} &= Q3 + 1.5 \times \text{IQR} \\
\text{Outlier se} & \quad x < Q1 - 1.5 \times \text{IQR} \quad \lor \quad x > Q3 + 1.5 \times \text{IQR}
\end{align*}''')


data = Dataset.builderData(df, "?")
st.header("Tabela de Outliers")

grafico_outliers= data.outliers('style')
st.dataframe(grafico_outliers, height=900,use_container_width=True)#Tabela de Outliers
st.markdown('''
Apesar dos outliers representarem valores fora do normal decidimos nao os altdulterar pois estes valores representam diversidade de dados e podem ser importantes para a previsão de sobrevivencia dos pacientes num caso de resultados semelhantes.<br>
Eis o respetivo código:




''',unsafe_allow_html=True)

st.code('''    
    def outliers(self):
        # Selecionar apenas as colunas numéricas
        numeric_df = self.df_num()

        colunas_numericas = numeric_df.columns
        outliers = set()
        for coluna in colunas_numericas:#calcular os outliers usando o IQR
            q1 = numeric_df[coluna].quantile(0.25)
            q3 = numeric_df[coluna].quantile(0.75)
            iqr = q3 - q1
            limite_inferior = q1 - 1.5 * iqr
            limite_superior = q3 + 1.5 * iqr
            for index, value in numeric_df[coluna].items():#adicionar outliers ao set
                if value < limite_inferior or value > limite_superior:
                    outliers.add((coluna, index))

        styled_df = self.pintarOutliers(numeric_df, outliers)# Aplicar highlight aos outliers

        return styled_df''', language="python")
st.markdown('''
<br>
            
******
            
### 1.3 Estatísticas descritivas
Neste tópico iremos analisar as estatísticas padrão do data-set para melhor entender os dados nos apresentados como:
      
    ''',unsafe_allow_html=True)

col10,col1,col6,col2,col7,col3,clo8,col4,col9,col5,col11= st.columns(spec=[0.05,0.08,0.0225,0.08,0.02250,0.1,0.0225,0.08,0.02250,0.08,0.05])

with col1:#grafico da media

    col1.header("Média")
    st.markdown("<br>", unsafe_allow_html=True)
    data = Dataset.builderData("hcc_dataset.csv", "?")
    tabela = data.df_num().mean(numeric_only=True).to_frame("")
    st.dataframe(tabela, height=840, use_container_width=True)  # Tabela da media

with col2:#grafico da media

    col2.header("Mediana")
    st.markdown("<br>", unsafe_allow_html=True)
    data = Dataset.builderData("hcc_dataset.csv", "?")
    tabela = data.df_num().median(numeric_only=True).to_frame("")
    st.dataframe(tabela, height=840, use_container_width=True)  # Tabela da media

with col3:#grafico desvio padrao

    col3.header("Desvio Padrão")
    st.markdown("<br>", unsafe_allow_html=True)
    data = Dataset.builderData("hcc_dataset.csv", "?")
    tabela = data.df_num().std(numeric_only=True).to_frame("")
    st.dataframe(tabela, height=840, use_container_width=True)  # Tabela da media

with col4:#grafico da media

    col4.header("Assimetria")
    st.markdown("<br>", unsafe_allow_html=True)
    data = Dataset.builderData("hcc_dataset.csv", "?")
    tabela = data.df_num().skew(numeric_only=True).to_frame("")
    st.dataframe(tabela, height=840, use_container_width=True)  # Tabela da media

with col5:#grafico da media
    col5.header("Curtose")
    st.markdown("<br>", unsafe_allow_html=True)
    data = Dataset.builderData("hcc_dataset.csv", "?")
    tabela = data.df_num().kurtosis(numeric_only=True).to_frame("")
    st.dataframe(tabela, height=840, use_container_width=True)  # Tabela da media

st.markdown('''
<br>
            
******
            
### 1.3  Heterogeneous Euclidean-Overlap Metric (HEOM)
Para Tratar de missng values, foi necessário calcular a distância entre dois pacientes, usando a métrica HEOM (Heterogeneous Euclidean-Overlap Metric).
Esta métrica é usada para calcular a distância entre dois pacientes, mesmo que tenham diferentes tipos de variáveis(categóricas e numéricas).<br>
<br><br>
Este calculo baseia se todo em ``semelhanças`` que se relacionam em posteormente em ``distancias``. Quantas mais parecenças menos distantes. <br>
<br><br>
            
>Neste primeiro passo inicia-se uma tripla condição:
>* Se x ou y são desconhecidos, a distância é 1
>* Se a coluna ``a`` é categórica, faz-se um cálculo simples de overlap das variáveis .
>* Por fim se a coluna ``a`` é numérica, calcula-se a diferença relativa entre x e y na função rn_diff.
      
    ''',unsafe_allow_html=True)

st.latex(r'''
d_a(x, y) = 
\begin{cases} 
1 & \text{se} x \text{ ou } y \text{ é desconhecido; else} \\
\text{overlap}(x, y) & \text{se } a \text{ é categórico,else} \\
\text{rn\_diff}_a(x, y) & \text{senão}
\end{cases}
''')

st.markdown('''
    <br><br>
            
    >Como as variáveis são categóricas só as podemos avaliar quanto à sua igualdade: ``iguais`` ou ``difentes``. Por isso esta funçao é apenas um calculo binario básico .

 ''', unsafe_allow_html=True)

st.latex(r'''
overlap(x, y) = 
\begin{cases} 
0 & \text{se } x = y \\
1 & \text{senão}
\end{cases}
''')

st.markdown('''
    <br><br>
            
    >Nas variáveis numéricas pode-se fazer um desvio relativo entre as variáveis x e y. Assim o valor é reparemetrizado e evitar disperção de valores e aplicar uma avaliação justa.''', unsafe_allow_html=True)

st.latex(r'''\text{rn\_diff}_a(x, y) = \frac{|x-y|}{\text{range}_a}
''')

st.latex(r'''\text{range}_a(x, y) = \text{max}_a - \text{min}_a''')
#codigo para os calculos utilizados na metrica HEOM
st.markdown('''<br><br>
            
>Finalmente, aplica-se formula final que faz o calculo Euclideano em todas as distancias que conclui assim a fórmula HEOM. 

''', unsafe_allow_html=True)
st.latex(r'''\text{HEOM}(x, y) = \sqrt{\sum_{a=1}^{m} d_a(x_a, y_a)^2}
''')

st.markdown('''
Este Cálculo é muito demorado devido a todas as suas operações. Só neste Dataset existem 50 variáveis que têm de ser comparadas entre cada par de pacientes.
             
Todos estes loops contribuem para um aumento da *Time complexity* que acaba por resultar  <span style="color: red; font-weight: bold;">${O_n(n^2*m)}$</span> no qual n é o nr de variáveis e m o número de pacientes . <br>
Eis a respetiva tabela HEOM: <br><br>''', unsafe_allow_html=True)

st.code('''
    def tabelaHEOM(self):
        self.df = self.replace_nan_with_none()#Trocar missing values para none
        tabela = pd.DataFrame()
        for i in range(len(self.df)):
            lista = []
            for j in range(len(self.df)):#Não interessa comparar pares de pacientes duas vezes
                if i >= j:
                    lista.append("X")# colocar x por motivos estéticos
                else:
                    lista.append(self.HEOM(i, j))# lista de um paciente em calculo HEOM

            tabela = pd.concat([tabela, pd.DataFrame({i: lista})], axis=1)#adicionar a lista à tabela
        return tabela
    
    def HEOM(self, paciente_1, paciente_2): #Heterogeneous Euclidean-Overlap Metric
        soma = 0
        for feature in self.df.columns:# iterar sobre as V
            distancia = self.distanciaGeral(feature, paciente_1, paciente_2)# calcular a sua "distancia"
            soma += distancia**2
        soma= soma**(1/2)
        return soma
    
    def distanciaGeral(self, feature:str, paciente_1:int, paciente_2:int)->int:
        try :#Se a variavel for numerica vem para aqui
            #distancia normalizada
            valorPaciente_1 = float(self.df.loc[paciente_1, feature])
            valorPaciente_2 = float(self.df.loc[paciente_2, feature])
            numeric_feature = pd.to_numeric(self.df[feature], errors='coerce')
            return abs(valorPaciente_1 - valorPaciente_2) / (numeric_feature.max() - numeric_feature.min())# retornar a range 
        except :#Se a variavel for categorica vem para aqui
            valorPaciente_1 = self.df.loc[paciente_1, feature]
            valorPaciente_2 = self.df.loc[paciente_2, feature]
            if valorPaciente_1 == valorPaciente_2 and  not pd.isna(valorPaciente_1):#Se forem iguais e não forem missing values
                return 0
            else: 
                return 1

''', language="python")


st.header("Tabela com missing values substituidos")
data = Dataset.builderData("Tabela_sem_missing_values_3.csv", "?")
st.dataframe(data.df, height=840, use_container_width=False)  # Tabela da media

col1, col2,col3,col4 = st.columns(spec=[0.2,0.1,0.1,0.2])
with col2:
    st.header("Tabela com missing values ")
    data = Dataset.builderData("hcc_dataset.csv", "?")
    tabela1 = data.df_num().mean().to_frame("Média")
    st.dataframe(tabela1, height=840, use_container_width=False)  # Tabela da media

with col3:
    st.header("Tabela com missing values substituidos")
    data = Dataset.builderData("Tabela_sem_missing_values_3.csv", "?")
    tabela2 = data.df.mean(numeric_only=True).to_frame("")
    st.dataframe(tabela2, height=840, use_container_width=False)  # Tabela da media
with col4:
    st.header("Desvio  Relativo (%)")
    # Calculate the relative deviation
    relative_deviation = (abs(tabela1.iloc[:, 0] - tabela2.iloc[:, 0]) / tabela1.iloc[:, 0]) * 100

    # Convert the relative deviation to a DataFrame and name the column
    relative_deviation = relative_deviation.to_frame("Desvio Relativo (%)")

    # Display the relative deviation DataFrame
    st.dataframe(relative_deviation, height=840)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
import altair as alt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


data = Dataset.builderData("Tabela_OT_antes_MV.csv", "?")
X = data.df.drop(columns=['Class']).dropna()
y = data.df['Class']


test_sizes = [0.1,0.15,0.20,0.25, 0.3,0.35, 0.4]  # List of different test sizes
random_states = [42, 123, 456]  # List of different random states
k_neighbors = range(1, 99)  # Range of k neighbors

results = []  # List to store the results
#_______________________________________________________________________________________________________________________



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=456)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=4)
fit = knn.fit(X_train, y_train)
accuracy = accuracy_score(y_test, fit.predict(X_test))


#_______________________________________________________________________________________________________________________



k=list(range(1,100))
accuracy=[]
for i in k:
    knn = KNeighborsClassifier(n_neighbors=i)
    fit = knn.fit(X_train, y_train)
    accuracy.append(accuracy_score(y_test, fit.predict(X_test)))
df_acc = pd.DataFrame({'K Values': k, 'Accuracy Score': accuracy})  
alt_c = alt.Chart(df_acc).mark_circle().encode(
    alt.X('K Values:Q', scale=alt.Scale(zero=False)),
    alt.Y('Accuracy Score:Q', scale=alt.Scale(domain=[0.2, 0.8]))
).interactive()
st.header("Gráfico de KNN por Hyperparameter antes de outliers")
st.altair_chart(alt_c, use_container_width=True)  



#_______________________________________________________________________________________________________________________



# Reduce the dimensionality of your data to 2 dimensions using PCA
pca = PCA(n_components=2)
# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_pca = pca.fit_transform(X)

# Create a DataFrame with the data
df_pca = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Class': y})
df_pca['Class'] = df_pca['Class'].map({1: 'Vive', 0: 'Morre'})
# Create the chart with scaled x values
alt_c = alt.Chart(df_pca).mark_circle().encode(
    alt.X('PC1:Q', scale=alt.Scale(domain=[-5.5, 7.5])),
    alt.Y('PC2:Q',scale=alt.Scale(domain=[-6, 10])),
    color='Class'
).interactive().properties(height=800)

# Display the chart
st.header("Gráfico de PCA (Principal Component Analysis) antes da modificação de outliers")
st.altair_chart(alt_c, use_container_width=True, theme=None)
st.markdown('''<br>
            
Como sao muitas variaveis e muitos pacientes muita da data é perdida e por isso o grafico tender para esta forma linear. <br>
Fizemos uma especie de Hyperparameter tuning para encontrar o melhor valor de K para o KNN. <br>            
            ''',unsafe_allow_html=True)



#_______________________________________________________________________________________________________________________



k_values = [i for i in range (1,100)]
scores = []

scaler = StandardScaler()
X = scaler.fit_transform(X)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=10)
    scores.append(np.mean(score))
df_scores = pd.DataFrame({'K Values': k_values, 'Accuracy Score': scores})

alt_c = alt.Chart(df_scores).mark_circle().encode(
    alt.X('K Values:Q', scale=alt.Scale(zero=False)),
    alt.Y('Accuracy Score:Q', scale=alt.Scale(domain=[0.5, 0.8]))
).interactive().properties(height=800)
st.header("Gráfico de KNN por cross-validation antes de outliers 10 folds")
st.altair_chart(alt_c, use_container_width=True)



#_______________________________________________________________________________________________________________________



results= Dataset.builderData("results_DCHP.csv", "?")
# Create the Altair chart
chart = alt.Chart(results.df).mark_circle().encode(
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



from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

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
            cv_scores = cross_val_score(clf, X, y, cv=5)

            # Store the results
            results_list.append({
                'Mean_CV_Score': cv_scores.mean(),
                'Std_CV_Score': cv_scores.std(),
                'Depth': depth,
                'Min_Samples_Split': min_samples_split,
                'Min_Samples_Leaf': min_samples_leaf
            })

# Convert the list of results into a DataFrame
resultados_DC_CV = pd.DataFrame(results_list)
grafico_DC_CV = alt.Chart(resultados_DC_CV).mark_circle().encode(
    x=alt.X('Mean_CV_Score:Q', scale=alt.Scale(zero=False)),
    y=alt.Y('Std_CV_Score:Q', scale=alt.Scale(zero=False)),
    color=alt.Color('Depth:N'),
    size=alt.value(600),
    tooltip=['Mean_CV_Score', 'Std_CV_Score', 'Depth', 'Min_Samples_Split', 'Min_Samples_Leaf']
).interactive().properties(height=800)
st.header("Gráfico de Decision Tree por Cross-Validation antes de outliers")
st.altair_chart(grafico_DC_CV, use_container_width=True, theme=None)



#_______________________________________________________________________________________________________________________



from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Assuming X is your feature set and y is the target variable
# X, y = load_your_data()

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the parameter grid
param_grid = {
    'max_depth': np.arange(1, 11),
    'min_samples_split': np.arange(2, 11),
    'min_samples_leaf': np.arange(1, 11)
}

# Create the classifier
clf = DecisionTreeClassifier()

# Create the GridSearchCV object
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)

# Fit the GridSearchCV object to the data
grid_search.fit(X, y)

# Get the best parameters
best_params = grid_search.best_params_

# Get the best score
best_score = grid_search.best_score_

# Create a DataFrame to store the best results
results = pd.DataFrame([best_params])
results['Best_Score'] = best_score

# Display the results
print(results)



#_______________________________________________________________________________________________________________________



from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Assuming X is your feature set and y is the target variable
# X, y = load_your_data()

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the parameter grid
param_grid = {
    'max_depth': np.arange(1, 11),
    'min_samples_split': np.arange(2, 11),
    'min_samples_leaf': np.arange(1, 11)
}

# Create the classifier
clf = DecisionTreeClassifier()

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=1000, cv=5, random_state=456)

# Fit the RandomizedSearchCV object to the data
random_search.fit(X, y)

cv_results = random_search.cv_results_

# Create a DataFrame to store the best results
results = pd.DataFrame(cv_results).sort_values('rank_test_score')
st.dataframe(results)



#_______________________________________________________________________________________________________________________


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score
import pandas as pd

# Assume X is your features and y is your target
test_sizes = [ 0.15 , 0.2 , 0.25 , 0.3 , 0.35 , 0.4 , 0.45 , 0.5]
random_states = [12, 23, 35, 47, 52, 58, 63, 75, 84, 92, 101, 112, 118, 25, 99]
weights = ['balanced', None]

results = []

for test_size in test_sizes:
    for random_state in random_states:
        for weight in weights:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Initialize the Logistic Regression model
            model = LogisticRegression(class_weight=weight)

            # Fit the model to the training data
            model.fit(X_train, y_train)

            # Use the model to make predictions on unseen data
            predictions = model.predict(X_test)

            # Calculate accuracy and precision
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions)

            # Append results to the results list
            results.append({
                'Accuracy': accuracy,
                'Precision': precision,
                'Test Size': test_size,
                'Random State': random_state,
                'Weight': weight
            })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
st.dataframe(results_df)
grafico_RG = alt.Chart(results_df).mark_circle().encode(
    x=alt.X('Precision:Q', scale=alt.Scale(zero=False)),
    y=alt.Y('Accuracy:Q', scale=alt.Scale(zero=False)),
    color='Weight:N',
    size='Test Size',
    tooltip=['Accuracy','Precision','Test Size','Random State','Weight']
).interactive().properties(height=800)
st.header("Gráfico de Logistic Regression por Hyperparameter depois de outliers")
st.altair_chart(grafico_RG, use_container_width=True, theme=None)
#_______________________________________________________________________________________________________________________


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Assuming X is your feature set and y is the target variable
# X, y = load_your_data()

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the parameter grid
param_grid = {
    'C': np.logspace(-4, 4, 20),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
    'class_weight': ['balanced', None],
    'random_state': [42, 52, 62]
}

# Create the classifier
clf = LogisticRegression()

# Create the GridSearchCV object
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)

# Fit the GridSearchCV object to the data
grid_search.fit(X, y)

# Get the cross-validation results
cv_results = grid_search.cv_results_

# Lines 714-729: Your existing code

# Convert the results to a DataFrame
results_df = pd.DataFrame(cv_results)

# Add a new column 'Scores' that contains the scores of each iteration
results_df['Scores'] = cv_results['mean_test_score']

# Add a new column 'Std Scores' that contains the standard deviation of the scores of each iteration
results_df['Std Scores'] = cv_results['std_test_score']

# Display the results
st.dataframe(results_df)
import altair as alt

# Convert the 'param_' columns from the results_df DataFrame to a more readable format
params_df = results_df[['param_C', 'param_penalty', 'param_solver', 'param_class_weight', 'param_random_state']].applymap(str)

# Create a new column 'params' in results_df that contains the combined parameters
results_df['params'] = params_df.apply(lambda row: '_'.join(row.values), axis=1)

# Melt the DataFrame to get a long format where each row is a unique combination of parameters and random state
melted_df = results_df.melt(id_vars=['params','Std Scores'], value_vars=['param_random_state'], var_name='Random State', value_name='Mean Test Score')

# Create a heatmap
heatmap = alt.Chart(melted_df).mark_rect().encode(
    x='Std Scores:Q',
    y='Mean Test Score:Q',
    color='Random State:Q',
    tooltip=['Std Scores', 'Random State', 'Mean Test Score']
)

# Display the heatmap
st.altair_chart(heatmap, use_container_width=True, theme=None)

#_______________________________________________________________________________________________________________________\


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
results = []

from sklearn.metrics import recall_score

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

                # Calculate accuracy, precision and recall
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')

                # Store the results
                results.append({
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'Depth': depth,
                    'Min_Samples_Split': min_samples_split,
                    'Min_Samples_Leaf': min_samples_leaf,
                    'Test_Size': test_size
                })
results= pd.DataFrame(results)
results.to_csv("results.csv", index=False)

# Create the Altair chart
chart = alt.Chart(results).mark_circle().encode(
    x=alt.X('Precision:Q', scale=alt.Scale(zero=False)),
    y=alt.Y('Recall:Q', scale=alt.Scale(zero=False)),
    color=alt.Color('Test_Size:N'),
    size='Depth:N',
    tooltip=['Depth', 'Min_Samples_Split', 'Min_Samples_Leaf', 'Test_Size','Accuracy', 'Precision', 'Recall']
).interactive().properties(height=800)

# Display the chart
st.header("Gráfico de Decision Tree por Hyperparameter antes de outliers")
st.altair_chart(chart, use_container_width=True, theme=None)