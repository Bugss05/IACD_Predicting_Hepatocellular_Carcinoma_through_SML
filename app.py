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

grafico_outliers= data.outliers()
st.dataframe(grafico_outliers, height=750,use_container_width=True)#Tabela de Outliers
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
>* Se a coluna ``a`` é nominal (variável categórica), faz-se um cálculo simples de overlap das variaveis .
>* Por fim se a coluna ``a`` é numérica, calcula-se a diferença relativa entre x e y na função rn_diff.
      
    ''',unsafe_allow_html=True)

st.latex(r'''
d_a(x, y) = 
\begin{cases} 
1 & \text{if } x \text{ or } y \text{ is unknown; else} \\
\text{overlap}(x, y) & \text{if } a \text{ is nominal,else} \\
\text{rn\_diff}_a(x, y) & \text{otherwise}
\end{cases}
''')

st.markdown('''
    <br><br>
            
    >Como as variáveis são categóricas só as podemos avaliar quanto à sua igualdade: ``iguais`` ou ``difentes``. Por isso esta funçao é apenas um calculo binario básico .

 ''', unsafe_allow_html=True)

st.latex(r'''
overlap(x, y) = 
\begin{cases} 
0 & \text{if } x = y \\
1 & \text{otherwise}
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
            
>Finalmente, aplica-se formula final que adiciona o quadrado das ditancias de cada coluna e dps fazendo a sua raiz quadrada que conclui assim a formula HEOM. 

''', unsafe_allow_html=True)
st.latex(r'''\text{HEOM}(x, y) = \sqrt{\sum_{a=1}^{m} d_a(x_a, y_a)^2}
''')

st.markdown('''Eis a respetiva tabela HEOM:<br><br>''', unsafe_allow_html=True)

data = Dataset.builderData("Tabela_HEOM.csv", "?")
st.dataframe(data.df, height=840, use_container_width=True)







