
import streamlit as st
import pandas as pd 
import numpy as np
from notebook import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.preprocessing import StandardScaler
import altair as alt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from KNN_scrpt import *


st.set_page_config(layout="wide")
df=pd.read_csv("hcc_dataset.csv")# Abrir o data-set



st.markdown('''# <font size="65">Predicting Hepatocellular Carcinoma through Supervised Machine learning</font>

Trabalho realizado por:

* Afonso Coelho (FCUP_IACD:202305085)
* Diogo Amaral (FCUP_IACD:202305187) 
* Miguel Carvalho (FCUP_IACD:202305229)

******

# 0. Introdução

O `Carcinoma Hepatocelular (HCC)` surge do mesmo processo de destruição e multiplicação de células que leva à cirrose. É um tumor altamente maligno que apresenta uma taxa de crescimento exponencial, dobrando o seu tamanho a cada 180 dias (em média). Um tumor deste tipo mesmo em fase inicial do seu desenvolvimento condena o seu portador a uma esperança de vida de 8 meses. Já no seu estágio avançado, é expectável que o paciente viva por mais 3 meses. Este é um problema real que assuta muitos pacientes e famílias, mas a maior preocupação clínica é o `desconhecimento das causas e parâmetros deste desenvolvimento anormal`, operando no paciente como um assassino silencioso.

É devido a este desconhecimento que a medicina alia-se à ciência de dados pela busca de certezas. Quanto mais se conhecer sobre, mais rápida a atuação preventiva será. O rápido diagnóstico de um paciente pode ser o fator decisivo para que este sobreviva.

Debruçados sobre este cenário, comprometemo-nos ao objetivo primordial de desenvolver um `algoritmo SML (Supervised Machine Learning)` para classificar os doentes após 1 ano diagnóstico em dois possíveis resultados: `Sobrevive ou Morre`. A amostra deste estudo é um conjunto de dados - `dataset` - do `Centro Hospitalar e Universitário de Coimbra (CHUC)`.

Para o desenvolvimento algorítmico de Supervised Machine Learning, iremos ter por base os seguintes métodos:
-colocar metodos que utilizamos-

* KNN
* Decision Tree
* Random Forest
* Logistic Regression
<br><br>
            
### 0.1 Sobre o estudo

Tipicamente como data science é trabalhada, todo o código implementado foi desenvolvido em `Jupyter Notebook`.
 
Alguns trechos de código ser-lhe-ão apresentados ao longo do desta documentação, os quais são baseado numa `Class` que criamos chamada `Dataset`. Para além do foco primário do projeto, desejavamos que qualquer dataset pudesse ser convertido num `DataFrame` e que, posteriormente, pudesse ser processado e polido segundo os métodos que desenhamos,a fim a alimentar um algoritmo de machine learning.

Segue-se um exemplar da definição `init` e do construtor `BuilderData`:  
                      
   
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
            ********

<br><br>                  
# 1. Data Mining

Talvez o processo mais importante de todo o estudo. Um dos grandes problemas que qualquer data scientist enfrenta é crua forma como os dados são-lhe apresentados. Analogamente, suponhamos que o data set inicial é um minério ouro recém estraído, o qual tem pedaços de outras rochas e impurezas ao seu redor. O trabalho do mineiro é, entre outros, limpar, polir e extrair o máximo de ouro daquele minério: o nosso trabalho não é diferente. Uma boa data análise reproduz um bom resultado final. É nos encarregue analisar os dados, a forma como estes são apresentados e estão representados, relacionar variáveis, lidar com valores em falta, etc.

Neste projeto, no data set `hcc_dataset.csv`, cada `linha` do seu DataFrame representa `um conjunto de caraterísticas de um paciente`, e cada `coluna` representa uma caraterística singular, que chamar-lhes-emos de `atributo` ou `feature`. Estas caraterísticas no contexto do problema são a forma como o doente lidou com o carcinoma, desde caso manifestou `sintomas` até aos níveis de `hemoglobina` ou `ferro`. Conta-se `165 pacientes` com `50 atributos` diferentes, dos quais `27 categóricos` e `23 numéricos`.

Eis o `DataFrame` do nosso dataset: <br> ''',unsafe_allow_html=True)

DF=Dataset.builderData('hcc_dataset.csv', "?")  
st.dataframe(DF.df, height=840, use_container_width=False)  # Tabela da media





st.markdown(''' <br>    
Assim, de modo a ultrapassarmos esta fase corretamente, decidimos que a nossa `data analysis` guiar-se-ia pelos seguintes aspetos:

* Estatisticas descritivas básicas
* Análise de Variáveis
* Missing Values
* Outliers''',unsafe_allow_html=True)
            
data = Dataset.builderData("Tabela_sem_missing_values_3.csv", "?")
tabela = data.df
tabela = data.df_num()
num_colunas = tabela.columns

st.markdown('''
# Interação entre 2 features do DataFrame
<br>
''', unsafe_allow_html=True)

col2, col1 = st.columns([0.75,0.25])

with col1:
    x_axis = st.selectbox('Eixo X:', num_colunas, index = 0)
    y_axis = st.selectbox('Eixo Y:', num_colunas, index = 0)
with col2:
    scatter_plot = alt.Chart(data.df).mark_circle(size=60).encode(
        x=x_axis,
        y=y_axis,
        tooltip=[x_axis, y_axis]
    ).interactive().properties(height=600)

    # Display the chart in Streamlit
    st.altair_chart(scatter_plot, use_container_width=True)        
            






st.markdown('''       
            *******
<br>
            
## 1.1 Estatísticas Descritivas básicas

Um dos sistemas mais simples, no entanto dos mais efetivos, é a realização de uma `análise descritiva` de cada atributo. Entende-se por `estatísticas descritivas básicas` as seguintes estatísticas que são apenas aplicáveis a features com valores numéricas:

- Média
- Mediana
- Desvio Padrão
- Assimetria
- Curtose

Abaixo encontra-se tabelas que exprimem uma estatistica de um determinado atributo, bem como uma descrição suscinta da importância de cada estatística:

        
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
<br><br>
* `Média`: representa o valor central dos valores de cada atibuto, muito útil para resumir grandes quantidades de dados num único valor

* `Mediana`: útil para entender a têndencia central dos dados, especialmente quando há valores muito extremos fora de padrão `(outliers)`, pois não é afetada por eles como a média

* `Desvio Padrão`: Indica o grau de dispersão dos dados. Um desvio padrão alto significa que os dados estão espalhados em uma ampla gama de valores, enquanto um desvio padrão baixo indica que os valores estão próximos da média

* `Assimetria`: Ajuda a entender a distribuição dos dados. Uma distribuição assimétrica pode indicar a presença de outliers ou a necessidade de uma transformação dos dados.

* `Curtose`: Informa sobre a forma da distribuição dos dados, ajudando a identificar a presença de picos acentuados ou distribuições mais uniformes. Isso pode ser útil para análises estatísticas mais aprofundadas e para a modelagem de dados.
<br><br>    
            
## 1.2 Missing Values 

Os missing values são valores de um dado atributo dos quais se desconhece o seu valor real. Ao trabalhar com dados de larga escala, é perfeitamente comum que alguns valores sejam desconhecidos e, por isso, abordagem a este problema é fulcral para o bom funcionamento do algoritmo de Machine Lerning. Ao longo deste capítulo queremos que entenda a nossa linha de pensamento e a forma como abordamos esta questão. 

Resumidamente, o primeiro passo e mais simplório é a `identificação visual dos missing values`. Aqui dispõe do DataFrame com os `missing values` devidamente assinalados e também de um gráfico de barras que denota a quantidade de variáveis por atributo:           
            
            
            
           '''  ,unsafe_allow_html=True)







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
<br><br><br>
            
### 1.2.1 Tratamento de Missing Values (HEOM)
<br>
            
Agora com os missing values identificados, podemos debruçar a nossa atenção sobre em como inputar os possíveis valores de cada missing value de uma feature.

Inicialmente, e como já deve ter passado pela cabeça de qualquer um, julgamos que a substituição dos missing values seria adequada pela `média` ou `mediana`. Embora este método não implicasse porblemas futuros no algoritmo de Machine Learning, este método tem um grande prejuízo: a perca de variabilidade dos dados. Nada garante que o verdadeiro valor do missing value inputado semelhante à média ou à mediana, e por isso, o sistema de classificação final ficaria como um cavalo com palas: restrito a um "campo de visão" muito curto.

Por isso, acreditamos que a melhor forma de imputação de missing values fosse por `Heterogeneous Euclidean-Overlap Metric`, ou `HEOM`.Passamos a explicar:
<br>

	O Manel tem seu valor da Hemoglobina em falta;
	A forma como o Manel manifestou a doença é bastante semelhante à forma da Joana, do João e do Pedro.
	Então será de se esperar que o valor da Hemoglobina do Manel seja (no mínimo) semelhante à média dos valores da Joana, do João e do Pedro.

Ou seja, em traços gerais, podemos imputar um missing value de um atributo de um paciente se calcularmos a distância entre pacientes por HEOM, sinalizarmos os "x" --meter em bonito escrito a mao tipo varivel-- pacientes mais próximos, fizermos a média dos valores do determinado atributo dos pacientes e atribuirmos o valor da média ao missing value. Protemos que soa mais complicado do que realmente é.

Este método calcula a distância entre dois pacientes diferentes pela sua semelhança entre cada atributo de ambos. Vejamos como funciona esta métrica.  
>Neste primeiro passo inicia-se uma tripla condição:
>* Se x ou y são desconhecidos, a distância é 1
>* Se a coluna ``a`` é categórica, faz-se um cálculo simples de overlap das variáveis .
>* Por fim se a coluna ``a`` é numérica, calcula-se a diferença relativa entre x e y na função rn_diff.
            
            ''', unsafe_allow_html=True)   

st.latex(r'''
d_a(x, y) = 
\begin{cases} 
1 & \text{se } x \text{ ou } y \text{ é desconhecido; else} \\
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
            
>Finalmente, aplica-se fórmula final que faz o calculo Euclideano em todas as distancias que conclui assim a fórmula HEOM. 

''', unsafe_allow_html=True)
st.latex(r'''\text{HEOM}(x, y) = \sqrt{\sum_{a=1}^{m} d_a(x_a, y_a)^2}
''')
st.markdown('''
<br>
            
Este Cálculo é muito demorado devido a todas as suas operações. Só neste Dataset existem 50 variáveis que têm de ser comparadas entre cada par de pacientes.
<br>

Todos estes loops contribuem para um aumento da *Time complexity* que acaba por resultar  <span style="color: red; font-weight: bold;">${O_n(n^2*m)}$</span> no qual n é o nr de pacientes e m o número de variáveis . <br>
Agora, com a distancia entre cada paciente calculadado, podemos formar uma matriz que em cada célula contém o valor da distância entre dois pacientes. (Nota que a distancia entre os pacientes X , Y é a mesma que a distancia entre os pacientes Y , X - por isso não deve ser duplamente calculada)
Abaixo encontra-se tal matriz: <br><br>''', unsafe_allow_html=True)
Heom=Dataset.builderData("Tabela_HEOM.csv", "?")
st.dataframe(Heom.df, height=840, use_container_width=False)  # Tabela da media
st.markdown('''Eis o respetivo código:''',unsafe_allow_html=True)
st.code('''
    def tabelaHEOM(self)->pd.DataFrame:
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
    

    def HEOM(self, paciente_1:int, paciente_2:int)->int: #Heterogeneous Euclidean-Overlap Metric
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
st.markdown('''<br><br>
### 1.2.2 Substituição de Missing Values
<br>
Texto explicativo sobre a substituição de missing values
<br>
            ''' ,unsafe_allow_html=True)

Tabela_preenchida=Dataset.builderData("Tabela_sem_missing_values_3.csv", "?")
st.dataframe(Tabela_preenchida.df, height=840, use_container_width=False)  # Tabela da media

st.markdown('''
<br><br>
Eis o respetivo código:''',unsafe_allow_html=True)

st.code('''
        
    def fill_missing_values(self, nr_vizinhos:int) -> pd.DataFrame:

        self.df = self.replace_nan_with_none() # Trocar missing values para None 

        self.df = self.df.drop(['Iron', 'Sat', 'Ferritin'], axis=1)# Eliminar colunas desnecessárias
        df_copiada = self.df.copy()# Criar uma copia do dataframe

        for i in range(len(self.df)): #iterar por todas as linhas
            row = self.df.iloc[i]
            
            if row.isnull().any():# Ver se essa linha tem missing values
                closest_rows = self.linhas_mais_proximas(nr_vizinhos, i)# Calcular as linhas mais proximas
    
                for col in self.df.columns:# Iterar por todas as colunas
                    if pd.isnull(row[col]): # Se a célula for um missing value substituir na tabela o valor
                        df_copiada.loc[i, col] = self.subs_na_tabela(closest_rows, col,nr_vizinhos,i)
        return df_copiada
    
    def linhas_mais_proximas(self, vizinhos:int,i:int)->list: # Calcular as linhas mais proximas por cálculo HEOM e retornar as linhas mais próximas
       
        heom_values = []

        for j in range(len(self.df)):

            if j != i:
                heom_distance = self.HEOM(i, j)# Calcular a distância HEOM entre as linhas
                if len(heom_values) < vizinhos: # Se o número de valores na heap for menor que o número de vizinhos, adicionamos o valor

                    heapq.heappush(heom_values, (-heom_distance, j))
                else:
                    if -heom_distance > heom_values[0][0]:#Se a distância for maior que o valor mais pequeno na heap, substituímos o valor
                        heapq.heapreplace(heom_values, (-heom_distance, j))
    
        # Selecionar as linhas mais próximas
        closest_rows = [item[1] for item in heom_values]

        return closest_rows    

    def subs_na_tabela(self, closest_rows:list, col:int,vizinhos,i)->float | str :

        column_values = []

        for row_index in closest_rows:
            try:#Ver se a tabela é numérica ou categórica

                value = float(self.df.loc[row_index, col])
            except:
                value = self.df.loc[row_index, col]

            if value is not None and not pd.isna(value):#Se o valor não for None, adicionamos à lista
                column_values.append(value)
        if len(column_values) == 0:

            return self.subs_na_tabela(self.linhas_mais_proximas(vizinhos+1,i), col,vizinhos+1,i)#Se não houver valores na lista, aumentamos o número de vizinhos e calcula se outra vez até encontrar 
        
        if isinstance(column_values[0], str):

            # Se os valores forem strings retornar a moda
            return max(set(column_values), key=column_values.count)
        elif isinstance(column_values[0], (int, float)):

            # Se o valor for numérico retornar a média
            return np.mean(column_values)''' , language="python")


st.markdown('''<br><br> 
Talvez um texto aqui''' ,unsafe_allow_html=True)
col1, col2,col3,col4 = st.columns(spec=[0.2,0.2,0.2,0.4])
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



st.markdown('''
            
 ******

# 1.2 Identificação de outliers
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








st.header("Tabela com missing values substituidos por tratamento de outiliers")
data = Dataset.builderData("Tabela_OT_antes_MV.csv", "?")
st.dataframe(data.df, height=840, use_container_width=False)  # Tabela da media
st.markdown('''talvez um texto aqui''',unsafe_allow_html=True)  

st.markdown('''
<br><br>
******
# 2. Algoritmos de Supervised Machine Learning
<br>
texto explicativo sobre os algoritmos de machine learning 
<br>
Quando confrontados com os graficos resultantes de Hyperparameters dos vários algoritmos, surgiu a questão: como devemos escolher?
Assim, para além de ter em conta aspetos como o `test_size` e `n_neighbors` (exemplos de parâmetros do KNN), procuramos implementar outras métricas, nomeadamente **Sensitivity** (ou **Recall**) e **Specificity** cujas formas se apresentam abaixo: <br>
<br>
''',unsafe_allow_html=True)

st.latex(r'''{\text{Specificity} = \frac{\text{True Negatives}}{\text{True Negatives} + \text{False Positives}}}''')

st.markdown('''<br>''',unsafe_allow_html=True) 

st.latex(r'''{\text{Sensitivity} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}}''')

st.markdown('''
<br>

Tal significa, portanto, que a Sensitivity representa, em termos leigos, "dos resultados positivos (`1` ou `"Lives"`), quantos foram analisados corretamente" e a Specificity "dos resultados negativos (`0` ou `"Dies"`), quantos foram analisados corretamente". <br>

Ora, como sendo este um dataset representativo de um diagnóstico médico, é largamente preferível obter um falso positivo em vez de um falso negativo. Esse facto levou-nos a priorizar pontos que maximizem o Recall, garantindo que, um paciente que é, de facto, positivo, seja, categorizado como tal. <br>

Logo, no sentido de definir os melhores parâmetros criamos o gráfico de Accuracy em função de Recall, e escolhemos o ponto mais próximo ao caso ideal (ou seja, o ponto `(1,1)` ). Atente no seguinte mencionado: <br>


## 2.1 Algoritmo KNN 
<br>
texto explicativo sobre o algoritmo KNN e utilização de PCA
<br>''',unsafe_allow_html=True)            

tab1, tab2 = st.tabs(['Tabela Missing values sem outliers', 'Tabela Missing values com outliers'])
with tab1:
    pca= Dataset.builderData("Graficos\Grafico_PCA_MV.csv", "?")
    alt_c = alt.Chart(pca.df).mark_circle().encode(
    alt.X('PC1:Q', scale=alt.Scale(domain=[-5.5, 7.5])),
    alt.Y('PC2:Q',scale=alt.Scale(domain=[-6, 10])),
    color=alt.Color('Class')
).interactive().properties(height=800)
    st.altair_chart(alt_c, use_container_width=True,theme=None)
with tab2:
    pca= Dataset.builderData("Graficos\Grafico_PCA_OT_MV.csv", "?")
    alt_c = alt.Chart(pca.df).mark_circle().encode(
    alt.X('PC1:Q', scale=alt.Scale(domain=[-5.5, 7.5])),
    alt.Y('PC2:Q',scale=alt.Scale(domain=[-6, 10])),
    color=alt.Color('Class')
    ).interactive().properties(height=800)
    st.altair_chart(alt_c, use_container_width=True,theme=None)
st.markdown('''<br>
Conbluso sobre o PCA ''',unsafe_allow_html=True)            

st.markdown('''
<br><br>
            
###2.1.1 Aplicaçãoo do KNN 
<br>
texto explicativo sobre a aplicação do KNN e cross validation''' ,unsafe_allow_html=True)

tab1,tab2,tab3,tab4 = st.tabs(['KNN hyperparameters com apenas tratamento de Missing Values','KNN hyperparameters com tratamento de Missing Values e Outliers','KNN cross validation com tratamento de Missing Values','KNN  cross validation com tratamento de Missing Values e Outliers'])

with tab1:
    col1, col2,col3 = st.columns(spec=[0.85,0.05, 0.1])
    #inicializar o data-set
    with col3:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        optiony1 = st.selectbox(
            "Eixo  y",(
            tuple("Accuracy Precision Recall Specificity".split())), index=2)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        optionx1 = st.selectbox(
            "Eixo  x",
            (tuple("Accuracy Precision Recall Specificity".split())), index=0)
        opção_escolhiday = optiony1+":Q"
        opção_escolhidax = optionx1+":Q"
    with col1:

        grafico1= Dataset.builderData("Graficos\Grafico_KNN_HP_MV.csv", "?")
        grafico_RG = alt.Chart(grafico1.df).mark_circle().encode(
        x=alt.X(opção_escolhidax, scale=alt.Scale(domain=[0.36,0.86])),
        y=alt.Y(opção_escolhiday, scale=alt.Scale(domain=[0.15,1.3])),
        color=alt.Color('Neighbors:N'),
        size='Test Size:Q',
        tooltip=['Accuracy','Precision','Test Size','Random State','Neighbors','Weight','Metric','Recall','Specificity']
        ).interactive().properties(height=800)
        st.altair_chart(grafico_RG, use_container_width=True)

with tab2:
    col1, col2,col3 = st.columns(spec=[0.85,0.05, 0.1])
    #inicializar o data-set
    with col3:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        optiony2 = st.selectbox(
            "Eixo y",(
            tuple("Accuracy Precision Recall Specificity".split())), index=2)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        optionx2 = st.selectbox(
            "Eixo x",
            (tuple("Accuracy Precision Recall Specificity".split())), index=0)
        opção_escolhiday = optiony2+":Q"
        opção_escolhidax = optionx2+":Q"
    with col1:
        grafico2= Dataset.builderData("Graficos\Grafico_KNN_HP_OT_MV.csv", "?")
        grafico_RG = alt.Chart(grafico2.df).mark_circle().encode(
        x=alt.X(opção_escolhidax, scale=alt.Scale(domain=[0.36,0.86])),
        y=alt.Y(opção_escolhiday, scale=alt.Scale(domain=[0.15,1.3])),
        color=alt.Color('Neighbors:N'),
        size='Test Size:Q',
        tooltip=['Accuracy','Precision','Test Size','Random State','Neighbors','Weight','Metric','Recall','Specificity']
        ).interactive().properties(height=800)
        st.altair_chart(grafico_RG, use_container_width=True)

with tab3:
    col1, col2,col3 = st.columns(spec=[0.85,0.05, 0.1])
    #inicializar o data-set
    with col3:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        optiony3 = st.selectbox(
            "Eixo   y",(
            tuple("Accuracy Precision Recall".split())), index=2)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        optionx3 = st.selectbox(
            "Eixo   x",
            (tuple("Accuracy Precision Recall".split())), index=0)
        opção_escolhiday = optiony3+":Q"
        opção_escolhidax = optionx3+":Q"
    with col1:
        grafico3= Dataset.builderData("Graficos\Grafico_KNN_CV_MV.csv", "?")    
        grafico_KNN = alt.Chart(grafico3.df).mark_circle().encode(
        x=alt.X(opção_escolhidax, scale=alt.Scale(domain=[0.585,0.770])),
        y=alt.Y(opção_escolhiday, scale=alt.Scale(domain=[0.7,1.08])),
        color=alt.Color('N_Neighbors:N'),
        size='Test Size:Q',
        tooltip=['Accuracy','Precision','Test Size','Random State','N_Neighbors','Recall']
        ).interactive().properties(height=800)
        st.altair_chart(grafico_KNN, use_container_width=True)

with tab4:
    col1, col2,col3 = st.columns(spec=[0.85,0.05, 0.1])
    #inicializar o data-set
    with col3:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        optiony4 = st.selectbox(
            "Eixo    y",(
            tuple("Accuracy Precision Recall".split())), index=2)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        optionx4 = st.selectbox(
            "Eixo    x",
            (tuple("Accuracy Precision Recall ".split())), index=0)
        opção_escolhiday = optiony4+":Q"
        opção_escolhidax = optionx4+":Q"
    with col1:
        grafico4= Dataset.builderData("Graficos\Grafico_KNN_CV_OT_MV.csv", "?")    
        grafico_KNN = alt.Chart(grafico4.df).mark_circle().encode(
        x=alt.X(opção_escolhidax, scale=alt.Scale(domain=[0.585,0.770])),
        y=alt.Y(opção_escolhiday, scale=alt.Scale(domain=[0.7,1.08])),
        color=alt.Color('N_Neighbors:N'),
        size='Test Size:Q',
        tooltip=['Accuracy','Precision','Test Size','Random State','N_Neighbors','Recall']
        ).interactive().properties(height=800)

        st.altair_chart(grafico_KNN, use_container_width=True)


data = Dataset.builderData("Tabela_sem_missing_values_3.csv", "?")
data1= data.categorical_to_numerical()
X = data1.drop(columns=['Class']).dropna()
y = data1['Class']

st.markdown('''<br><br>
Como funcionam os vizinhos mais proximos''' ,unsafe_allow_html=True)
st.altair_chart( Grafico_vizinhos(X,y), use_container_width=True)


test_sizes = [0.1,0.15,0.20,0.25, 0.3,0.35, 0.4]  # List of different test sizes
random_states = [42, 123, 456]  # List of different random states
k_neighbors = range(1, 99)  # Range of k neighbors





tab5,tab6,tab7,tab8 = st.tabs(['Decision Tree hyperparameters com apenas tratamento de Missing Values','Decision Tree hyperparameters com tratamento de Missing Values e Outliers','Decision Tree cross validation com tratamento de Missing Values','Decision Tree  cross validation com tratamento de Missing Values e Outliers'])
with tab5:
    col1, col2,col3 = st.columns(spec=[0.85,0.05, 0.1])
    #inicializar o data-set
    with col3:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        optiony1 = st.selectbox(
            "Eixo     y",(
            tuple("Accuracy Precision Recall Specificity".split())), index=2)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        optionx1 = st.selectbox(
            "Eixo     x",
            (tuple("Accuracy Precision Recall Specificity".split())), index=1)
        opção_escolhiday = optiony1+":Q"
        opção_escolhidax = optionx1+":Q"
    with col1:
        st.header("Gráfico Decision Tree hyperparameters com trataemnto de Missing Values")
        grafico5= Dataset.builderData("Graficos\Grafico_DC_HP_MV.csv", "?")
        chart = alt.Chart(grafico5.df).mark_circle().encode(
        x=alt.X(opção_escolhidax, scale=alt.Scale(domain=[0.25,1.1])),
        y=alt.Y(opção_escolhiday, scale=alt.Scale(domain=[0.3,1.1])),
        color=alt.Color('Test_Size:N'),
        size='Depth:N',
        tooltip=['Depth', 'Min_Samples_Split', 'Min_Samples_Leaf', 'Test_Size','Accuracy', 'Precision', 'Recall','Specificity','Random_State']
        ).interactive().properties(height=800)
        st.altair_chart(chart, use_container_width=True)

with tab6:
    col1, col2,col3 = st.columns(spec=[0.85,0.05, 0.1])
    #inicializar o data-set
    with col3:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        optiony2 = st.selectbox(
            "Eixo      y",(
            tuple("Accuracy Precision Recall Specificity".split())), index=2)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        optionx2 = st.selectbox(
            "Eixo      x",
            (tuple("Accuracy Precision Recall Specificity".split())), index=1)
        opção_escolhiday = optiony2+":Q"
        opção_escolhidax = optionx2+":Q"
    with col1:
        st.header("Gráfico Decision Tree hyperparameters com tratamento de Missing Values e Outliers")
        grafico5= Dataset.builderData("Graficos\Grafico_DC_HP_OT_MV.csv", "?")
        chart = alt.Chart(grafico5.df).mark_circle().encode(
        x=alt.X(opção_escolhidax, scale=alt.Scale(domain=[0.25,1.1])),
        y=alt.Y(opção_escolhiday, scale=alt.Scale(domain=[0.3,1.1])),
        color=alt.Color('Test_Size:N'),
        size='Depth:N',
        tooltip=['Depth', 'Min_Samples_Split', 'Min_Samples_Leaf', 'Test_Size','Accuracy', 'Precision', 'Recall','Specificity','Random_State']
        ).interactive().properties(height=800)
        st.altair_chart(chart, use_container_width=True)

with tab7:
    col1, col2,col3 = st.columns(spec=[0.85,0.05, 0.1])
    #inicializar o data-set
    with col3:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        optiony2 = st.selectbox(
            "Eixo       y",(
            tuple("Accuracy Precision Recall ".split())), index=2)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        optionx2 = st.selectbox(
            "Eixo       x",
            (tuple("Accuracy Precision Recall ".split())), index=1)
        opção_escolhiday = optiony2+":Q"
        opção_escolhidax = optionx2+":Q"
    with col1:
        st.header("Gráfico Decision Tree hyperparameters")
        grafico6= Dataset.builderData("Graficos\Grafico_DC_CV_MV.csv", "?")
        grafico_DC_CV = alt.Chart(grafico6.df).mark_circle().encode(
        x=alt.X(opção_escolhidax, scale=alt.Scale(domain=[0.64,0.76])),
        y=alt.Y(opção_escolhiday, scale=alt.Scale(domain=[0.61,0.76])),
        color=alt.Color('Depth:N'),
        size=alt.value(600),
        tooltip=['Depth', 'Min_Samples_Split', 'Min_Samples_Leaf', 'Accuracy', 'Precision', 'Recall']).interactive().properties(height=800)
        st.altair_chart(grafico_DC_CV, use_container_width=True)

with tab8:
    col1, col2,col3 = st.columns(spec=[0.85,0.05, 0.1])
    #inicializar o data-set
    with col3:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        optiony2 = st.selectbox(
            "Eixo          y",(
            tuple("Accuracy Precision Recall ".split())), index=2)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        optionx2 = st.selectbox(
            "Eixo         x",
            (tuple("Accuracy Precision Recall ".split())), index=1)
        opção_escolhiday = optiony2+":Q"
        opção_escolhidax = optionx2+":Q"
    with col1:
        st.header("Gráfico Decision Tree hyperparameters")
        grafico7= Dataset.builderData("Graficos\Grafico_DC_CV_OT_MV.csv", "?")
        grafico_DC_CV = alt.Chart(grafico7.df).mark_circle().encode(
        x=alt.X(opção_escolhidax, scale=alt.Scale(domain=[0.64,0.74])),
        y=alt.Y(opção_escolhiday, scale=alt.Scale(domain=[0.61,0.76])),
        color=alt.Color('Depth:N'),
        size=alt.value(600),
        tooltip=['Depth', 'Min_Samples_Split', 'Min_Samples_Leaf', 'Accuracy', 'Precision', 'Recall']).interactive().properties(height=800)
        st.altair_chart(grafico_DC_CV, use_container_width=True)


st.markdown('''
<br><br>
### 2.2 Algoritmo Decision Tree
talvez um texto aqui 

eis as nossa Decision trees ''' ,unsafe_allow_html=True)


tab9,tab10 = st.tabs(['Decision Tree com apenas tratamento de Missing Values','Decision Tree com tratamento de Missing Values e Outliers'])

with tab9:
    st.image('best_DC_HP_OT_MD.png', caption='Decision Tree com tratamento de Missing Values')
with tab10:
    st.image('DC_OT_MV.png', caption='Decision Tree com tratamento de Missing Values e Outliers')

st.markdown('''
## 2.3 Logistic regression 
Texto aqui 
            
            
            
            
            
            
            
            ''',unsafe_allow_html=True)


tab11,tab12,tab13,tab14 = st.tabs(['Logistic Regression hyperparameters com apenas tratamento de Missing Values','Logistic Regression hyperparameters com tratamento de Missing Values e Outliers','Logistic Regression cross validation com tratamento de Missing Values','Logistic Regression  cross validation com tratamento de Missing Values e Outliers'])
with tab11:
    col1, col2,col3 = st.columns(spec=[0.85,0.05, 0.1])
    #inicializar o data-set
    with col3:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        optiony1 = st.selectbox(
            "Eixo       y",(
            tuple("Accuracy Precision Recall Specificity".split())), index=2)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        optionx1 = st.selectbox(
            "Eixo       x",
            (tuple("Accuracy Precision Recall Specificity".split())), index=1)
        opção_escolhiday = optiony1+":Q"
        opção_escolhidax = optionx1+":Q"
    with col1:
        grafico6= Dataset.builderData("Graficos\Grafico_LR_HP_MV.csv", "?")
        grafico_RG = alt.Chart(grafico6.df).mark_circle().encode(
        x=alt.X(opção_escolhidax, scale=alt.Scale(domain=[0.36,0.92])),
        y=alt.Y(opção_escolhiday, scale=alt.Scale(domain=[0.3,1.1])),
        color=alt.Color('C:N'),
        size='Test Size:Q',
        tooltip=['Accuracy','Precision','Test Size','Random State','C','Penalty','Solver','Class Weight','Recall','Specificity']
        ).interactive().properties(height=800).interactive()
        st.altair_chart(grafico_RG, use_container_width=True)

with tab12:
    col1, col2,col3 = st.columns(spec=[0.85,0.05, 0.1])
    #inicializar o data-set
    with col3:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        optiony2 = st.selectbox(
            "Eixo         y",(
            tuple("Accuracy Precision Recall Specificity".split())), index=2)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        optionx2 = st.selectbox(
            "Eixo         x",
            (tuple("Accuracy Precision Recall Specificity".split())), index=1)
        opção_escolhiday = optiony2+":Q"
        opção_escolhidax = optionx2+":Q"
    with col1:
        grafico7= Dataset.builderData("Graficos\Grafico_LR_HP_OT_MV.csv", "?")
        grafico_RG = alt.Chart(grafico7.df).mark_circle().encode(
        x=alt.X(opção_escolhidax, scale=alt.Scale(domain=[0.36,0.92])),
        y=alt.Y(opção_escolhiday, scale=alt.Scale(domain=[0.3,1.1])),
        color=alt.Color('C:Q'),
        size='Test Size:Q',
        tooltip=['Accuracy','Precision','Test Size','Random State','C','Penalty','Solver','Class Weight','Recall','Specificity']
        ).interactive().properties(height=800).interactive()
        st.altair_chart(grafico_RG, use_container_width=True)

with tab13:
    col1, col2,col3 = st.columns(spec=[0.85,0.05, 0.1])
    #inicializar o data-set
    with col3:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        optiony2 = st.selectbox(
            "Eixo           y",(
            tuple("Accuracy Precision Recall ".split())), index=2)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        optionx2 = st.selectbox(
            "Eixo           x",
            (tuple("Accuracy Precision Recall ".split())), index=1)
        opção_escolhiday = optiony2+":Q"
        opção_escolhidax = optionx2+":Q"
    with col1:
        grafico7= Dataset.builderData("Graficos\Grafico_LR_CV_MV.csv", "?")
        grafico_RG = alt.Chart(grafico7.df).mark_circle().encode(
        x=alt.X(opção_escolhidax, scale=alt.Scale(domain=[0.65,0.89])),
        y=alt.Y(opção_escolhiday, scale=alt.Scale(domain=[0.52,0.88])),
        color=alt.Color('C:N'),
        size='Test Size:Q',
        tooltip=['Accuracy','Precision','Test Size','Random State','C','Penalty','Solver','Class Weight','Recall']
        ).interactive().properties(height=800).interactive()
        st.altair_chart(grafico_RG, use_container_width=True)

with tab14:
    col1, col2,col3 = st.columns(spec=[0.85,0.05, 0.1])
    #inicializar o data-set
    with col3:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        optiony2 = st.selectbox(
            "Eixo               y",(
            tuple("Accuracy Precision Recall ".split())), index=2)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        optionx2 = st.selectbox(
            "Eixo              x",
            (tuple("Accuracy Precision Recall ".split())), index=1)
        opção_escolhiday = optiony2+":Q"
        opção_escolhidax = optionx2+":Q"
    with col1:
        grafico8= Dataset.builderData("Graficos\Grafico_LR_CV_OT_MV.csv", "?")
        grafico_RG = alt.Chart(grafico8.df).mark_circle().encode(
        x=alt.X(opção_escolhidax, scale=alt.Scale(domain=[0.65,0.89])),
        y=alt.Y(opção_escolhiday, scale=alt.Scale(domain=[0.52,0.88])),
        color=alt.Color('C:N'),
        size='Test Size:Q',
        tooltip=['Accuracy','Precision','Test Size','Random State','C','Penalty','Solver','Class Weight','Recall']
        ).interactive().properties(height=800).interactive()
        st.altair_chart(grafico_RG, use_container_width=True)

st.markdown('''
conclusoed sobre o logistic regression
*******
            



<br><br>
## 2.5 Random Forest
Texto aqui       
            ''',unsafe_allow_html=True)

tab15,tab16,tab17,tab18 = st.tabs(['Random Forest hyperparameters com apenas tratamento de Missing Values','Random Forest hyperparameters com tratamento de Missing Values e Outliers','Random Forest cross validation com tratamento de Missing Values','Random Forest  cross validation com tratamento de Missing Values e Outliers'])
with tab15:
    col1, col2,col3 = st.columns(spec=[0.85,0.05, 0.1])
    #inicializar o data-set
    with col3:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        optiony1 = st.selectbox(
            "Eixo                  y",(
            tuple("Accuracy Precision Recall Specificity".split())), index=2)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        optionx1 = st.selectbox(
            "Eixo                  x",
            (tuple("Accuracy Precision Recall Specificity".split())), index=1)
        opção_escolhiday = optiony1+":Q"
        opção_escolhidax = optionx1+":Q"
    with col1:
        grafico14= Dataset.builderData("Graficos\Grafico_RF_HP_MV.csv", "?")
        grafico_RG = alt.Chart(grafico14.df).mark_circle().encode(
        x=alt.X(opção_escolhidax, scale=alt.Scale(domain=[0.4,0.94])),
        y=alt.Y(opção_escolhiday, scale=alt.Scale(domain=[0.5,1.1])),
        color=alt.Color('N_Estimators:N'),
        size='Test Size:Q',
        tooltip=['Accuracy','Precision','Test Size','Random State','N_Estimators','Class Weight','Recall','Specificity']
        ).interactive().properties(height=800)
        st.altair_chart(grafico_RG, use_container_width=True)

with tab16:
    col1, col2,col3 = st.columns(spec=[0.85,0.05, 0.1])
    #inicializar o data-set
    with col3:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        optiony2 = st.selectbox(
            "Eixo                             y",(
            tuple("Accuracy Precision Recall Specificity".split())), index=2)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        optionx2 = st.selectbox(
            "Eixo                              x",
            (tuple("Accuracy Precision Recall Specificity".split())), index=1)
        opção_escolhiday = optiony2+":Q"
        opção_escolhidax = optionx2+":Q"
    with col1:
        grafico9= Dataset.builderData("Graficos\Grafico_RF_HP_OT_MV.csv", "?")
        grafico_RG = alt.Chart(grafico9.df).mark_circle().encode(
        x=alt.X(opção_escolhidax, scale=alt.Scale(domain=[0.4,0.94])),
        y=alt.Y(opção_escolhiday, scale=alt.Scale(domain=[0.5,1.1])),
        color=alt.Color('N_Estimators:N'),
        size='Test Size:Q',
        tooltip=['Accuracy','Precision','Test Size','Random State','N_Estimators','Class Weight','Recall','Specificity']
        ).interactive().properties(height=800)  
        st.altair_chart(grafico_RG, use_container_width=True)

with tab17:
    col1, col2,col3 = st.columns(spec=[0.85,0.05, 0.1])
    #inicializar o data-set
    with col3:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        optiony2 = st.selectbox(
            "Eixo                                      y",(
            tuple("Accuracy Precision Recall ".split())), index=2)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        optionx2 = st.selectbox(
            "Eixo                                      x",
            (tuple("Accuracy Precision Recall ".split())), index=1)
        opção_escolhiday = optiony2+":Q"
        opção_escolhidax = optionx2+":Q"
    with col1:
        grafico11= Dataset.builderData("Graficos\Grafico_RF_CV_MV.csv", "?")
        grafico_RG = alt.Chart(grafico11.df).mark_circle().encode(
        x=alt.X(opção_escolhidax, scale=alt.Scale(domain=[0.65,0.84])),
        y=alt.Y(opção_escolhiday, scale=alt.Scale(domain=[0.6,0.98])),
        color=alt.Color('N_Estimators:N'),
        size='Test Size:Q',
        tooltip=['Accuracy','Precision','Test Size','Random State','N_Estimators','Class Weight','Recall']
        ).interactive().properties(height=800)
        st.altair_chart(grafico_RG, use_container_width=True)

with tab18:
    col1, col2,col3 = st.columns(spec=[0.85,0.05, 0.1])
    #inicializar o data-set
    with col3:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        optiony2 = st.selectbox(
            "Eixo                                             y",(
            tuple("Accuracy Precision Recall ".split())), index=2)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        optionx2 = st.selectbox(
            "Eixo                                              x",
            (tuple("Accuracy Precision Recall ".split())), index=1)
        opção_escolhiday = optiony2+":Q"
        opção_escolhidax = optionx2+":Q"
    with col1:
        grafico12= Dataset.builderData("Graficos\Grafico_RF_CV_OT_MV.csv", "?")
        grafico_RG = alt.Chart(grafico12.df).mark_circle().encode(
        x=alt.X(opção_escolhidax, scale=alt.Scale(domain=[0.65,0.84])),
        y=alt.Y(opção_escolhiday, scale=alt.Scale(domain=[0.6,0.98])),
        color=alt.Color('N_Estimators:N'),
        size='Test Size:Q',
        tooltip=['Accuracy','Precision','Test Size','Random State','N_Estimators','Class Weight','Recall']
        ).interactive().properties(height=800)
        st.altair_chart(grafico_RG, use_container_width=True)