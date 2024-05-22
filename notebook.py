import pandas as pd
import numpy as np
import heapq
class Dataset:
    def __init__(self, df, missing_values):
        self.df = df
        self.missing_values = missing_values


    def pintarMissingValues(self):#pintar a tabela de missing values
        if self.missing_values is not None:#se existirem missing values
            self.df.replace(self.missing_values, "NaN", inplace=True)#substituir missing values por string "NaN" devido a limitação do site 
            return self.df.style.applymap(lambda valor: "color: red;" if valor=="NaN" else "")#pintar missing values a vermelho
        else: return self.df #se não existirem missing values


    def missing_values_percentagem(self):#Percentagem de missing values
        self.df.replace(self.missing_values, np.nan, inplace=True)#substituir missing values por NaN e nao string "NaN"
        missing_values_percentages = self.df.isnull().mean() * 100#calcular a percentagem de missing values
        return missing_values_percentages.tolist()#retornar a percentagem de missing values
    
    
    def remove_int_columns(self):
        df_copy = self.df.copy()  # create a copy of the dataframe
        numerical=self.df_num()
        common_columns = set(self.df.columns) & set(numerical.columns)
        df_copy = df_copy.drop(common_columns, axis=1)
        
        return df_copy
    

    def df_num(self):
        # Replace missing values with None
        dataframe= self.replace_nan_with_none()

        # Convert all columns to numeric, replacing non-numeric values with NaN
        for col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')

        # Remove columns that only contain None values
        self.df = dataframe.dropna(axis=1, how='all')

        return self.df


    def replace_nan_with_none(self):
        self.df.replace(self.missing_values, None, inplace=True)
        return self.df


    def pintarOutliers(self, df, outliers):
        def highlight_value(series, column):#Pintar as células que são outliers de azul
            return ['background-color: blue' if (column, index) in outliers else '' for index in series.index]
        return df.style.apply(lambda x: highlight_value(x, x.name), axis=0)#Aplicar a função a cada coluna
    

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
    

    def outliers(self,info:str,vizinhos:int=None)->pd.DataFrame:
        categorical_features = self.remove_int_columns() #selecionar as colunas categoricas
        numeric_df = self.df_num()  #selecionar as colunas numericas

        colunas_numericas = numeric_df.columns
        if info == 'style':
            outliers = set()
        for coluna in colunas_numericas: #calcular os outliers usando o IQR
            if info == 'tratamento':
                outliers = []
            q1 = numeric_df[coluna].quantile(0.25) #calculo de quartis
            q3 = numeric_df[coluna].quantile(0.75)
            iqr = q3 - q1
            limite_inferior = q1 - 1.5 * iqr #calculo de limites (acima de superior ou abaixo de inferior = outliers)
            limite_superior = q3 + 1.5 * iqr
            for index, value in numeric_df[coluna].items(): #adicionar outliers ao set
                if value < limite_inferior or value > limite_superior: # identificação de outliers
                    if info == 'tratamento' and coluna not in ["Iron", "Sat", "Ferritin"]: #drop de colunas com mais de 35% missing values
                        if self.df.loc[index, coluna] > limite_superior * 1.5 or self.df.loc[index, coluna] < limite_inferior * 1.5:
                            outliers.append((index, coluna))
                    elif info == 'style':
                        outliers.add((coluna, index))
            if info == 'tratamento':
                self.df= self.tratamentoOutliers(outliers, coluna,vizinhos)
        if info == 'style':
            styled_df = self.pintarOutliers(numeric_df, outliers) # Pintar os outliers
            return styled_df
        if info == 'tratamento': 
            self.df = (pd.concat([categorical_features,self.df ], axis=1))
            return self.df
    
    
    def tratamentoOutliers(self, outliers, coluna, vizinhos): #substituir os outliers pela média dos k vizinhos mais proximos
        lista_valores = self.df[coluna].tolist() #todos os valores da coluna 
        contador = -1
        valores_out = [self.df.loc[index,coluna] for index,coluna in outliers] #valores dos outliers
        for valor_outlier in valores_out: #iterar por todos os outliers
            contador+=1
            outlier = valor_outlier
            dicionario_distancias = []
            for valor in lista_valores:
                #o valor so e valido se nao for um far outlier e nao for um missing value
                if outlier != valor and valor not in valores_out and not pd.isna(valor):

                    distancia = self.HEOM(lista_valores.index(valor), lista_valores.index(outlier)) #calcular a distancia entre o outlier e os outros valores
                    if len(dicionario_distancias) < vizinhos: # se o numero de valores na heap for menor que o numero de vizinhos, adicionar o valor
                        heapq.heappush(dicionario_distancias, (-distancia, valor))
                    else:
                        if -distancia > dicionario_distancias[0][0]: #se a distancia for maior que o valor mais pequeno na heap, substituir o valor
                            heapq.heapreplace(dicionario_distancias, (-distancia, valor))

            k_proximos = [abs(item[1]) for item in dicionario_distancias] #selecionar os k vizinhos mais proximos
            
            media = sum(k_proximos)/len(k_proximos)
            self.df.loc[outliers[contador][0], coluna] = media # imputaçao do ajuste no dataframe original
        return self.df

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
            return np.mean(column_values)
        


    

    def categorical_to_numerical(self):
        """
        _summary_: converts all categorical features to numerical values

        _conversion_dictionary_:
            Male -> 0
            Female -> 1
            No -> 0
            Yes -> 1
            Disabled -> 0
            Ambulatory -> 1
            Restricted -> 2
            Selfcare -> 3
            Active -> 4
            None -> 0
            Grade I/II -> 1
            Grade III/IV -> 2
            Mild -> 1
            Moderate/Severe -> 2
            Dies -> 0
            Lives -> 1
            
        """
        words = ("Male","Female","No","Yes","Disabled","Ambulatory",
                 "Restricted","Selfcare","Active","None","Grade I/II",
                 "Grade III/IV","Mild","Moderate/Severe","Dies","Lives")
        values = (0,1,0,1,0,1,2,3,4,0,1,2,1,2,0,1)
        self.df.replace(words, values, inplace=True)
        return self.df


    @classmethod #este classmethod funciona como um construtor alternativo e construir um dataframe a partir de um arquivo cs

    def builderData(cls, df, missing_values): 
        try:
            if not isinstance(df, pd.DataFrame):# Handle DataFrame input directly
                df = pd.read_csv(df)
            df = df.copy()# Avoid modifying the original DataFrame
            return cls(df, missing_values)
        except (FileNotFoundError, pd.errors.ParserError):
            # Handle potential errors: file not found or parsing errors
            print(f"Erro: Não conseguiu ler a data de {df}.")
            raise

        
