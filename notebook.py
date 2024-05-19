import pandas as pd
import numpy as np
import streamlit as st
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
    
    def outliers(self,info):
        # Selecionar apenas as colunas numéricas
        numeric_df = self.df_num()

        colunas_numericas = numeric_df.columns
        if info == 'index':
            outliers = []
        elif info == 'style':
            outliers = set()
        for coluna in colunas_numericas:#calcular os outliers usando o IQR
            q1 = numeric_df[coluna].quantile(0.25)
            q3 = numeric_df[coluna].quantile(0.75)
            iqr = q3 - q1
            limite_inferior = q1 - 1.5 * iqr
            limite_superior = q3 + 1.5 * iqr
            for index, value in numeric_df[coluna].items():#adicionar outliers ao set
                if value < limite_inferior or value > limite_superior:
                    if info == 'index' and coluna not in ["Iron", "Sat", "Ferritin"]:
                        outliers.append((index, coluna))
                    elif info == 'style':
                        outliers.add((coluna, index))
        if info == 'index':
            return self.tratamentoOutliers(outliers, limite_inferior, limite_superior)
        
        elif info == 'style':
            # Apply styling to outliers
            styled_df = self.pintarOutliers(numeric_df, outliers)
            return styled_df

    def fill_missing_values(self, nr_vizinhos:int) -> pd.DataFrame:

        self.df = self.replace_nan_with_none() # Replace missing values with None 
        self.df = self.df.drop(['Iron', 'Sat', 'Ferritin'], axis=1)# Drop unnecessary columns
        df_copiada = self.df.copy()# Create a copy of the DataFrame

        for i in range(len(self.df)): # Iterate over each row
            row = self.df.iloc[i]
            
            if row.isnull().any():# Check if the row has any missing values
                closest_rows = self.linhas_mais_proximas(nr_vizinhos, i)# Get the indices of the closest rows
    
                for col in self.df.columns:# Iterate over each column
                    if pd.isnull(row[col]): # If the value is missing, replace it with the most common value or mean from the closest rows
                        df_copiada.loc[i, col] = self.subs_na_tabela(closest_rows, col,nr_vizinhos,i)
        return df_copiada
    

    def subs_na_tabela(self, closest_rows:list, col:int,vizinhos,i)->float | str :
        # Initialize values
        column_values = []

        for row_index in closest_rows:
            try:
                # Check the type of values
                value = float(self.df.loc[row_index, col])
            except:
                value = self.df.loc[row_index, col]

            if value is not None:
                column_values.append(value)
        if len(column_values) == 0:
            return self.subs_na_tabela(self.linhas_mais_proximas(vizinhos+1,i), col,vizinhos+1,i)
        # Calculate the result based on the type of values
        if isinstance(column_values[0], str):
            # If values are strings, return the most frequent value
            return max(set(column_values), key=column_values.count)
        elif isinstance(column_values[0], (int, float)):
            # If values are numeric, return the mean
            return np.mean(column_values)
        

    def linhas_mais_proximas(self, vizinhos:int,i:int)->list: # Calculate the HEOM distance for each other row
       
        heom_values = []

        for j in range(len(self.df)):

            if j != i:
                heom_distance = self.HEOM(i, j)# Calculate the HEOM distance
                if len(heom_values) < vizinhos: # If we have less than 'vizinhos' distances, we add it to the heap

                    heapq.heappush(heom_values, (-heom_distance, j))
                else:
                    if -heom_distance > heom_values[0][0]: # If the current distance is smaller than the largest distance in the heap, we replace it
                        heapq.heapreplace(heom_values, (-heom_distance, j))
    
        # Get the rows with the smallest HEOM distance
        closest_rows = [item[1] for item in heom_values]

        return closest_rows
    
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

        
