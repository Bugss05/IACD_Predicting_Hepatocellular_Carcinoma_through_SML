import pandas as pd
import numpy as np
import streamlit as st

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

        return styled_df
    
    def pintarOutliers(self, df, outliers):
        def highlight_value(series, column):#Pintar as células que são outliers de azul
            return ['background-color: blue' if (column, index) in outliers else '' for index in series.index]
        return df.style.apply(lambda x: highlight_value(x, x.name), axis=0)#Aplicar a função a cada coluna

    def HEOM(self, paciente_1, paciente_2): #Heterogeneous Euclidean-Overlap Metric
        dataframe = self.replace_nan_with_none()
        if paciente_1 == paciente_2:
            return 0
        soma = 0
        for feature in self.df.columns:
            distancia = self.distanciaGeral(dataframe,feature, paciente_1, paciente_2)
            soma += distancia**2
        soma= soma**(1/2)
        return round(soma,2)
    
    def distanciaGeral(self,dataframe, feature:str, paciente_1:int, paciente_2:int)->int:
        try :        
            #display("3")
            #distancia normalizada
            valorPaciente_1 = float(dataframe.loc[paciente_1, feature])
            valorPaciente_2 = float(dataframe.loc[paciente_2, feature])
            numeric_feature = pd.to_numeric(dataframe.copy()[feature], errors='coerce')
            return abs(valorPaciente_1 - valorPaciente_2) / (numeric_feature.max() - numeric_feature.min())
        except :
            valorPaciente_1 = dataframe.loc[paciente_1, feature]
            valorPaciente_2 = dataframe.loc[paciente_2, feature]
            if pd.isna(valorPaciente_1) or pd.isna(valorPaciente_2):
                #display("1")
                return 1
            else:
            #display("2")
            #overlap
            
                if valorPaciente_1 == valorPaciente_2:
                    return 0
                else: 
                    return 1


    def tabelaHEOM(self):
        tabela = pd.DataFrame()
        for i in range(len(self.df)):
            lista = []
            for j in range(len(self.df)):
                if i >= j:
                    lista.append("X")
                else:
                    lista.append(self.HEOM(i, j))
            tabela = pd.concat([tabela, pd.DataFrame({i: lista})], axis=1)
        tabela.to_csv("Tabela_HEOM.csv", index=False)
        return tabela

    '''def tratamentoMissingValues(self, paciente_1):
        paciente_distancia = {}
        #criar dicionario com todas as distancias dos pacientes
        for i in range(165):
            if i != paciente_1:
                paciente_distancia[i] = data.HEOM(paciente_1, i)
        
        #ordenar o dicionario pelas distancias de forma crescente
        sorted_paciente_distancia = sorted(paciente_distancia.items(), key=lambda x: x[1])

        #escolher os 5 melhores pacientes
        melhores_pacientes = [item[0] for item in sorted_paciente_distancia[:3]]
         
        #escolher feature a trocar no paciente original pela media das n features dos n pacientes
        for feature in self.df.columns:
            valorPaciente_1 = self.df.loc[paciente_1, feature]
            if pd.isna(valorPaciente_1):
                media = 0
                contador = 0
                for melhoresPacientes in range(5):
                    #se feature nao numerica, moda do que mais aparece
                    #....

                    #se feature numerica (fzr a condição)
                    if is not pd.isna(self.df.loc[melhoresPacientes, feature]):
                        valorPaciente_X = self.df.loc[melhoresPacientes, feature]
                        media += valorPaciente_X
                        contador += 1 
                media /= contador
                self.df.replace(valorPaciente_1, media, inplace=True)'''

            
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

        
#_______________________________________________________________

data = Dataset.builderData("hcc_dataset.csv", "?")
data.pintarMissingValues()

#_______________________________________________________________

data = Dataset.builderData("hcc_dataset.csv", "?")
data.outliers()