# <font size="65">Predicting Hepatocellular Carcinoma through Supervised Machine learning</font>

Trabalho realizado por:

* Afonso Coelho (FCUP_IACD:202305085)
* Diogo Amaral (FCUP_IACD:202305187) 
* Miguel Carvalho (FCUP_IACD:202305229)

<div style="padding: 10px;padding-left:5%">
<img src="fotos_md/Cienciasporto.png" style="float:left; height:75px;width:200px">
<img src="fotos_md/Feuporto.png" style="float:left ; height:75px; padding-left:20px;width:200px">
</div>

<div style="clear:both;"></div>

******

## Projeto  

O objetivo deste projeto é abordar um caso real do conjunto de dados, representativo de pacientes que sofrem de Carcinoma Hepatocelular (HCC), mais comumente conhecido como cancro do fígado. O conjunto mencionado de dados HCC ([hcc_dataset.csv](hcc_dataset.csv)) foi recolhido no Centro Hospitalar e Universitário de Coimbra (CHUC) em Portugal, e contém dados clínicos reais de pacientes diagnosticados com HCC.<br>

Pretende-se, portanto, desenvolver vários algoritmos SML (Supervised Machine Learning), capazes de classificar os pacientes relativamente à sua sobrevivência após 1 ano do diagnóstico (aspeto identificado na coluna `"Class"` com `"Lives"` ou `"Dies"`). <br>

## Streamlit
Para visitar o Streamlit onde está documentado todo o processo, auxiliado por gráficos relevantes ao projeto, clique [aqui]((https://predictinghepatocellularcarcinomathroughsml.streamlit.app/)).

 ## Como opera o programa

Este programa trabalha com um intrepetador `Python` e usa um ambiente virtual `conda`  de forma a facilitar a instalação das dependências necessárias para a utilização do Jupyter Notebook. O notebook mencionado está guardado no ficheiro de nome [polirTabela.ipybn](polirTabela.ipybn) que, passo a passo, mostrará a progressão do projeto, bem como o nosso processo lógico e a forma como decidimos atacar o problema. <br>

A escolha da utilização de um ambiente `conda` derivou dos seguintes fatores:

* **Bibliotecas extensas**: a implementação de um ambiente virtual automatiza a instalação das bibliotecas, facilitando o acesso e reduzindo o de tempo perdido na instalação das mesmas;
 
* **Isolamento de dependências**: ao criar um ambiente separado, evitam-se conflitos entre bibliotecas de outros projetos e garante-se a compatibilidade;

* **Organização**: sendo este ambiente naturalmente mais reduzido em relaçáo ao ambiente nativo da máquina, a sua utilização mantém a pasta do Python organizada e facilita a identificação de bibliotecas;

* **Reprodutibilidade**: a criação de um ficheiro [requirements.txt](requirements.txt) facilita a partilha e a execução do código em diferentes máquinas, tornando o programa compatível em qualquer máquina;

* **Gestão de versões**: pela simplicidade da ferramenta `conda`, torna-se fácil instalar e manter diferentes versões de bibliotecas para cada projeto, sem nunca correr o risco de causar conflitos de dependências;

* **Leveza**: a transmissão e instalação do ambiente é facilitada com pelo ficheiro [requirements.txt](requirements.txt) sendo portanto apenas necessários menos que `30 KB` de espaço livre em disco para obter a lista detalhada com todas as bibliotecas utilizadas.



## Instalar o programa

#### Pré-Requisitos
* Conda
* VSCode
* Git *(opcional)*

#### Primeiro passo 
Extrair o `.zip` da página GitHub e descomprimir o ficheiro

*OU*

Abrir terminal (`CMD`, `PowerShell`, `Anaconda Prompt`, ou outros que reconheçam o comando `conda`), navegar até a pasta onde deseja instalar o repositório, e introduzir o seguinte código:
```
git clone https://github.com/Bugss05/IACD_Predicting_Hepatocellular_Carcinoma_through_SML.git
```

#### Segundo passo
Caso ainda não o tenha feito, abrir um dos terminais mecionados no passo anterior

#### Terceiro passo
Introduzir o seguinte código:
```
cd <diretorio_do_repositorio>
conda create -n dataSci --file requirements.txt
```
E esperar que a instalação esteja concluida

#### Quarto passo 
Abrir o VSCode e, na barra de pesquisa no topo do ecrã, digitar:
```
>Python: Select Interpreter
```
Clicar `Enter`, e selecionar o interpretador de Python que tenha como nome `Python 3.11.7 ('dataSci')`


#### Quinto passo
Navegar até ao diretório correto através do terminal e abrir o ficheiro `.py` ou `.ipynb` desejado:
```
cd <diretorio_do_repositorio>
polirTabela.ipynb (por exemplo)
```

## Bibliotecas utilizadas e as suas versões
#### As bibliotecas principais são:
>Lembrete: Não é necessário instalar estas bibliotecas individualmente.
<table>
  <thead>
    <tr>
      <th>Package</th>
      <th>Version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>pandas</td>
      <td>1.5.3</td>
    </tr>
    <tr>
      <td>numpy</td>
      <td>1.26.4</td>
    </tr>
    <tr>
      <td>jupyterlab</td>
      <td>4.0.11</td>
    </tr>
    <tr>
      <td>dataprep</td>
      <td>0.4.5</td>
    </tr>
    <tr>
      <td>streamlit</td>
      <td>1.32.0</td>
    </tr>
    <tr>
      <td>scikit-learn</td>
      <td>1.4.2</td>
    </tr>
    <tr>
      <td>matplotkib</td>
      <td>3.8.4</td>
    </tr>
    <tr>
      <td>altair</td>
      <td>5.0.1</td>
    </tr>
  </tbody>
</table>
<br>

**Nota:** As restantes bibliotecas utilizadas e as suas versões podem ser encontradas em [requirements.txt](requirements.txt), ou instalando o ambiente virtual como instruido acima e executando o comando `conda list`.

