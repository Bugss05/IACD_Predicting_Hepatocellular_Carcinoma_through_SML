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

Pretende-se, portanto, desenvolver vários algoritmos SML (Supervised Machine Learning), capazes de determinar a probabilidade de sobrevivência dos pacientes após 1 ano do diagnóstico (aspeto identificado na coluna `"Class"` com `"Lives"` ou `"Dies"`). <br>


 ## Como opera o programa

Este programa trabalha com um intrepetador `Python` e usa um ambiente virtual `conda`  de forma a facilitar a instalação das dependências necessárias para a utilização do Jupyter Notebook. O notebook mencionado está guardado no ficheiro de nome [data.ipybn](data.ipybn) que, passo a passo, mostrará a progressão do projeto, bem como o nosso processo lógico e a forma como decidimos atacar o problema. <br>

A escolha da utilização de um ambiente `conda` derivou dos seguintes fatores:

* ``Bibliotecas extensas``: a implementação de um ambiente virtual automatiza a instalação das bibliotecas, facilitando o acesso e reduzindo o de tempo perdido na instalação das mesmas;
 
* ``Isolamento de dependências``: ao criar um ambiente separado, evitam-se conflitos entre bibliotecas de outros projetos e garante-se a compatibilidade;

* ``Organização``: sendo este ambiente naturalmente mais reduzido em relaçáo ao ambiente nativo da máquina, a sua utilização mantém a pasta do Python organizada e facilita a identificação de bibliotecas;

* ``Reprodutibilidade``: a criação de um ficheiro [requirements.txt](requirements.txt) facilita a partilha e a execução do código em diferentes máquinas, tornando o programa compatível em qualquer máquina;

* ``Gestão de versões``: pela simplicidade da ferramenta `conda`, torna-se fácil instalar e manter diferentes versões de bibliotecas para cada projeto, sem nunca correr o risco de causar conflitos de dependências;

* ``Leveza``: a transmissão e instalação do ambiente é facilitada com pelo ficheiro [requirements.txt](requirements.txt) sendo portanto apenas necessários menos que `30 KB` de espaço livre em disco para obter a lista detalhada com todas as bibliotecas utilizadas.

## Instalar o programa

#### Pré-Requisitos
* Conda
* Git *(opcional)*
* VSCode

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
conda install -n dataSci --file requirements.txt
```
E esperar que a instalação esteja concluida

#### Quarto passo 
Abrir o VSCode e, na barra de pesquisa no topo do ecrã, digitar:
```
>Python: Select Interpreter
```
Clicar `Enter`, e selecionar o interpretador de Python que tenha como nome `Python 3.11.9 ('dataSci')`


#### Quinto passo
Navegar até ao diretório correto através do terminal e abrir o ficheiro `.py` ou `.ipynb` desejado:
```
cd <diretorio_do_repositorio>
data.ipynb (por exemplo)
```
>**Nota:** caso pretenda utilizar o ficheiro [app.py](app.py), o codigo é ligeiramente diferente:
```
streamlit run app.py
```

## Bibliotecas utilizadas e as suas versões
#### As bibliotecas principais são:
>Lembrete : Não é necessário instalar nenhuma das bibliotecas.
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
      <td>1.23.5</td>
    </tr>
    <tr>
      <td>ydata-profiling</td>
      <td>4.1.1</td>
    </tr>
    <tr>
      <td>jupyter_core</td>
      <td>5.5.0</td>
    </tr>
    <tr>
      <td>dataprep</td>
      <td>0.4.5</td>
    </tr>
  </tbody>
</table>
<br><br><br>

#### Estas são todas as bibliotecas utilizadas:


<table>
  <thead>
    <tr>
      <th>Package</th>
      <th>Version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>aiohttp</td>
      <td>3.9.5</td>
    </tr>
    <tr>
      <td>aiosignal</td>
      <td>1.3.1</td>
    </tr>
    <tr>
      <td>annotated-types</td>
      <td>0.6.0</td>
    </tr>
    <tr>
      <td>anyio</td>
      <td>4.3.0</td>
    </tr>
    <tr>
      <td>argon2-cffi</td>
      <td>23.1.0</td>
    </tr>
    <tr>
      <td>argon2-cffi-bindings</td>
      <td>21.2.0</td>
    </tr>
    <tr>
      <td>arrow</td>
      <td>1.3.0</td>
    </tr>
    <tr>
      <td>asttokens</td>
      <td>2.4.1</td>
    </tr>
    <tr>
      <td>async-lru</td>
      <td>2.0.4</td>
    </tr>
    <tr>
      <td>async-timeout</td>
      <td>4.0.3</td>
    </tr>
    <tr>
      <td>attrs</td>
      <td>23.2.0</td>
    </tr>
    <tr>
      <td>Babel</td>
      <td>2.14.0</td>
    </tr>
    <tr>
      <td>backcall</td>
      <td>0.2.0</td>
    </tr>
    <tr>
      <td>beautifulsoup4</td>
      <td>4.12.3</td>
    </tr>
    <tr>
      <td>bleach</td>
      <td>6.1.0</td>
    </tr>
    <tr>
      <td>bokeh</td>
      <td>2.4.3</td>
    </tr>
    <tr>
      <td>certifi</td>
      <td>2024.2.2</td>
    </tr>
    <tr>
      <td>cffi</td>
      <td>1.16.0</td>
    </tr>
    <tr>
      <td>charset-normalizer</td>
      <td>3.3.2</td>
    </tr>
    <tr>
      <td>click</td>
      <td>8.1.7</td>
    </tr>
    <tr>
      <td>cloudpickle</td>
      <td>3.0.0</td>
    </tr>
    <tr>
      <td>colorama</td>
      <td>0.4.6</td>
    </tr>
    <tr>
      <td>comm</td>
      <td>0.2.2</td>
    </tr>
    <tr>
      <td>contourpy</td>
      <td>1.1.1</td>
    </tr>
    <tr>
      <td>cycler</td>
      <td>0.12.1</td>
    </tr>
    <tr>
      <td>dacite</td>
      <td>1.8.1</td>
    </tr>
    <tr>
      <td>dask</td>
      <td>2023.5.0</td>
    </tr>
    <tr>
      <td>dataprep</td>
      <td>0.4.5</td>
    </tr>
    <tr>
      <td>debugpy</td>
      <td>1.8.1</td>
    </tr>
    <tr>
      <td>decorator</td>
      <td>5.1.1</td>
    </tr>
    <tr>
      <td>defusedxml</td>
      <td>0.7.1</td>
    </tr>
    <tr>
      <td>exceptiongroup</td>
      <td>1.2.1</td>
    </tr>
    <tr>
      <td>executing</td>
      <td>0.8.3</td>
    </tr>
    <tr>
      <td>fastjsonschema</td>
      <td>2.19.1</td>
    </tr>
    <tr>
      <td>Flask</td>
      <td>2.2.5</td>
    </tr>
    <tr>
      <td>Flask-Cors</td>
      <td>3.0.10</td>
    </tr>
    <tr>
      <td>fonttools</td>
      <td>4.51.0</td>
    </tr>
    <tr>
      <td>fqdn</td>
      <td>1.5.1</td>
    </tr>
    <tr>
      <td>frozenlist</td>
      <td>1.4.1</td>
    </tr>
    <tr>
      <td>fsspec</td>
      <td>2024.3.1</td>
    </tr>
    <tr>
      <td>h11</td>
      <td>0.14.0</td>
    </tr>
    <tr>
      <td>htmlmin</td>
      <td>0.1.12</td>
    </tr>
    <tr>
      <td>httpcore</td>
      <td>1.0.5</td>
    </tr>
    <tr>
      <td>httpx</td>
      <td>0.27.0</td>
    </tr>
    <tr>
      <td>idna</td>
      <td>3.7</td>
    </tr>
    <tr>
      <td>ImageHash</td>
      <td>4.3.1</td>
    </tr>
    <tr>
      <td>importlib_metadata</td>
      <td>7.1.0</td>
    </tr>
    <tr>
      <td>importlib_resources</td>
      <td>6.4.0</td>
    </tr>
    <tr>
      <td>ipykernel</td>
      <td>6.29.4</td>
    </tr>
    <tr>
      <td>ipython</td>
      <td>8.12.3</td>
    </tr>
    <tr>
      <td>ipython-genutils</td>
      <td>0.2.0</td>
    </tr>
    <tr>
      <td>ipywidgets</td>
      <td>7.8.1</td>
    </tr>
    <tr>
      <td>isoduration</td>
      <td>20.11.0</td>
    </tr>
    <tr>
      <td>itsdangerous</td>
      <td>2.2.0</td>
    </tr>
    <tr>
      <td>jedi</td>
      <td>0.19.1</td>
    </tr>
    <tr>
      <td>Jinja2</td>
      <td>3.0.3</td>
    </tr>
    <tr>
      <td>joblib</td>
      <td>1.4.2</td>
    </tr>
    <tr>
      <td>json5</td>
      <td>0.9.25</td>
    </tr>
    <tr>
      <td>jsonpath-ng</td>
      <td>1.6.1</td>
    </tr>
    <tr>
      <td>jsonpointer</td>
      <td>2.4</td>
    </tr>
    <tr>
      <td>jsonschema</td>
      <td>4.22.0</td>
    </tr>
    <tr>
      <td>jsonschema-specifications</td>
      <td>2023.12.1</td>
    </tr>
    <tr>
      <td>jupyter</td>
      <td>1.0.0</td>
    </tr>
    <tr>
      <td>jupyter_client</td>
      <td>8.6.1</td>
    </tr>
    <tr>
      <td>jupyter_console</td>
      <td>6.6.3</td>
    </tr>
    <tr>
      <td>jupyter_core</td>
      <td>5.7.2</td>
    </tr>
    </tr>
    <tr>
      <td>jupyter_events</td>
      <td>0.10.0</td>
    </tr>
    <tr>
      <td>jupyter-lsp</td>
      <td>2.2.5</td>
    </tr>
    <tr>
      <td>jupyter_server</td>
      <td>2.14.0</td>
    </tr>
    <tr>
      <td>jupyter_server_terminals</td>
      <td>0.5.3</td>
    </tr>
    <tr>
      <td>jupyterlab</td>
      <td>4.1.8</td>
    </tr>
    <tr>
      <td>jupyterlab_pygments</td>
      <td>0.3.0</td>
    </tr>
    <tr>
      <td>jupyterlab_server</td>
      <td>2.27.1</td>
    </tr>
    <tr>
      <td>jupyterlab-widgets</td>
      <td>1.1.7</td>
    </tr>
    <tr>
      <td>kiwisolver</td>
      <td>1.4.5</td>
    </tr>
    <tr>
      <td>llvmlite</td>
      <td>0.41.1</td>
    </tr>
    <tr>
      <td>locket</td>
      <td>1.0.0</td>
    </tr>
    <tr>
      <td>MarkupSafe</td>
      <td>2.1.5</td>
    </tr>
    <tr>
      <td>matplotlib</td>
      <td>3.7.5</td>
    </tr>
    <tr>
      <td>matplotlib-inline</td>
      <td>0.1.7</td>
    </tr>
    <tr>
      <td>Metaphone</td>
      <td>0.6</td>
    </tr>
    <tr>
      <td>mistune</td>
      <td>3.0.2</td>
    </tr>
    <tr>
      <td>multidict</td>
      <td>6.0.5</td>
    </tr>
    <tr>
      <td>multimethod</td>
      <td>1.10</td>
    </tr>
    <tr>
      <td>nbclient</td>
      <td>0.10.0</td>
    </tr>
    <tr>
      <td>nbconvert</td>
      <td>7.16.4</td>
    </tr>
    <tr>
      <td>nbformat</td>
      <td>5.10.4</td>
    </tr>
    <tr>
      <td>nest-asyncio</td>
      <td>1.6.0</td>
    </tr>
    <tr>
      <td>networkx</td>
      <td>3.1</td>
    </tr>
    <tr>
      <td>nltk</td>
      <td>3.8.1</td>
    </tr>
      <td>notebook</td>
      <td>7.1.3</td>
    </tr>
    <tr>
      <td>notebook_shim</td>
      <td>0.2.4</td>
    </tr>
    <tr>
      <td>numba</td>
      <td>0.58.1</td>
    </tr>
    <tr>
      <td>numpy</td>
      <td>1.24.4</td>
    </tr>
    <tr>
      <td>overrides</td>
      <td>7.7.0</td>
    </tr>
    <tr>
      <td>packaging</td>
      <td>24.0</td>
    </tr>
    <tr>
      <td>pandas</td>
      <td>1.5.3</td>
    </tr>
    <tr>
      <td>pandocfilters</td>
      <td>1.5.1</td>
    </tr>
    <tr>
      <td>parso</td>
      <td>0.8.4</td>
    </tr>
    <tr>
      <td>partd</td>
      <td>1.4.1</td>
    </tr>
    <tr>
      <td>patsy</td>
      <td>0.5.6</td>
    </tr>
    <tr>
      <td>phik</td>
      <td>0.12.4</td>
    </tr>
    <tr>
      <td>pickleshare</td>
      <td>0.7.5</td>
    </tr>
    <tr>
      <td>pillow</td>
      <td>10.3.0</td>
    </tr>
    <tr>
      <td>pip</td>
      <td>24.0</td>
    </tr>
    <tr>
      <td>pkgutil_resolve_name</td>
      <td>1.3.10</td>
    </tr>
    <tr>
      <td>platformdirs</td>
      <td>4.2.1</td>
    </tr>
    <tr>
      <td>ply</td>
      <td>3.11</td>
    </tr>
    <tr>
      <td>prometheus_client</td>
      <td>0.20.0</td>
    </tr>
    <tr>
      <td>prompt-toolkit</td>
      <td>3.0.43</td>
    </tr>
    <tr>
      <td>psutil</td>
      <td>5.9.8</td>
    </tr>
    <tr>
      <td>pure-eval</td>
      <td>0.2.2</td>
    </tr>
    <tr>
      <td>pycparser</td>
      <td>2.22</td>
    </tr>
    <tr>
      <td>pydantic</td>
      <td>1.10.15</td>
    </tr>
    <tr>
      <td>pydantic_core</td>
      <td>2.18.2</td>
    </tr>
    <tr>
      <td>pydot</td>
      <td>1.4.2</td>
    </tr>
    <tr>
      <td>Pygments</td>
      <td>2.18.0</td>
    </tr>
        <tr>
      <td>pyparsing</td>
      <td>3.1.2</td>
    </tr>
    <tr>
      <td>python-crfsuite</td>
      <td>0.9.8</td>
    </tr>
    <tr>
      <td>python-dateutil</td>
      <td>2.9.0.post0</td>
    </tr>
    <tr>
      <td>python-json-logger</td>
      <td>2.0.7</td>
    </tr>
    <tr>
      <td>python-stdnum</td>
      <td>1.20</td>
    </tr>
    <tr>
      <td>pytz</td>
      <td>2024.1</td>
    </tr>
    <tr>
      <td>PyWavelets</td>
      <td>1.4.1</td>
    </tr>
    <tr>
      <td>pywin32</td>
      <td>306</td>
    </tr>
    <tr>
      <td>pywinpty</td>
      <td>2.0.13</td>
    </tr>
    <tr>
      <td>PyYAML</td>
      <td>6.0.1</td>
    </tr>
    <tr>
      <td>pyzmq</td>
      <td>26.0.3</td>
    </tr>
    <tr>
      <td>qtconsole</td>
      <td>5.5.2</td>
    </tr>
    <tr>
      <td>QtPy</td>
      <td>2.4.1</td>
    </tr>
    <tr>
      <td>rapidfuzz</td>
      <td>2.15.2</td>
    </tr>
    <tr>
      <td>referencing</td>
      <td>0.35.1</td>
    </tr>
    <tr>
      <td>regex</td>
      <td>2021.11.10</td>
    </tr>
    <tr>
      <td>requests</td>
      <td>2.31.0</td>
    </tr>
    <tr>
      <td>rfc3339-validator</td>
      <td>0.1.4</td>
    </tr>
    <tr>
      <td>rfc3986-validator</td>
      <td>0.1.1</td>
    </tr>
    <tr>
      <td>rpds-py</td>
      <td>0.18.0</td>
    </tr>
    <tr>
      <td>scipy</td>
      <td>1.10.1</td>
    </tr>
    <tr>
      <td>seaborn</td>
      <td>0.12.2</td>
    </tr>
    <tr>
      <td>Send2Trash</td>
      <td>1.8.3</td>
    </tr>
    <tr>
      <td>setuptools</td>
      <td>56.0.0</td>
    </tr>
    <tr>
      <td>six</td>
      <td>1.16.0</td>
    </tr>
    <tr>
      <td>sniffio</td>
      <td>1.3.1</td>
    </tr>
    <tr>
      <td>soupsieve</td>
      <td>2.5</td>
    </tr>
    <tr>
      <td>SQLAlchemy</td>
      <td>1.3.24</td>
    </tr>
    <tr>
      <td>stack-data</td>
      <td>0.5.1</td>
    </tr>
    <tr>
      <td>statsmodels</td>
      <td>0.14.1</td>
    </tr>
    <tr>
      <td>terminado</td>
      <td>0.18.1</td>
    </tr>
    <tr>
      <td>tinycss2</td>
      <td>1.3.0</td>
    </tr>
    <tr>
      <td>tomli</td>
      <td>2.0.1</td>
    </tr>
    <tr>
      <td>toolz</td>
      <td>0.12.1</td>
    </tr>
    <tr>
      <td>tornado</td>
      <td>6.4</td>
    </tr>
    <tr>
      <td>tqdm</td>
      <td>4.66.4</td>
    </tr>
    <tr>
      <td>traitlets</td>
      <td>5.14.3</td>
    </tr>
    <tr>
      <td>typeguard</td>
      <td>4.2.1</td>
    </tr>
    <tr>
      <td>types-python-dateutil</td>
      <td>2.9.0.20240316</td>
    </tr>
    <tr>
      <td>typing_extensions</td>
      <td>4.11.0</td>
    </tr>
    <tr>
      <td>tzdata</td>
      <td>2024.1</td>
    </tr>
    <tr>
      <td>uri-template</td>
      <td>1.3.0</td>
    </tr>
    <tr>
      <td>urllib3</td>
      <td>2.2.1</td>
    </tr>
    <tr>
      <td>varname</td>
      <td>0.8.3</td>
    </tr>
    <tr>
      <td>visions</td>
      <td>0.7.6</td>
    </tr>
    <tr>
      <td>wcwidth</td>
      <td>0.2.13</td>
    </tr>
    <tr>
      <td>webcolors</td>
      <td>1.13</td>
    </tr>
    <tr>
      <td>webencodings</td>
      <td>0.5.1</td>
    </tr>
    <tr>
      <td>websocket-client</td>
      <td>1.8.0</td>
    </tr>
    <tr>
      <td>Werkzeug</td>
      <td>3.0.2</td>
    </tr>
    <tr>
      <td>widgetsnbextension</td>
      <td>3.6.6</td>
    </tr>
    <tr>
      <td>wordcloud</td>
      <td>1.9.3</td>
    </tr>
    <tr>
      <td>yarl</td>
      <td>1.9.4</td>
    </tr>
    <tr>
      <td>ydata-profiling</td>
      <td>4.7.0</td>
    </tr>
    <tr>
      <td>zipp</td>
      <td>3.18.1</td>
    </tr>
  </tbody>
</table>
 
