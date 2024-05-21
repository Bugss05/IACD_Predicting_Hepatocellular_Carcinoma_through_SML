Quando confrontados com os graficos resultantes de Hyperparameters dos vários algoritmos, surgiu a questão: como devemos escolher?
Assim, para além de ter em conta aspetos como o `test_size` e `n_neighbors` (exemplos de parâmetros do KNN), procuramos implementar outras métricas, nomeadamente **Sensitivity** (ou **Recall**) e **Specificity** cujas formas se apresentam abaixo: <br>
<br>
<div style="text-align:center">

${\text{Specificity} = \frac{\text{True Negatives}}{\text{True Negatives} + \text{False Positives}}}$

<br>

${\text{Sensitivity} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}}$

</div>
<br>

Tal significa, portanto, que a Sensitivity representa, em termos leigos, "dos resultados positivos (`1` ou `"Lives"`), quantos foram analisados corretamente" e a Specificity "dos resultados negativos (`0` ou `"Dies"`), quantos foram analisados corretamente". <br>

Ora, como sendo este um dataset representativo de um diagnóstico médico, é largamente preferível obter um falso positivo em vez de um falso negativo. Esse facto levou-nos a priorizar pontos que maximizem o Recall, garantindo que, um paciente que é, de facto, positivo, seja, categorizado como tal. <br>

Logo, no sentido de definir os melhores parâmetros criamos o gráfico de Accuracy em função de Recall, e escolhemos o ponto mais próximo ao caso ideal (ou seja, o ponto `(1,1)` ). Atente no seguinte mencionado: <br>

