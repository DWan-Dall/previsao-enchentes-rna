## Relatório

### Base de Dados (dataset_enchentes_sc.csv)

Foi utilizado um dataset simulado com:

- 300 registros entre 2022-01-01 e 2022-10-27

Variáveis:

- cidade, precipitacao_mm, umidade_solo, nivel_mar_m, elevacao_m, uso_solo, enchente

Cidades selecionadas: 

- Itajaí, Balneário Camboriú, Navegantes, Florianópolis, entre outras

### Metodologia

#### Pré-processamento

- Conversão de variáveis categóricas (Cidade, Uso_Solo) para fatores numéricos.

- Normalização das variáveis numéricas (Precipitacao_mm, Umidade_Solo, Nivel_Rio_m,Elevacao_m).

- Divisão dos dados em 70% treino e 30% teste

#### Modelagem

##### Modelo 1: Rede Neural Artificial (RNA)

- Implementada com caret e nnet

- Validação cruzada 5-fold

- Ajuste de hiperparâmetros: size (neurônios) e decay (penalização)

##### Modelo 2: Random Forest (baseline comparativo)

- Implementada com caret e randomForest

- Validação cruzada 5-fold

#### Avaliação

- Matrizes de confusão

- Curvas ROC

- Métrica de área sob a curva (AUC)

### Resultados

#### Matriz de Confusão

##### RNA:

- Accuracy: 100%

- Kappa: 1.0

- Sensibilidade e especificidade: 100%

##### Random Forest:

- Accuracy: 98.89%

- Sensibilidade: 100%, Especificidade: 0%

- Kappa: 0

#### Curvas ROC

- RNA: AUC = 1.0

- Random Forest: AUC = 0.9775

- Gráfico salvo como: resultados/curva_roc_comparativa.png


##### Possíveis melhorias futuras:

- Utilizar dados reais de precipitação e marés.

- Aplicar técnicas de balanceamento de classes.

- Testar outros algoritmos como XGBoost e redes neurais profundas (Deep Learning).

##### Anexos:

- Dataset utilizado - dataset_enchentes_sc.csv

- Gráficos e matrizes de confusão gerados

- Scripts .R