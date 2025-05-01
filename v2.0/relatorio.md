## Relatório

### Base de Dados (dataset_flood.csv)

Foi utilizado um dataset simulado com:

- 50.000 registros e 21 variáveis (incluindo a variável-alvo FloodProbability)

Variáveis:

- Colunas: variáveis relacionadas a características ambientais e sociais (ex: MonsoonIntensity (Intensidade das Monções), Urbanization (Urbanização), DrainageSystems (Sistemas de Drenagem), etc.), com FloodProbability (Probabilidade de Inundação) como variável alvo, sendo a mesma numérica contínua (não multiclasse), sendo um problema de regressão (prever uma probabilidade numérica de inundação) - A variável-alvo FloodProbability é contínua, parecendo ser uma probabilidade entre 0 e 1.

### Metodologia

#### Pré-processamento

- Conversão de variáveis categóricas para fatores
 
- Normalização das variáveis numéricas

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

- Acurácia (Accuracy): 98,89%

- Kappa: 0,97

- Sensibilidade: 98%

- Especificidade: 100%

##### Random Forest:

- Acurácia (Accuracy): 97,78%

- Kappa: 0,95

- Sensibilidade: 96%

- Especificidade: 99%

#### Curvas ROC/AUC

Não comparados ROC pois gráficos ROC/AUC realmente precisam ser ignorados ou refeitos de outra maneira (ex: One-vs-Rest para cada classe, mas é muito mais complicado), pois é mais usado para problemas binários (2 classes: "Sim" ou "Não").

#### Possíveis melhorias futuras:

- Inclusão de dados reais de níveis de rio, precipitação e uso do solo atualizado.

- Testes com algoritmos como XGBoost, LightGBM e redes neurais profundas (Deep Learning).

- Aplicação de técnicas de balanceamento de dados para eventos raros.

##### Anexos:

- Dataset utilizado - dataset_flood.csv

- Gráficos e matrizes de confusão gerados.

- Scripts .R