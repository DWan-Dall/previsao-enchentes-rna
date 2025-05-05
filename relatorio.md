# RELATÓRIO DE ESTUDO DO REPOSITÓRIO REFERÊNTE A V2.0

## 1. Descrição do Problema
O trabalho busca prever enchentes utilizando Redes Neurais Artificiais (RNAs), uma abordagem justificada pela crescente incidência de eventos climáticos extremos. A escolha se justifica como base nas regiões como Santa Catarina, que sofrem impactos frequentes e severos. 
As RNAs são capazes de capturar relações complexas entre variáveis ambientais e urbanas, oferecendo vantagem frente a modelos tradicionais.

## 2. Base de Dados Utilizada

- Fonte: Kaggle – Flood Prediction Dataset
- Tamanho: ~50.000 registros e 21 variáveis
- Variável-alvo inicial: FloodProbability
- Futuro objetivo: Construção de dataset mais localizado para regiões costeiras brasileiras

## 3. Pré-processamento dos Dados

- Conversão de variáveis categóricas em fatores
- Normalização das variáveis numéricas
- Divisão dos dados em 70% treino / 30% teste
- Criação da variável binária FloodRisk com base no FloodProbability

```R
FloodRisk <- ifelse(FloodProbability >= 0.6, "Alto", "Baixo")
```

## 4. Modelagem com Redes Neurais Artificiais

- Implementação feita em R com os pacotes caret e nnet
- Uso de validação cruzada (5-fold) para garantir generalização
- Ajuste de hiperparâmetros: size (número de neurônios na camada oculta) e decay (penalização)

## 5. Estratégias Avançadas

- Balanceamento de Classes: SMOTE com DMwR::SMOTE() para lidar com desbalanceamento entre "Alto" e "Baixo"
- Seleção de Variáveis: LASSO via glmnet para redução de dimensionalidade

### Métricas de Avaliação:

- Matriz de Confusão
- Acurácia
- Curva ROC e AUC (via pROC)

## 6. Resultados Obtidos

A RNA conseguiu distinguir casos de risco real (“Alto”) de forma robusta após balanceamento e seleção de variáveis.
A inclusão da curva ROC/AUC melhorou a interpretação da performance em relação à acurácia isolada. 

**Conclusão:** O modelo se mostrou eficaz na detecção de risco de enchentes, com forte potencial para futuras implementações em bases mais específicas da realidade regional.

