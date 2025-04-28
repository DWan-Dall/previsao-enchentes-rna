---
title: "Previsão de Enchentes com Redes Neurais Artificiais - v1.0"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## 1. Carregar pacotes e preparar ambiente

# Carregar pacotes necessários (comente install.packages se já tiver instalado)
#install.packages(c("tidyverse", "caret", "nnet", "pROC", "randomForest", "ggplot2"))

# install.packages("rmarkdown")

library(tidyverse)
library(caret)
library(nnet)
library(pROC)
library(randomForest)
library(ggplot2)

# Criar pasta 'resultados' para salvar saídas, se não existir
if (!dir.exists("resultados")) {
  dir.create("resultados")
}

# Carregar o dataset
dados <- read.csv("dataset_flood.csv")
str(dados)


## 2. Pré-processamento dos dados

# Converter variáveis categóricas para fator (dados)
dados$FloodProbability <- as.factor(dados$FloodProbability)

# Normalizar todas as variáveis de entrada (menos a FloodProbability, que é a variável alvo)
dados_norm <- dados %>%
  mutate(across(
    .cols = c(
      MonsoonIntensity, TopographyDrainage, RiverManagement, Deforestation, Urbanization,
      ClimateChange, DamsQuality, Siltation, AgriculturalPractices, Encroachments,
      IneffectiveDisasterPreparedness, DrainageSystems, CoastalVulnerability, Landslides,
      Watersheds, DeterioratingInfrastructure, PopulationScore, WetlandLoss,
      InadequatePlanning, PoliticalFactors
    ),
    .fns = scale
  ))

# Conferir se deu certo
str(dados_norm)

# Separar em treino (70%) e teste (30%)
set.seed(123)
train_index <- createDataPartition(dados_norm$FloodProbability, p = 0.7, list = FALSE)
train_data <- dados_norm[train_index, ]
test_data <- dados_norm[-train_index, ]


## 3. Treinamento da Rede Neural (RNA)

# Treinamento inicial com nnet (para multiclasse - 83 categorias)
modelo <- nnet(FloodProbability ~ .,
               data = train_data, size = 5, decay = 0.01, maxit = 200, trace = FALSE)

summary(modelo)

# Treinamento com validação cruzada
ctrl <- trainControl(method = "cv", number = 5)
# ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Ajustar a variável target para "Sim"/"Não"
# train_data$enchente <- as.factor(ifelse(train_data$enchente == 1, "Sim", "Não"))
# test_data$enchente <- as.factor(ifelse(test_data$enchente == 1, "Sim", "Não"))

set.seed(123)
modelo_rna <- train(FloodProbability ~ .,
                    data = train_data,
                    method = "nnet",
                    trControl = ctrl,
                    # metric = "ROC",
                    tuneLength = 5,
                    trace = FALSE)

print(modelo_rna)


## 4. Treinamento do Random Forest (Baseline)

set.seed(123)
modelo_rf <- train(FloodProbability ~ .,
                   data = train_data,
                   method = "rf",
                   trControl = ctrl,
                  #  metric = "ROC",
                   tuneLength = 3)

print(modelo_rf)

## 5. Avaliação dos Modelos

# Previsões RNA e RF
pred_rna <- predict(modelo_rna, newdata = test_data)
pred_rf <- predict(modelo_rf, newdata = test_data)

# Matrizes de confusão
conf_rna <- confusionMatrix(pred_rna, test_data$FloodProbability)
conf_rf <- confusionMatrix(pred_rf, test_data$FloodProbability)

print(conf_rna)
print(conf_rf)

# Salvar matrizes de confusão
write.csv(as.data.frame(conf_rna$table), "resultados/matriz_confusao_rna.csv", row.names = FALSE)
write.csv(as.data.frame(conf_rf$table), "resultados/matriz_confusao_rf.csv", row.names = FALSE)

# Calcular acurácias
acuracia_rna <- conf_rna$overall["Accuracy"]
acuracia_rf  <- conf_rf$overall["Accuracy"]

# Salvar AUCs em txt
texto_acuracia <- paste0("Acurácia RNA: ", round(acuracia_rna, 4), "\nAcurácia Random Forest: ", round(acuracia_rf, 4))
writeLines(texto_acuracia, "resultados/acuracia_resultados.txt")

