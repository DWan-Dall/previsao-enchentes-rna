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
dados <- read.csv("dataset_enchentes_sc.csv")
str(dados)


## 2. Pré-processamento dos dados

# Converter variáveis categóricas para fator
dados$uso_solo <- as.factor(dados$uso_solo)
dados$cidade <- as.factor(dados$cidade)

# Normalizar variáveis numéricas
dados_norm <- dados %>%
  mutate(across(c(precipitacao_mm, umidade_solo, nivel_mar_m, elevacao_m), scale))

# Separar em treino (70%) e teste (30%)
set.seed(123)
train_index <- createDataPartition(dados_norm$enchente, p = 0.7, list = FALSE)
train_data <- dados_norm[train_index, ]
test_data <- dados_norm[-train_index, ]


## 3. Treinamento da Rede Neural (RNA)

# Treinamento inicial com nnet
modelo <- nnet(enchente ~ precipitacao_mm + umidade_solo + nivel_mar_m + elevacao_m + uso_solo + cidade,
               data = train_data, size = 5, decay = 0.01, maxit = 200, trace = FALSE)

summary(modelo)

# Treinamento com validação cruzada
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Ajustar a variável target para "Sim"/"Não"
train_data$enchente <- as.factor(ifelse(train_data$enchente == 1, "Sim", "Não"))
test_data$enchente <- as.factor(ifelse(test_data$enchente == 1, "Sim", "Não"))

set.seed(123)
modelo_rna <- train(enchente ~ precipitacao_mm + umidade_solo + nivel_mar_m + elevacao_m + uso_solo + cidade,
                    data = train_data,
                    method = "nnet",
                    trControl = ctrl,
                    metric = "ROC",
                    tuneLength = 5,
                    trace = FALSE)

print(modelo_rna)


## 4. Treinamento do Random Forest (Baseline)

set.seed(123)
modelo_rf <- train(enchente ~ precipitacao_mm + umidade_solo + nivel_mar_m + elevacao_m + uso_solo + cidade,
                   data = train_data,
                   method = "rf",
                   trControl = ctrl,
                   metric = "ROC",
                   tuneLength = 3)

print(modelo_rf)


## 5. Avaliação dos Modelos

# Previsões RNA e RF
pred_rna <- predict(modelo_rna, newdata = test_data)
pred_rf <- predict(modelo_rf, newdata = test_data)

# Matrizes de confusão
conf_rna <- confusionMatrix(pred_rna, test_data$enchente)
conf_rf <- confusionMatrix(pred_rf, test_data$enchente)

print(conf_rna)
print(conf_rf)

# Probabilidades para curvas ROC
prob_rna <- predict(modelo_rna, newdata = test_data, type = "prob")[, "Sim"]
prob_rf <- predict(modelo_rf, newdata = test_data, type = "prob")[, "Sim"]

# Curvas ROC
roc_rna <- roc(response = test_data$enchente, predictor = prob_rna, levels = rev(levels(test_data$enchente)))
roc_rf <- roc(response = test_data$enchente, predictor = prob_rf, levels = rev(levels(test_data$enchente)))


## 6. Visualização das Curvas ROC

# Preparar dados para ggplot
roc_rna_data <- data.frame(TPR = roc_rna$sensitivities, FPR = 1 - roc_rna$specificities, Modelo = "RNA")
roc_rf_data <- data.frame(TPR = roc_rf$sensitivities, FPR = 1 - roc_rf$specificities, Modelo = "Random Forest")
roc_data <- bind_rows(roc_rna_data, roc_rf_data)

# Criar o gráfico

grafico_roc <- ggplot(roc_data, aes(x = FPR, y = TPR, color = Modelo)) +
  geom_line(size = 1.2) +
  geom_abline(linetype = "dashed", color = "gray") +
  theme_minimal(base_size = 14) +
  labs(
    title = "Curvas ROC - RNA vs Random Forest",
    subtitle = paste0("AUC RNA: ", round(auc(roc_rna), 3), " | AUC Random Forest: ", round(auc(roc_rf), 3)),
    x = "Taxa de Falsos Positivos (1 - Especificidade)",
    y = "Taxa de Verdadeiros Positivos (Sensibilidade)",
    color = "Modelo"
  ) +
  theme(
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(size = 12),
    legend.position = "bottom"
  )

# Mostrar o gráfico
print(grafico_roc)

# Salvar o gráfico
ggsave("resultados/curva_roc_comparativa.png", plot = grafico_roc, width = 8, height = 6, dpi = 300)


## 7. Salvar Resultados em Arquivos

# Salvar Matrizes de Confusão
write.csv(as.data.frame(conf_rna$table), "resultados/matriz_confusao_rna.csv", row.names = FALSE)
write.csv(as.data.frame(conf_rf$table), "resultados/matriz_confusao_rf.csv", row.names = FALSE)

# Salvar AUCs em txt
auc_texto <- paste0("AUC RNA: ", round(auc(roc_rna), 4), "\nAUC Random Forest: ", round(auc(roc_rf), 4))
writeLines(auc_texto, "resultados/auc_resultados.txt")


## 8. Conclusão Final

# Imprimir AUCs
auc_rna <- auc(roc_rna)
auc_rf <- auc(roc_rf)

cat("AUC RNA:", auc_rna, "\n")
cat("AUC Random Forest:", auc_rf, "\n")
