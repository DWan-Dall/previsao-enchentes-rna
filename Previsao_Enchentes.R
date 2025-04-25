
---
title: "Previsão de Enchentes com Redes Neurais Artificiais"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## 1. Carregando pacotes e dados (instalar se necessário)
install.packages(c("tidyverse", "caret", "nnet", "pROC", "randomForest", "ggplot2"))

library(tidyverse)
library(caret)
library(nnet)
library(pROC)
library(randomForest)
library(ggplot2)

# Carregar o dataset
dados <- read.csv("dataset_enchentes_sc.csv")
str(dados)

## 2. Pré-processamento - Converter uso_solo para fator
dados$uso_solo <- as.factor(dados$uso_solo)
dados$cidade <- as.factor(dados$cidade)

# Normalização das variáveis numéricas
dados_norm <- dados %>%
  mutate(across(c(precipitacao_mm, umidade_solo, nivel_mar_m, elevacao_m), scale))

# Separar treino e teste
set.seed(123)
train_index <- createDataPartition(dados_norm$enchente, p = 0.7, list = FALSE)
train_data <- dados_norm[train_index, ]
test_data <- dados_norm[-train_index, ]

## 3. Treinamento com nnet (Rede Neural de uma camada oculta)
modelo <- nnet(enchente ~ precipitacao_mm + umidade_solo + nivel_mar_m + elevacao_m + uso_solo + cidade,
               data = train_data, size = 5, decay = 0.01, maxit = 200, trace = FALSE)

summary(modelo)

## 3.1 Modelagem com Validação Cruzada RNA
# Configuração da validação cruzada
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Convertendo target para fator
train_data$enchente <- as.factor(ifelse(train_data$enchente == 1, "Sim", "Não"))
test_data$enchente <- as.factor(ifelse(test_data$enchente == 1, "Sim", "Não"))

# Treinando modelo RNA
set.seed(123)
modelo_rna <- train(enchente ~ precipitacao_mm + umidade_solo + nivel_mar_m + elevacao_m + uso_solo + cidade,
                    data = train_data,
                    method = "nnet",
                    trControl = ctrl,
                    metric = "ROC",
                    tuneLength = 5,
                    trace = FALSE)

print(modelo_rna)

## 4. Comparação: Random Forest (Baseline)
set.seed(123)
modelo_rf <- train(enchente ~ precipitacao_mm + umidade_solo + nivel_mar_m + elevacao_m + uso_solo + cidade,
                   data = train_data,
                   method = "rf",
                   trControl = ctrl,
                   metric = "ROC",
                   tuneLength = 3)

print(modelo_rf)

## 5. Avaliação do Modelo

# Previsão
pred_prob <- predict(modelo, newdata = test_data, type = "raw")
roc_curve <- roc(response = test_data$enchente, predictor = pred_prob)

pred_class <- ifelse(pred_prob > 0.5, 1, 0)

# Métricas
conf_mat <- confusionMatrix(as.factor(pred_class), as.factor(test_data$enchente))
conf_mat

# Curva ROC
roc_curve <- roc(test_data$enchente, pred)
plot(roc_curve, main = "Curva ROC - RNA")
auc(roc_curve)


####################

# Previsões RNA
pred_rna <- predict(modelo_rna, newdata = test_data)
confusionMatrix(pred_rna, test_data$enchente)

# Previsões RF
pred_rf <- predict(modelo_rf, newdata = test_data)
confusionMatrix(pred_rf, test_data$enchente)

# ROC Curve Comparativa
prob_rna <- predict(modelo_rna, newdata = test_data, type = "prob")[, "Sim"]
prob_rf <- predict(modelo_rf, newdata = test_data, type = "prob")[, "Sim"]

roc_rna <- roc(response = test_data$enchente, predictor = prob_rna, levels = rev(levels(test_data$enchente)))
roc_rf <- roc(response = test_data$enchente, predictor = prob_rf, levels = rev(levels(test_data$enchente)))

# Extraindo dados para o ggplot
roc_rna_data <- data.frame(
  TPR = roc_rna$sensitivities,
  FPR = 1 - roc_rna$specificities,
  Modelo = "RNA"
)

roc_rf_data <- data.frame(
  TPR = roc_rf$sensitivities,
  FPR = 1 - roc_rf$specificities,
  Modelo = "Random Forest"
)

roc_data <- bind_rows(roc_rna_data, roc_rf_data)

# Plot usando ggplot2
ggplot(roc_data, aes(x = FPR, y = TPR, color = Modelo)) +
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

## 6. Conclusão

# Imprimir AUC dos dois modelos
auc_rna <- auc(roc_rna)
auc_rf <- auc(roc_rf)

cat("AUC RNA:", auc_rna, "\\n")
cat("AUC Random Forest:", auc_rf, "\\n")


## 7. Salvar o gráfico em alta qualidade
ggsave("curva_roc_comparativa.png", width = 8, height = 6, dpi = 300)
