---
title: "Previsão de Enchentes com Redes Neurais Artificiais - v2.0"
output:
  html_document: default
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## 1. Carregar pacotes e preparar ambiente

# Carregar pacotes necessários (comente install.packages se já tiver instalado)
# install.packages(c("tidyverse", "caret", "nnet", "pROC", "ggplot2", "DMwR2", "smotefamily", "MuMIn", "glmnet"))

# install.packages("rmarkdown", "knitr")
# Rodar no bash para gerar o arquivo html/pdf e não esquecer de transformar os arquivos R em Rmd para execução
# Rscript -e "rmarkdown::render('Previsao_Enchentes_dataset_flood.Rmd')"
# Rscript -e "rmarkdown::render('Previsao_Enchentes_dataset_flood.Rmd', output_format = 'pdf_document')"
# Rscript -e "rmarkdown::render('relatorio.Rmd', output_format = 'pdf_document')"

library(tidyverse)
library(caret)
library(nnet)
library(pROC)
library(ggplot2)
library(DMwR2)
library(smotefamily)
# library(MuMIn)
library(glmnet)

# Criar pasta 'resultados' para salvar saídas, se não existir
if (!dir.exists("resultados")) {
  dir.create("resultados")
}

# Carregar o dataset
dados <- read.csv("dataset_flood.csv")
str(dados)

## 2. Pré-processamento dos dados

# Criando variável alvo binária - Threshold definido em 0.6: acima disso, risco é considerado "Alto"
dados <- dados %>%
  mutate(FloodRisk = ifelse(FloodProbability >= 0.6, "Alto", "Baixo"),
         FloodRisk = as.factor(FloodRisk))

# Normalizar todas as variáveis de entrada (menos a FloodRisk, que é a variável alvo)
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
set.seed(42)
train_index <- createDataPartition(dados_norm$FloodRisk, p = 0.7, list = FALSE) # Troca da variável alvo
train_data <- dados_norm[train_index, ]
test_data <- dados_norm[-train_index, ]

## 2.1 Aplicando SMOTE no treino

# Convertendo alvo para binário numérico (SMOTE do smotefamily precisa disso)
train_data$FloodRisk_num <- ifelse(train_data$FloodRisk == "Alto", 1, 0)

# Aplicar SMOTE
smote_result <- SMOTE(X = train_data[, -which(names(train_data) %in% c("FloodRisk", "FloodProbability", "FloodRisk_num"))],
                      target = train_data$FloodRisk_num,
                      K = 5,
                      dup_size = 2  # <== Equivalente a perc.over = 200
                      )

# Recuperar dataset final
train_data_bal <- smote_result$data
train_data_bal$FloodRisk <- as.factor(ifelse(train_data_bal$class == 1, "Alto", "Baixo"))
train_data_bal$class <- NULL

# Separar
baixos <- train_data_bal %>% filter(FloodRisk == "Baixo")
altos <- train_data_bal %>% filter(FloodRisk == "Alto")

# Manter 2x mais baixos do que altos
baixos <- baixos %>% sample_n(size = nrow(altos) * 2)

# Juntar
train_data_bal <- bind_rows(altos, baixos)

# ## 2.2 Seleção de variáveis com dredge (baseado em AIC)

# # Ajustar um modelo GLM para avaliação com dredge (precisa ser GLM, e não nnet)
# modelo_glm <- glm(FloodRisk ~ ., data = train_data_bal, family = binomial)

# # Apçicar o dredge para testar combinações de variáveis
# options(na.action = "na.fail") 
# modelo_dredge <- dredge(modelo_glm)

# # Selecionar o melhor modelo (menor AIC)
# melhor_modelo <- get.models(modelo_dredge, subset = 1)[[1]]

# # Ver quais variáveis ficaram
# variaveis_selecionadas <- names(coef(melhor_modelo))[-1]

# print(variaveis_selecionadas)

# # 2.2 Seleção de variáveis com RFE (Recursive Feature Elimination)
# set.seed(123)
# rfe_ctrl <- rfeControl(functions = caretFuncs, method = "cv", number = 5)

# # Usamos as variáveis numéricas normalizadas (sem FloodRisk)
# rfe_result <- rfe(x = train_data_bal %>% select(-FloodRisk),
#                   y = train_data_bal$FloodRisk,
#                   sizes = c(5, 10, 15, 20),
#                   rfeControl = rfe_ctrl,
#                   method = "nnet",
#                   trace = FALSE)

# # Visualizar as variáveis escolhidas
# print(rfe_result)
# predictors(rfe_result)  # <- Isso retorna os nomes das variáveis selecionadas

# # 2.2 Definindo pesos com glmnet (LASSO)

# Remove variáveis que não serão usadas (FloodProbability e FloodRisk)
dados_lasso <- train_data_bal %>%
  select(-FloodRisk)

# Garante que x será uma matriz numérica com 2 ou mais colunas
x <- as.matrix(dados_lasso)

# Resposta binária (0 = Baixo, 1 = Alto)
y <- ifelse(train_data_bal$FloodRisk == "Alto", 1, 0)

# Ajuste do modelo com validação cruzada
cv_lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")

# Salvar gráfico do lambda (validação cruzada do LASSO)
png("resultados/cv_lasso.png", width = 800, height = 600)
plot(cv_lasso, main = "Validação Cruzada - LASSO")
dev.off()

# Coeficientes no melhor lambda
coef_lasso <- coef(cv_lasso, s = "lambda.min")

# Variáveis selecionadas
variaveis_selecionadas <- rownames(coef_lasso)[coef_lasso[, 1] != 0]
variaveis_selecionadas <- setdiff(variaveis_selecionadas, "(Intercept)")

print(variaveis_selecionadas)

texto_variaveis <- paste0("Variáveis Selecionadas: ", paste(variaveis_selecionadas, collapse = ", "))
writeLines(texto_variaveis, "resultados/variaveis_selecionadas.txt")

## 3. Treinamento da Rede Neural (RNA) - com dados balanceados

ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
# form <- as.formula(paste("FloodRisk ~", paste(variaveis_selecionadas, collapse = " + ")))

# set.seed(123)
# modelo_rna <- train(form,
#                     data = train_data_bal,
#                     method = "nnet",
#                     trControl = ctrl,
#                     metric = "ROC",
#                     tuneLength = 5,
#                     trace = FALSE)

# print(modelo_rna)

# # Treinar rede neural com as variáveis selecionadas pelo RFE
# modelo_rna <- train(
#   FloodRisk ~ .,  # Temporariamente, use só as selecionadas
#   data = train_data_bal[, c(predictors(rfe_result), "FloodRisk")],
#   method = "nnet",
#   trControl = ctrl,
#   metric = "ROC",
#   tuneLength = 5,
#   trace = FALSE
# )

# Fórmula da RNA com apenas as variáveis selecionadas
formula_rna <- as.formula(paste("FloodRisk ~", paste(variaveis_selecionadas, collapse = " + ")))

# Treinar novamente a RNA com essas variáveis
set.seed(123)
modelo_rna_lasso <- train(formula_rna,
                          data = train_data_bal,
                          method = "nnet",
                          trControl = ctrl,
                          metric = "ROC",
                          tuneLength = 5,
                          trace = FALSE)

print(modelo_rna_lasso)

## 4. Avaliação do Modelo

# Previsões e probabilidades
pred_rna <- predict(modelo_rna_lasso, newdata = test_data)
prob_rna <- predict(modelo_rna_lasso, newdata = test_data, type = "prob")

# Matrizes de confusão
conf_rna <- confusionMatrix(pred_rna, test_data$FloodRisk)

print(conf_rna)

# Salvar matrizes de confusão
write.csv(as.data.frame(conf_rna$table), "resultados/matriz_confusao_rna.csv", row.names = FALSE)

# Adicionado Curva ROC e AUC
roc_rna <- roc(response = test_data$FloodRisk,
               predictor = prob_rna[["Alto"]],
               levels = rev(levels(test_data$FloodRisk)))

plot(roc_rna, main = "Curva ROC - RNA com SMOTE")
auc(roc_rna)

# Preparar dados para ggplot
roc_rna_data <- data.frame(TPR = roc_rna$sensitivities, FPR = 1 - roc_rna$specificities, Modelo = "RNA")
roc_data <- bind_rows(roc_rna_data)

# Criar o gráfico

grafico_roc <- ggplot(roc_data, aes(x = FPR, y = TPR, color = Modelo)) +
  geom_line(linewidth = 1.2) +
  geom_abline(linetype = "dashed", color = "gray") +
  theme_minimal(base_size = 14) +
  labs(
    title = "Curvas ROC - RNA",
    subtitle = paste0("AUC RNA: ", round(auc(roc_rna), 3)),
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

# Salvar o gráfico
ggsave("resultados/curva_roc_rna_smote.png", plot = grafico_roc, width = 8, height = 6, dpi = 300)


# Calcular acurácias
acuracia_rna <- conf_rna$overall["Accuracy"]

# Salvar AUCs em txt
texto_acuracia <- paste0("Acurácia RNA: ", round(acuracia_rna, 4))
writeLines(texto_acuracia, "resultados/acuracia_resultados.txt")

