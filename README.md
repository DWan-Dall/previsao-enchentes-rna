# Previsão de Enchentes com Redes Neurais Artificiais 🌧️🤖

Este projeto utiliza técnicas de Machine Learning (Redes Neurais e Random Forest) para prever a ocorrência de enchentes em cidades de Santa Catarina (Brasil), baseado em dados meteorológicos e geográficos.

---

## 📁 Estrutura do Projeto

- `Previsao_Enchentes.R`: Script principal de análise, modelagem, avaliação e exportação de resultados.
- `dataset_enchentes_sc.csv`: Base de dados utilizada para treinamento e teste dos modelos.
- `resultados/`: Pasta gerada automaticamente contendo:
  - `matriz_confusao_rna.csv`
  - `matriz_confusao_rf.csv`
  - `auc_resultados.txt`
  - `curva_roc_comparativa.png`

## 🚀 Como Executar o Projeto

1. Instale os pacotes necessários no R:

```r
install.packages(c("tidyverse", "caret", "nnet", "pROC", "randomForest", "ggplot2"))
```

2. Abra o arquivo Previsao_Enchentes.R no RStudio ou VSCode.

3. Execute o script linha por linha ou clique em Knit para gerar um relatório em HTML (Recurso do RStudio). 

4. Após a execução:

- A pasta 'resultados/' será criada
- Matrizes de confusão, gráficos e métricas de avaliação serão salvos automaticamente.

## 📊 Técnicas Utilizadas

- Normalização de variáveis
- Particionamento treino/teste
- Redes Neurais Artificiais (RNA/nnet)
- Random Forest (RF)
- Validação Cruzada (caret)
- Avaliação por Curva ROC e AUC (pROC)
- Visualização de dados (ggplot2) através dos gráficos gerados e salvando automaticamente na pasta resultados/


## 📈 Exemplos de Resultados

### 🎯 Gráfico Comparativo das Curvas ROC

<p align="center">

### 📄 Arquivos Gerados

- Matriz de Confusão RNA (matriz_confusao_rna.csv)
- Matriz de Confusão Random Forest (matriz_confusao_rf.csv)
- Resultados de AUC (auc_resultados.txt)

Exemplo de conteúdo do auc_resultados.txt:

```nginx
AUC RNA: 0.8921
AUC Random Forest: 0.9415
```

## 📜 Requisitos

- R (versão >= 4.0)
- RStudio ou VSCode
- Pacotes: tidyverse, caret, nnet, pROC, randomForest, ggplot2

## 📚 Referências

- [Documentação caret](https://topepo.github.io/caret/index.html)
- [Documentação pROC](https://cran.r-project.org/web/packages/pROC/pROC.pdf)
- [Documentação nnet](https://cran.r-project.org/web/packages/nnet/nnet.pdf)

<details>

<summary></summary>

### ✍️ Autoria
Projeto desenvolvido para o Mestrado Profissional em Computação Aplicada - UNIVALI<br>

Auxílio de suporte técnico da OpenAI ChatGPT<br>
Adaptado e customizado por <a href="https://github.com/DWan-Dall">DWD</a>💜.

</details>
