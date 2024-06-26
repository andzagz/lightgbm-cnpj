# Carregar as bibliotecas necessárias
library(readxl)
library(caret)
library(dplyr)
library(lightgbm)

# Carregar os dados
dados <- read_excel("dados_exportadosidlimpo.xlsx")

# Eliminar valores ausentes
dados <- na.omit(dados)

# Definir os parâmetros iniciais
params <- list(
  objective = "multiclass",
  num_class = length(unique(dados$ID)), # Substituído por length(unique(dados$ID))
  metric = "multi_logloss",
  num_leaves = 31,
  max_depth = -1,
  min_child_samples = 20,
  learning_rate = 0.1
)

# Estratificar os dados em treino e teste
index <- createDataPartition(dados$ID, p = 0.8, list = FALSE)
dados_treino <- dados[index, ]
dados_teste <- dados[-index, ]

# Remove as colunas "ID", "CNPJ" e "Razao_social" de dados_treino e dados_teste
dados_treino_limpos <- dados_treino[, !(names(dados_treino) %in% c("ID"))]
dados_teste_limpos <- dados_teste[, !(names(dados_teste) %in% c("ID"))]

# Treine o modelo
model <- lgb.train(
  params = params,
  data = lgb.Dataset(data = as.matrix(dados_treino_limpos), label = dados_treino$ID),
  nrounds = 100
)

# Faça previsões na base de teste
predictions <- predict(model, newdata = as.matrix(dados_teste_limpos))


# Imprimir as previsões para verificar
View(predictions)

library(openxlsx)

write.xlsx(predictions,"previsoeslightgbm.xlsx")
