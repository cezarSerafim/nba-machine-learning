# 🏀 NBA WinShares Predictor: Machine Learning Pipeline

## 📖 Sobre o Projeto
Desenvolvido como parte de uma avaliação acadêmica, este projeto aplica técnicas de Machine Learning (Aprendizado Supervisionado de Regressão) para prever a métrica `WinShares` de jogadores da NBA com base em suas estatísticas de jogo. O tema e a base de dados esportiva foram escolhidos a partir de uma lista de opções fornecidas pelo professor para a elaboração do trabalho.

O foco técnico principal desta implementação foi a estruturação de um **Pipeline de Dados** robusto utilizando o Scikit-Learn. O objetivo foi aplicar as melhores práticas de validação de modelos, realizando a divisão dos dados (Treino e Teste) *antes* de qualquer etapa de preparação, garantindo assim a prevenção contra o *Data Leakage* (vazamento de dados) na imputação de valores nulos.

## 🛠️ Tecnologias e Bibliotecas Utilizadas
* **Linguagem:** Python
* **Ambiente:** Jupyter Notebook / Anaconda
* **Processamento de Dados:** Pandas e NumPy
* **Machine Learning:** Scikit-Learn (`Pipeline`, `RandomForestRegressor`, `SelectKBest`, `SimpleImputer`)

## 🧠 Arquitetura do Pipeline
A construção do modelo seguiu as seguintes etapas:
1. **Separação de Dados:** Divisão da base em 75% para treinamento e 25% para testes com `train_test_split`.
2. **Imputação Automática:** Tratamento de valores nulos (NaN) preenchendo com a média através do `SimpleImputer` integrado ao Pipeline.
3. **Feature Selection:** Seleção automatizada das 10 variáveis estatísticas de maior impacto utilizando o `SelectKBest` (com a função de score `f_regression`).
4. **Treinamento do Modelo:** Regressão baseada em árvores de decisão com o algoritmo `RandomForestRegressor` utilizando 100 estimadores.

## 📊 Resultados e Análise Crítica
Na base de testes (dados ocultos ao modelo durante o treino), foram obtidas as seguintes métricas de avaliação:
* **Erro Quadrático Médio (MSE):** ~0.0101
* **R² Score:** 0.9990

**💡 Insights Analíticos:**
Apesar dos valores matematicamente excelentes indicarem que o modelo explica quase 100% da variância dos dados, a análise crítica do contexto esportivo revelou um cenário de "vazamento de dados" analítico. 

A variável alvo `WinShares` é uma composição matemática direta das variáveis `OffensiveWinShares` e `DefensiveWinShares`. O algoritmo de Feature Selection identificou corretamente essa forte correlação e as inseriu no modelo. Isso fez com que a IA aprendesse uma equação direta ao invés de prever um cenário esportivo incerto. 

Como próximos passos para aumentar o real valor preditivo do modelo para uma equipe técnica da NBA, essas duas variáveis específicas devem ser removidas da base de dados antes do treinamento, forçando o algoritmo a encontrar padrões em métricas puras de jogo (como arremessos, assistências e rebotes).
