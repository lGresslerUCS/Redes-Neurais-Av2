#  Implementação de Rede Neural MLP — Previsão de Desempenho de Jogadores de Basquete

**Disciplina:** FBX4054 - Inteligência Computacional  
**Instituição:** Universidade de Caxias do Sul (UCS)  

Este repositório contém a implementação de uma **Rede Neural Perceptron Multicamada (MLP)** construída estritamente "do zero" utilizando Python. O objetivo do modelo é prever se o desempenho de um jogador da NBA será "Bom" ou "Ruim" com base em suas estatísticas de jogo.

---

## Objetivo

O objetivo deste projeto é implementar uma rede neural do tipo **Perceptron Multicamadas (MLP — Multilayer Perceptron)** para prever o desempenho de jogadores de basquete.

A rede neural deverá ser desenvolvida do zero, utilizando uma linguagem de programação de livre escolha, sem o uso de bibliotecas ou frameworks especializados em *Machine Learning* para implementar, treinar ou avaliar o modelo.

Serão exploradas etapas de preparação dos dados, construção da arquitetura da rede neural, treinamento, avaliação dos resultados e interpretação do desempenho obtido.

## Etapas do Projeto

### 1. Preparação dos Dados

Nesta etapa, será realizada a preparação do conjunto de dados utilizado no treinamento e na avaliação da rede neural.

- Carregar e visualizar o dataset utilizando operações básicas de entrada e saída de dados e, se necessário, bibliotecas de manipulação de dados, como Pandas.
- Realizar a limpeza dos dados, identificando valores ausentes, inconsistências e possíveis registros duplicados.
- Executar o pré-processamento das variáveis, incluindo normalização ou padronização, conforme necessário.
- Separar os dados em conjuntos de treinamento e teste.

### 2. Implementação do Modelo

A rede neural MLP será construída do zero, sem a utilização de bibliotecas especializadas em *Machine Learning*.

As principais atividades incluem:

- Definir a arquitetura da rede neural, especificando o número de camadas e a quantidade de neurônios em cada camada.
- Implementar a inicialização dos pesos e dos vieses (*biases*).
- Implementar as operações matemáticas de propagação direta (*forward propagation*).
- Implementar uma função de ativação, como a função sigmoide.
- Definir e implementar uma função de perda para medir o erro entre as previsões e os valores reais.

### 3. Treinamento do Modelo

Nesta etapa, a rede neural será treinada para ajustar seus pesos e vieses, minimizando o erro das previsões.

- Implementar o algoritmo de retropropagação (*backpropagation*) para calcular os gradientes.
- Implementar o método de descida do gradiente (*gradient descent*) para atualizar os parâmetros da rede.
- Executar o treinamento utilizando o conjunto de dados de treinamento.
- Monitorar a evolução da função de perda ao longo das épocas.
- Ajustar parâmetros como taxa de aprendizado (*learning rate*) e número de épocas para buscar melhores resultados.

### 4. Avaliação e Ajuste do Modelo

O desempenho da rede neural será avaliado utilizando o conjunto de teste, verificando sua capacidade de generalização para dados não utilizados durante o treinamento.

Serão realizadas as seguintes atividades:

- Gerar previsões para os dados do conjunto de teste.
- Construir uma matriz de confusão para comparar as classes previstas com as classes reais.
- Calcular manualmente métricas de avaliação, como:
  - **Acurácia (*Accuracy*):** proporção de previsões corretas.
  - **Precisão (*Precision*):** proporção de previsões positivas que estão corretas.
  - **Revocação (*Recall*):** proporção de casos positivos reais identificados corretamente.
- Comparar os resultados obtidos com diferentes configurações de hiperparâmetros.
- Selecionar a configuração que apresente resultados mais adequados ao problema.

As métricas de classificação serão aplicadas considerando que a previsão do desempenho dos jogadores esteja formulada como um problema de classificação.

### 5. Interpretação e Conclusão

A etapa final consiste em interpretar os resultados e discutir a eficácia da rede neural na previsão do desempenho dos jogadores de basquete.

Serão abordados os seguintes pontos:

- Análise dos resultados obtidos durante o treinamento e a avaliação.
- Identificação das variáveis que apresentam maior influência nas previsões do modelo.
- Discussão sobre as limitações da abordagem e possíveis fontes de erro.
- Sugestões de melhorias na arquitetura, no pré-processamento ou nos hiperparâmetros.
- Conclusões sobre a aplicabilidade da rede neural MLP ao problema proposto.

## Entregáveis

Ao final do projeto, deverão ser entregues os seguintes materiais:

- **Relatório descritivo:** documento apresentando as cinco etapas do projeto, incluindo análises, arquitetura da rede neural, metodologia, métricas e resultados obtidos. O relatório será apresentado em aula.
- **Código-fonte:** implementação completa da rede neural, incluindo preparação dos dados, treinamento e avaliação do modelo.
- **Visualizações:** gráficos, tabelas e outras representações que auxiliem na compreensão dos resultados e dos principais insights identificados.

## Critérios de Avaliação

A avaliação do projeto considerará os seguintes critérios:

1. **Preparação dos dados:** qualidade da limpeza, organização e do pré-processamento utilizando uma linguagem de programação.
2. **Implementação da rede neural:** adequação e corretude da implementação da MLP do zero, incluindo arquitetura, funções de ativação, cálculo dos gradientes e atualização dos parâmetros.
3. **Avaliação e ajuste:** precisão dos cálculos das métricas e qualidade dos experimentos realizados sem bibliotecas especializadas em *Machine Learning*.
4. **Interpretação dos resultados:** clareza e profundidade das análises, conclusões e propostas de melhoria apresentadas no relatório.
5. **Apresentação oral:** clareza na exposição da metodologia, da implementação e dos resultados obtidos.

## Tecnologias e Ferramentas

As tecnologias serão definidas de acordo com a linguagem de programação escolhida para o desenvolvimento do projeto.

- **Linguagem de programação:** livre escolha.
- **Manipulação de dados:** operações básicas e bibliotecas auxiliares, como Pandas, quando necessário.
- **Implementação da MLP:** desenvolvida manualmente, sem frameworks especializados em *Machine Learning*.
- **Visualização dos resultados:** gráficos e tabelas para análise do desempenho do modelo.


# RESULTADOS: 

## Arquitetura e Implementação

A rede neural foi implementada sem o uso de bibliotecas focadas em Machine Learning (como TensorFlow, PyTorch ou o núcleo do Scikit-Learn) para o algoritmo de aprendizagem. Todo o processo de *Forward Pass* e *Backpropagation* foi programado de forma manual com operações matriciais da biblioteca NumPy.

* **Arquitetura:** Camada de Entrada + 1 Camada Oculta (16 neurônios) + Camada de Saída (1 neurônio binário).
* **Função de Ativação:** Sigmoide (com trava de segurança contra *overflow*).
* **Otimização:** Descida do Gradiente (*Gradient Descent*) padrão.
* **Hiperparâmetros:** 2000 épocas de treinamento com taxa de aprendizado (learning rate) de 0.01.

---

## Resultados e Métricas

Após o processamento (que inclui binarização de alvo, tratamento de nulos pela mediana e normalização via Min-Max Scaling), o conjunto de dados foi dividido em 80% para treino e 20% para teste. 

Os resultados extraídos da Matriz de Confusão para o conjunto de testes (124 instâncias) apresentaram excelente performance:

* **Acurácia Global:** 97,58%
* **Erro:** 2,42%
* **Especificidade:** 99,10%
* **Precisão:** 91,67%
* **Sensitividade (Recall):** 84,62%

**Variáveis de Maior Impacto:**  
As métricas que mais pesaram nas decisões da rede foram: Cestas de quadra (FG), Pontos (PTS), Idade (Age), Roubos de bola (STL) e Cestas de 3 pontos (3P).

---

## Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o Python instalado (versão 3.6 ou superior).
Para rodar a interface gráfica e o processamento, instale as dependências executando o comando abaixo no terminal:

```bash
pip install pandas numpy scikit-learn matplotlib
```

```bash
 python -m pip install customtkinter
```

```bash
python app.py
```

Link do repositorio: https://github.com/lGresslerUCS/Redes-Neurais-Av2

Link dos slides: https://docs.google.com/presentation/d/1wqD26aqUkt7ciry6pdgZE1XnYa8DOdfJF7GM6AjQwso/edit?usp=sharing