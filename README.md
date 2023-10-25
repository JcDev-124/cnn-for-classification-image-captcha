# README - Classificação de Imagens usando InceptionV3

## Visão Geral

Este repositório contém um projeto de classificação de imagens utilizando a arquitetura de rede neural convolucional InceptionV3. O modelo foi treinado para classificar imagens em cinco categorias: semáforos, faixas de pedestres, ônibus, hidrantes e bicicletas. Este README fornece uma visão geral do projeto, explica como utilizar o código e apresenta os resultados obtidos.

## Pré-requisitos

Antes de utilizar este projeto, você precisará ter as seguintes bibliotecas e ferramentas instaladas:

- Python 3.x
- TensorFlow
- scikit-learn
- Pandas
- Matplotlib
- Jupyter Notebook (opcional, para visualização de resultados)

Recomenda-se o uso de um ambiente virtual Python para isolar as dependências do projeto.

## Estrutura do Repositório

A estrutura do repositório é a seguinte:

- `README.md`: Este arquivo que fornece uma visão geral do projeto.
- `classification_inceptionv3.ipynb`: Jupyter Notebook com o código fonte para treinar e avaliar o modelo.
- `training/`: Diretório contendo imagens de treinamento organizadas por categoria.
- `validation/`: Diretório contendo imagens de validação organizadas por categoria.
- `test/`: Diretório contendo imagens de teste organizadas por categoria.

## Como Utilizar

Siga estas etapas para utilizar o código neste projeto:

1. **Configuração do Ambiente**: Certifique-se de que você tenha instalado todas as bibliotecas e ferramentas necessárias, conforme mencionado nos pré-requisitos.

2. **Preparação dos Dados**:
   - Organize suas imagens de treinamento, validação e teste em diretórios separados, agrupadas por categoria.
   - Atualize os diretórios especificando os caminhos corretos nos blocos de código onde estão definidos (variáveis `train_dir`, `validation_dir` e `test_dir`).

3. **Execução do Jupyter Notebook**:
   - Abra o arquivo `classification_inceptionv3.ipynb` em um ambiente Jupyter Notebook.
   - Execute cada célula de código sequencialmente para treinar o modelo, avaliar o desempenho e gerar visualizações.

4. **Avaliação dos Resultados**:
   - Após a execução das células, você obterá resultados detalhados, incluindo a precisão no conjunto de teste, matriz de confusão e gráficos de desempenho.

5. **Utilização do Modelo Treinado**:
   - O modelo treinado será salvo como 'model_inception.keras'. Você pode carregá-lo e utilizá-lo para fazer previsões em novas imagens.

## Resultados

Os resultados obtidos com este projeto foram os seguintes:

- Acurácia no conjunto de teste: 95%
- Acurácia na validação: 94%
- Acurácia no treinamento: 96%
- F1 Score: 95%
- Precisão global: 95%

## Personalização

Você pode personalizar o projeto e ajustar hiperparâmetros para melhorar o desempenho do modelo, como:

- Alterar o número de épocas de treinamento.
- Ajustar as taxas de aprendizado.
- Modificar o tamanho das imagens de entrada.
- Adicionar ou remover camadas de dropout.
- Experimentar diferentes arquiteturas de modelos.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir problemas (issues) ou enviar solicitações de pull (pull requests) para melhorar este projeto.

## Licença

Este projeto é distribuído sob a Licença MIT. Consulte o arquivo `LICENSE` para obter detalhes.

---

Este README fornece uma visão geral completa do projeto, como utilizá-lo e as principais informações relacionadas aos resultados e personalização. Sinta-se à vontade para adaptar e expandir este README conforme necessário para o seu projeto.
