# Lisbon Advertised Rental Market Prediction: An Augmented Approach

Este repositório contém a infraestrutura de dados e modelos da tese.

## Objetivo do Projeto
Prever o preço médio de arrendamento por $m^2$ nas freguesias de Lisboa através de um modelo **Augmented**. A inovação reside na integração de fontes heterogéneas:
* **Mercado:** Dados de anúncios (Idealista).
* **Urbanismo:** Licenciamentos e reabilitação urbana (RJUE).
* **Turismo:** Densidade de Alojamento Local (RNAL).
* **Remote Sensing:** Luminosidade noturna via satélite (VIIRS).
* **Demografia:** Dados estruturais (Censos 2011/2021).

## Estrutura do Repositório
* `notebooks/`: Pipeline completo de Processamento, Feature Engineering e Modelação.
* `reports/plots/`: Visualizações de impacto (SHAP values, ICE plots e métricas de performance).
* `src/`: Scripts auxiliares e o simulador interativo de preços.

## Ordem de Execução do Pipeline

O projeto foi desenvolvido através de um pipeline modular, desde a recolha dos dados até à construção do simulador. Para **replicar todo o estudo desde o início**, os notebooks devem ser executados pela seguinte ordem:

1. **Data Acquisition**  
   Execução dos scripts de recolha dos dados do Idealista.

2. **`VAR_*.ipynb`**  
   Processamento individual de cada fonte de dados, incluindo turismo, censos, construção, amenities e dados geoespaciais.

3. **`final_dataset.ipynb`**  
   Integra todas as fontes processadas numa única tabela mestre, usada posteriormente na modelação.

4. **`CRISP-ML.ipynb`**  
   Treina o modelo LightGBM, avalia o desempenho, executa validação, gera a análise de importância das variáveis e interpretações com SHAP/ICE.

5. **`app_simulator.py`**  
   Interface interativa em Streamlit para prever preços anunciados e simular cenários de política urbana.

---

### Nota importante sobre o simulador

Para correr o **Urban Policy Simulator**, **não é necessário executar todos os notebooks anteriores**.

O simulador utiliza diretamente os objetos já treinados e guardados na pasta `models/`, nomeadamente:

- `augmented_lightgbm_model.pkl` — modelo LightGBM final treinado;
- `X_test_aug.pkl` — observações de teste usadas como baseline para a simulação;
- `dicionario_freguesias.pkl` — correspondência entre freguesias e observações do dataset.

Além disso, o simulador utiliza o ficheiro geográfico:

- `data/processed/lisboa_poligonos_caop.geojson` — polígonos das freguesias de Lisboa usados no mapa interativo.

Assim, desde que estes ficheiros estejam presentes nas pastas corretas, o simulador pode ser executado diretamente com:

```bash
streamlit run app_simulator.py
