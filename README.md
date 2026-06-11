# Lisbon Advertised Rental Market Prediction: An Augmented Approach

Este repositório contém a infraestrutura de dados, modelação e análise desenvolvida no âmbito da tese de mestrado em Data Science.

## Objetivo do Projeto
Prever o preço médio anunciado de arrendamento por $m^2$ nas freguesias de Lisboa através de um modelo **Augmented LightGBM**, integrando múltiplas fontes de dados urbanas, turísticas, demográficas e estruturais.
A inovação reside na integração de fontes heterogéneas:
* **Mercado habitacional:** Dados de anúncios de arrendamento do Idealista
* **Urbanismo:** Licenciamentos, pedidos e alvarás de construção/reabilitação urbana (RJUE)
* **Turismo:** Densidade e stock de Alojamento Local (RNAL)
* **Remote Sensing:** Luminosidade noturna via satélite (VIIRS)
* **Amenidades urbanas:** Transportes, escolas, saúde, museus, parques e mercados (CML/OpenStreetMap)
* **Demografia e estrutura habitacional:** Dados dos Censos 2011/2021

## Estrutura do Repositório
* `notebooks/`: Pipeline completo de Processamento, Feature Engineering e Modelação e Interpretação.
* `src/`: Scripts auxiliares e o simulador interativo em Streamlit.
* `models/`: Objetos treinados e dados necessários para correr o simulador e análises suplementares.
* `reports/plots/`: Visualizações principais usadas na tese, incluindo SHAP, ICE plots e métricas de performance.
* `supplementary_analysis/`: Análises suplementares adicionadas para reforçar a interpretabilidade e validação do modelo.
* `data/`: Dados brutos e processados usados no pipeline.


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
streamlit run src/app_simulator.py
```

## Supplementary Analysis

A pasta **supplementary_analysis/** contém análises adicionais desenvolvidas para complementar a tese e reforçar a transparência do modelo.

Estas análises foram adicionadas como material suplementar, sobretudo para responder a duas necessidades metodológicas:

1. disponibilizar ICE plots adicionais para variáveis que não foram todas incluídas no corpo principal da tese;
2. documentar uma análise de resíduos do modelo no test set de 2025.

A estrutura é a seguinte:

supplementary_analysis/ 
│ 
├── ice_plots/ 
│   ├── generate_ice_plots.py 
│   └── outputs/ 
│       ├── ice_Densidade_AL_km2.png 
│       ├── ice_Proporcao_Populacao_Residente_Estrangeira_2021.png 
│       ├── ice_PCT_Edificios_Antigos.png 
│       ├── ice_PCT_Edificios_Modernos.png 
│       ├── ice_Taxa_Conversao_Reabilitacao.png 
│       ├── ice_Stock_12M_ALV_Reabilitacao_Count.png 
│       └── ... 
│ 
└── residual_analysis/ 
   ├── generate_residual_analysis.py 
   ├── recover_y_test_aug.py 
   └── outputs/ 
      └── residual_plot_predicted_vs_residuals_eur.png            

### Supplementary ICE Plots
A subpasta supplementary_analysis/ice_plots/ gera ICE plots adicionais para várias variáveis do modelo Augmented LightGBM.

Estes gráficos complementam os ICE plots apresentados na tese e permitem analisar efeitos não lineares, padrões de saturação e heterogeneidade nas previsões do modelo.

### Nota sobre a escala dos ICE plots

O modelo foi treinado com a variável dependente transformada em logaritmo. Por isso, os ICE plots representam a previsão na escala logarítmica do preço anunciado por m².

Assim, estes gráficos devem ser interpretados como variações proporcionais esperadas na previsão, e não como variações diretas em euros por metro quadrado.

## Residual Analysis
A subpasta **supplementary_analysis/residual_analysis/** contém a análise de resíduos do modelo no test set de 2025.

O objetivo desta análise é avaliar se o modelo apresenta padrões sistemáticos de erro no período mais recente.

O residual plot representa:

- residual = actual price_m2 - predicted price_m2

ao longo dos valores previstos pelo modelo.

Ao contrário dos ICE plots, a análise de resíduos é feita na escala original, em euros por metro quadrado, depois de aplicar a transformação inversa ao target.
Interpretação da Análise de Resíduos

O residual plot suplementar mostra que, no test set de 2025, o modelo tende a subestimar os preços reais em média.

Esta subestimação é interpretada como sinal de temporal drift moderado, possivelmente associado à aceleração recente dos preços anunciados no mercado de arrendamento em Lisboa.

Esta evidência não invalida o modelo. Pelo contrário, reforça a importância de atualizar periodicamente o modelo com dados mais recentes em contexto de utilização real.

## Como Correr as Análises Suplementares

As análises suplementares dependem dos objetos já guardados na pasta **models/**, nomeadamente o modelo final, as features do test set e os valores reais do target.

Ficheiros relevantes:

- models/augmented_lightgbm_model.pkl
- models/X_test_aug.pkl
- models/y_test_aug.pkl
- models/metadata_test_aug.pkl

Para correr os ICE plots:

```bash
python supplementary_analysis/ice_plots/generate_ice_plots.py
```

Para correr a análise de resíduos:

```bash
python supplementary_analysis/residual_analysis/generate_residual_a
```
