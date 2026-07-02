# Clima x Varejo — RJ x SP

Dashboard interativo que compara a previsão do tempo atual em **São Paulo** e
**Rio de Janeiro** com o mesmo período do ano passado e com a média histórica
dos últimos 5 anos, para apoiar análises de sazonalidade no varejo de
vestuário (fluxo de lojas x temperatura/chuva).

Dados: [Open-Meteo](https://open-meteo.com) — API pública, gratuita, sem
necessidade de chave/token.

## Como executar localmente

1. (Recomendado) crie um ambiente virtual:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS/Linux
   ```

2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Rode o app:

   ```bash
   streamlit run app.py
   ```

4. O navegador abrirá automaticamente em `http://localhost:8501`.

## O que o app mostra

- **Seletor de cidade**: São Paulo (SP) ou Rio de Janeiro (RJ).
- **Previsão dos próximos 5 dias**: temperatura máx/mín e condição de chuva.
- **Comparativo YoY**: dados reais de temperatura e precipitação dos mesmos
  dias, um ano atrás.
- **Média histórica**: média dos últimos 5 anos para as mesmas datas de
  calendário, usada como referência de "normalidade" climática.
- **Painel de Insights de Varejo**: cards com o desvio (°C) vs. ano passado e
  vs. média histórica, alerta de chuva em fins de semana, e uma leitura
  textual do possível impacto no varejo de vestuário (ex.: inverno mais
  quente reduz procura por casacos).
- **Gráficos**: linha (temperatura, este ano x ano passado x média) e barras
  (precipitação, este ano x ano passado).

## Notas técnicas

- Cache de 1h (previsão) e 6h (histórico) via `st.cache_data` para evitar
  chamadas repetidas à API.
- Ano passado é obtido via API de arquivo histórico da Open-Meteo
  (`archive-api.open-meteo.com`), que traz dados reais observados (não
  previsão).
- Botão "Atualizar dados" na barra lateral limpa o cache manualmente.
