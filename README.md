# Clima x Varejo — RJ x SP

Dashboard interativo que compara o clima de **São Paulo** e **Rio de Janeiro**
(últimos 14 dias observados + previsão) com o mesmo período do ano passado e com a
média histórica dos últimos 10 anos, para apoiar análises de sazonalidade e fluxo
de lojas no varejo de vestuário.

Dados: [Open-Meteo](https://open-meteo.com) — API pública, gratuita, sem
necessidade de chave/token.

## Como executar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

O navegador abre em `http://localhost:8501`.

## O que o app mostra

- **Semáforo de insights** (🟢 favorável / 🔴 atenção) com leitura textual do
  impacto no varejo, **ajustada à estação** (verão, outono, inverno e primavera).
- **KPIs**: Tmax e Tmin previstas vs. ano passado e média de 10 anos, **realizado
  recente** (14 dias), dias de chuva, dias quentes/noites frias e fim de semana.
- **Alertas**: frente fria (queda brusca da máxima), onda de calor e chuva em fim de
  semana que foi seco no ano passado (e vice-versa).
- **Horizonte de previsão** de 5, 7 ou 10 dias e **limiares ajustáveis** na barra lateral.
- **Gráficos**: temperatura (este ano x ano passado x média), desvio diário
  (barras vermelho/azul) e chuva.
- **SP x RJ**: comparação lado a lado (carregada sob demanda).
- Layout responsivo para celular, **link compartilhável** (`?cidade=Rio+de+Janeiro+(RJ)`)
  e **download CSV**.

## Notas técnicas

- Datas calculadas no fuso da cidade (o servidor da nuvem roda em UTC).
- Histórico baixado com concorrência baixa e cache de 24h; a API pública limita
  requisições simultâneas, então o app falha de forma graciosa e tenta de novo.
- As leituras de varejo são hipóteses sazonais baseadas em desvios climáticos,
  não previsões de vendas.
