"""
Dashboard de Clima x Varejo — Rio de Janeiro x São Paulo
Compara a previsão do tempo atual com o mesmo período do ano passado
e com a média histórica, para apoiar decisões de varejo de vestuário.

Fonte de dados: Open-Meteo (https://open-meteo.com) — API pública e gratuita,
sem necessidade de chave/token.
"""

import re
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuração geral
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Clima x Varejo | RJ x SP",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CITIES = {
    "São Paulo (SP)": {"lat": -23.5505, "lon": -46.6333, "tz": "America/Sao_Paulo"},
    "Rio de Janeiro (RJ)": {"lat": -22.9068, "lon": -43.1729, "tz": "America/Sao_Paulo"},
}

FORECAST_DAYS = 5
HIST_YEARS = 5  # quantos anos para trás usar na "média histórica"

WEATHER_CODES = {
    0: ("☀️", "Céu limpo"),
    1: ("🌤️", "Poucas nuvens"),
    2: ("⛅", "Parcialmente nublado"),
    3: ("☁️", "Nublado"),
    45: ("🌫️", "Nevoeiro"),
    48: ("🌫️", "Nevoeiro c/ geada"),
    51: ("🌦️", "Garoa fraca"),
    53: ("🌦️", "Garoa"),
    55: ("🌦️", "Garoa forte"),
    56: ("🌦️", "Garoa congelante"),
    57: ("🌦️", "Garoa congelante forte"),
    61: ("🌧️", "Chuva fraca"),
    63: ("🌧️", "Chuva"),
    65: ("🌧️", "Chuva forte"),
    66: ("🌧️", "Chuva congelante"),
    67: ("🌧️", "Chuva congelante forte"),
    71: ("🌨️", "Neve fraca"),
    73: ("🌨️", "Neve"),
    75: ("🌨️", "Neve forte"),
    77: ("🌨️", "Grãos de neve"),
    80: ("🌧️", "Pancadas de chuva fracas"),
    81: ("🌧️", "Pancadas de chuva"),
    82: ("⛈️", "Pancadas de chuva fortes"),
    85: ("🌨️", "Pancadas de neve fracas"),
    86: ("🌨️", "Pancadas de neve fortes"),
    95: ("⛈️", "Trovoada"),
    96: ("⛈️", "Trovoada c/ granizo"),
    99: ("⛈️", "Trovoada c/ granizo forte"),
}

WEEKDAY_PT = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weathercode"
DAILY_VARS_ARCHIVE = "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode"


# ---------------------------------------------------------------------------
# Funções de acesso à API (com cache)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_forecast(lat: float, lon: float, tz: str, days: int) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": DAILY_VARS,
        "timezone": tz,
        "forecast_days": days,
    }
    r = requests.get(FORECAST_URL, params=params, timeout=15)
    r.raise_for_status()
    daily = r.json()["daily"]
    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    return df


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def fetch_archive(lat: float, lon: float, tz: str, start: date, end: date) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": DAILY_VARS_ARCHIVE,
        "timezone": tz,
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=15)
    r.raise_for_status()
    daily = r.json().get("daily", {})
    if not daily or not daily.get("time"):
        return pd.DataFrame(columns=["time"] + DAILY_VARS_ARCHIVE.split(","))
    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    return df


def safe_replace_year(d: date, years_back: int) -> date:
    """Subtrai N anos de uma data, tratando 29/fev."""
    try:
        return d.replace(year=d.year - years_back)
    except ValueError:
        return d.replace(year=d.year - years_back, day=28)


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def build_comparison(lat: float, lon: float, tz: str, today: date) -> dict:
    forecast_dates = [today + timedelta(days=i) for i in range(FORECAST_DAYS)]

    forecast_df = fetch_forecast(lat, lon, tz, FORECAST_DAYS)

    # Ano passado — mesmas datas, 1 ano atrás (dados reais/históricos)
    ly_start = safe_replace_year(forecast_dates[0], 1)
    ly_end = safe_replace_year(forecast_dates[-1], 1)
    last_year_df = fetch_archive(lat, lon, tz, ly_start, ly_end)

    # Média histórica — mesmas datas de calendário, últimos N anos
    per_offset_max, per_offset_min, per_offset_precip = (
        [[] for _ in range(FORECAST_DAYS)],
        [[] for _ in range(FORECAST_DAYS)],
        [[] for _ in range(FORECAST_DAYS)],
    )
    for y in range(1, HIST_YEARS + 1):
        start = safe_replace_year(forecast_dates[0], y)
        end = safe_replace_year(forecast_dates[-1], y)
        yr_df = fetch_archive(lat, lon, tz, start, end)
        for i in range(min(FORECAST_DAYS, len(yr_df))):
            row = yr_df.iloc[i]
            if pd.notna(row.get("temperature_2m_max")):
                per_offset_max[i].append(row["temperature_2m_max"])
            if pd.notna(row.get("temperature_2m_min")):
                per_offset_min[i].append(row["temperature_2m_min"])
            if pd.notna(row.get("precipitation_sum")):
                per_offset_precip[i].append(row["precipitation_sum"])

    hist_avg = pd.DataFrame(
        {
            "offset": list(range(FORECAST_DAYS)),
            "temperature_2m_max": [
                sum(v) / len(v) if v else None for v in per_offset_max
            ],
            "temperature_2m_min": [
                sum(v) / len(v) if v else None for v in per_offset_min
            ],
            "precipitation_sum": [
                sum(v) / len(v) if v else None for v in per_offset_precip
            ],
        }
    )

    return {
        "forecast_dates": forecast_dates,
        "forecast": forecast_df,
        "last_year": last_year_df,
        "hist_avg": hist_avg,
    }


# ---------------------------------------------------------------------------
# Funções auxiliares de apresentação
# ---------------------------------------------------------------------------


def weekday_label(d: date) -> str:
    return WEEKDAY_PT[d.weekday()]


def get_season(d: date) -> str:
    m = d.month
    if m in (12, 1, 2):
        return "Verão"
    if m in (3, 4, 5):
        return "Outono"
    if m in (6, 7, 8):
        return "Inverno"
    return "Primavera"


def fmt_delta(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}°C"


def md_bold_to_html(text: str) -> str:
    """Converte **negrito** markdown em <b> — necessário pois o card de
    insight é montado como HTML bruto (unsafe_allow_html), que não passa
    pelo parser de markdown do Streamlit."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


# ---------------------------------------------------------------------------
# CSS — estilo dashboard financeiro (dark mode contrastado)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: #161b26;
        border: 1px solid #262d3d;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 0.80rem;
        color: #8b93a7;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #f2f4f8;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #8b93a7;
        margin-top: 2px;
    }
    .insight-card {
        background: linear-gradient(135deg, #161b26 0%, #1c2233 100%);
        border-left: 4px solid #4f8cff;
        border-radius: 10px;
        padding: 20px 22px;
        margin: 14px 0;
    }
    .insight-card.alert {
        border-left-color: #ff6b6b;
    }
    .insight-card.positive {
        border-left-color: #34d399;
    }
    .day-card {
        background: #161b26;
        border: 1px solid #262d3d;
        border-radius: 10px;
        padding: 14px 10px;
        text-align: center;
    }
    .day-card .day-name { font-weight: 600; color: #f2f4f8; font-size: 0.95rem; }
    .day-card .day-icon { font-size: 1.8rem; margin: 6px 0; }
    .day-card .day-temp { color: #f2f4f8; font-size: 1.0rem; }
    .day-card .day-temp .max { font-weight: 700; }
    .day-card .day-temp .min { color: #8b93a7; }
    .day-card .day-rain { font-size: 0.78rem; color: #6fb1ff; margin-top: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🌦️ Clima x Varejo")
st.sidebar.caption("Monitoramento climático para análise de fluxo/vendas")

city_name = st.sidebar.selectbox("Selecione a capital", list(CITIES.keys()))
city = CITIES[city_name]

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Sobre o app**

    Compara a previsão dos próximos dias com os dados reais
    do mesmo período do ano passado e com a média histórica
    (últimos 5 anos), por cidade.

    Dados: [Open-Meteo](https://open-meteo.com) (gratuito, sem chave).
    """
)

if st.sidebar.button("🔄 Atualizar dados"):
    st.cache_data.clear()

# ---------------------------------------------------------------------------
# Carregamento de dados
# ---------------------------------------------------------------------------

today = date.today()

with st.spinner(f"Buscando dados climáticos para {city_name}..."):
    try:
        data = build_comparison(city["lat"], city["lon"], city["tz"], today)
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao consultar a API Open-Meteo: {e}")
        st.stop()

forecast_df = data["forecast"]
last_year_df = data["last_year"]
hist_avg_df = data["hist_avg"]
forecast_dates = data["forecast_dates"]

n = min(len(forecast_df), FORECAST_DAYS)

# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------

st.title(f"📍 {city_name} — Painel de Clima para Varejo")
st.caption(
    f"Previsão para os próximos {FORECAST_DAYS} dias, comparada aos mesmos dias de "
    f"{today.year - 1} e à média dos últimos {HIST_YEARS} anos. "
    f"Estação atual: **{get_season(today)}**."
)

# ---------------------------------------------------------------------------
# Cards diários — previsão atual
# ---------------------------------------------------------------------------

st.subheader("Previsão dos próximos dias")
day_cols = st.columns(n)
for i in range(n):
    row = forecast_df.iloc[i]
    d = forecast_dates[i]
    icon, desc = WEATHER_CODES.get(int(row["weathercode"]), ("❓", "N/D"))
    with day_cols[i]:
        st.markdown(
            f"""
            <div class="day-card">
                <div class="day-name">{weekday_label(d)} {d.strftime('%d/%m')}</div>
                <div class="day-icon">{icon}</div>
                <div class="day-temp">
                    <span class="max">{row['temperature_2m_max']:.0f}°</span>
                    &nbsp;/&nbsp;
                    <span class="min">{row['temperature_2m_min']:.0f}°</span>
                </div>
                <div class="day-rain">💧 {row.get('precipitation_probability_max', 0):.0f}% chuva</div>
                <div class="day-rain">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Cálculo dos desvios (Insights de Varejo)
# ---------------------------------------------------------------------------

avg_max_now = forecast_df["temperature_2m_max"].iloc[:n].mean()
avg_min_now = forecast_df["temperature_2m_min"].iloc[:n].mean()
rain_days_now = int((forecast_df["precipitation_sum"].iloc[:n] > 1.0).sum())

n_ly = min(n, len(last_year_df))
avg_max_ly = last_year_df["temperature_2m_max"].iloc[:n_ly].mean() if n_ly else None
rain_days_ly = (
    int((last_year_df["precipitation_sum"].iloc[:n_ly] > 1.0).sum()) if n_ly else None
)

avg_max_hist = hist_avg_df["temperature_2m_max"].mean()

deviation_yoy = (avg_max_now - avg_max_ly) if avg_max_ly is not None else None
deviation_hist = (
    (avg_max_now - avg_max_hist) if pd.notna(avg_max_hist) else None
)

# alerta de fim de semana: chuva este ano x seco ano passado (ou vice-versa)
weekend_alert = None
for i in range(min(n, n_ly)):
    d = forecast_dates[i]
    if d.weekday() >= 5:  # sábado=5, domingo=6
        rain_now = forecast_df["precipitation_sum"].iloc[i] > 1.0
        rain_ly = last_year_df["precipitation_sum"].iloc[i] > 1.0
        if rain_now != rain_ly:
            weekend_alert = {
                "day": d,
                "rain_now": rain_now,
                "rain_ly": rain_ly,
            }
            break

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------

st.subheader("Painel de Insights")

kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Temp. média prevista (máx)</div>
            <div class="metric-value">{avg_max_now:.1f}°C</div>
            <div class="metric-sub">próximos {n} dias</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[1]:
    if deviation_yoy is not None:
        color = "#ff6b6b" if deviation_yoy > 0 else "#4f8cff"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Desvio vs. ano passado</div>
                <div class="metric-value" style="color:{color}">{fmt_delta(deviation_yoy)}</div>
                <div class="metric-sub">vs. {today.year - 1}: {avg_max_ly:.1f}°C</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">Desvio vs. ano passado</div>'
            '<div class="metric-value">N/D</div></div>',
            unsafe_allow_html=True,
        )

with kpi_cols[2]:
    if deviation_hist is not None:
        color = "#ff6b6b" if deviation_hist > 0 else "#4f8cff"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Desvio vs. média histórica</div>
                <div class="metric-value" style="color:{color}">{fmt_delta(deviation_hist)}</div>
                <div class="metric-sub">média {HIST_YEARS} anos: {avg_max_hist:.1f}°C</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">Desvio vs. média histórica</div>'
            '<div class="metric-value">N/D</div></div>',
            unsafe_allow_html=True,
        )

with kpi_cols[3]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Dias de chuva previstos</div>
            <div class="metric-value">{rain_days_now} / {n}</div>
            <div class="metric-sub">ano passado: {rain_days_ly if rain_days_ly is not None else 'N/D'} / {n_ly}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Card de insight textual (impacto no varejo)
# ---------------------------------------------------------------------------

season = get_season(today)
insight_lines = []
card_class = "insight-card"

if deviation_yoy is not None:
    direction = "mais quente" if deviation_yoy > 0 else "mais frio"
    insight_lines.append(
        f"🌡️ Os próximos {n} dias em **{city_name}** devem ficar, em média, "
        f"**{abs(deviation_yoy):.1f}°C {direction}** que o mesmo período em {today.year - 1}."
    )

if deviation_hist is not None:
    direction_h = "acima" if deviation_hist > 0 else "abaixo"
    insight_lines.append(
        f"📊 Isso representa **{abs(deviation_hist):.1f}°C {direction_h}** da média "
        f"histórica dos últimos {HIST_YEARS} anos para essas datas."
    )

# Recomendação de varejo baseada em estação + desvio
if deviation_yoy is not None:
    if season == "Inverno" and deviation_yoy > 1:
        insight_lines.append(
            "👕 **Alerta varejo:** inverno mais quente que o normal tende a "
            "**reduzir a procura por casacos, botas e roupas de frio** — considere "
            "antecipar promoções ou ajustar mix de vitrine."
        )
        card_class += " alert"
    elif season == "Inverno" and deviation_yoy < -1:
        insight_lines.append(
            "👕 **Oportunidade varejo:** frio mais intenso que o ano passado tende a "
            "**impulsionar vendas de peças de inverno** — bom momento para reforçar estoque."
        )
        card_class += " positive"
    elif season == "Verão" and deviation_yoy < -1:
        insight_lines.append(
            "👕 **Alerta varejo:** verão mais ameno que o normal pode "
            "**reduzir o fluxo para roupas leves e moda praia**."
        )
        card_class += " alert"
    elif season == "Verão" and deviation_yoy > 1:
        insight_lines.append(
            "👕 **Oportunidade varejo:** calor acima do normal tende a "
            "**favorecer vendas de verão e aumentar o fluxo em shoppings com ar-condicionado**."
        )
        card_class += " positive"

if weekend_alert is not None:
    dstr = f"{weekday_label(weekend_alert['day'])} {weekend_alert['day'].strftime('%d/%m')}"
    if weekend_alert["rain_now"] and not weekend_alert["rain_ly"]:
        insight_lines.append(
            f"☔ **Atenção ao fim de semana:** chuva prevista para **{dstr}**, enquanto no "
            f"ano passado esse dia foi seco — possível **queda no fluxo de lojas físicas**."
        )
        card_class += " alert"
    elif not weekend_alert["rain_now"] and weekend_alert["rain_ly"]:
        insight_lines.append(
            f"☀️ **Fim de semana favorável:** **{dstr}** deve ficar seco, diferente do ano "
            f"passado que teve chuva — possível **aumento no fluxo de lojas físicas**."
        )
        card_class += " positive"

if not insight_lines:
    insight_lines.append("Sem desvios relevantes identificados para o período.")

insight_html = "<br><br>".join(md_bold_to_html(line) for line in insight_lines)
st.markdown(
    f'<div class="{card_class}">{insight_html}</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Gráficos comparativos
# ---------------------------------------------------------------------------

st.subheader("Comparativo de Temperatura — Este ano x Ano passado x Média histórica")

x_labels = [f"{weekday_label(d)} {d.strftime('%d/%m')}" for d in forecast_dates[:n]]
x_labels_ly = [
    f"{weekday_label(safe_replace_year(d, 1))} {safe_replace_year(d, 1).strftime('%d/%m')}"
    for d in forecast_dates[:n]
]

tab1, tab2 = st.tabs(["🌡️ Temperatura", "🌧️ Precipitação"])

with tab1:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=forecast_df["temperature_2m_max"].iloc[:n],
            name=f"Máx. {today.year} (atual)",
            mode="lines+markers",
            line=dict(color="#ff6b6b", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=forecast_df["temperature_2m_min"].iloc[:n],
            name=f"Mín. {today.year} (atual)",
            mode="lines+markers",
            line=dict(color="#4f8cff", width=3),
        )
    )
    if n_ly:
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=last_year_df["temperature_2m_max"].iloc[:n_ly],
                name=f"Máx. {today.year - 1} (ano passado)",
                mode="lines+markers",
                line=dict(color="#ff6b6b", width=2, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=last_year_df["temperature_2m_min"].iloc[:n_ly],
                name=f"Mín. {today.year - 1} (ano passado)",
                mode="lines+markers",
                line=dict(color="#4f8cff", width=2, dash="dot"),
            )
        )
    if hist_avg_df["temperature_2m_max"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=hist_avg_df["temperature_2m_max"],
                name=f"Máx. média {HIST_YEARS} anos",
                mode="lines",
                line=dict(color="#8b93a7", width=2, dash="dash"),
            )
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=40, l=10, r=10, b=10),
        yaxis_title="Temperatura (°C)",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=x_labels,
            y=forecast_df["precipitation_sum"].iloc[:n],
            name=f"{today.year} (atual)",
            marker_color="#4f8cff",
        )
    )
    if n_ly:
        fig2.add_trace(
            go.Bar(
                x=x_labels,
                y=last_year_df["precipitation_sum"].iloc[:n_ly],
                name=f"{today.year - 1} (ano passado)",
                marker_color="#8b93a7",
            )
        )
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=40, l=10, r=10, b=10),
        yaxis_title="Precipitação (mm)",
        height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Tabela de dados brutos
# ---------------------------------------------------------------------------

with st.expander("📄 Ver dados brutos"):
    table = pd.DataFrame(
        {
            "Data (este ano)": [d.strftime("%d/%m/%Y") for d in forecast_dates[:n]],
            "Máx atual (°C)": forecast_df["temperature_2m_max"].iloc[:n].values,
            "Mín atual (°C)": forecast_df["temperature_2m_min"].iloc[:n].values,
            "Chuva atual (mm)": forecast_df["precipitation_sum"].iloc[:n].values,
            "Data (ano passado)": x_labels_ly,
            "Máx ano passado (°C)": last_year_df["temperature_2m_max"].iloc[:n_ly].values
            if n_ly
            else [None] * n,
            "Mín ano passado (°C)": last_year_df["temperature_2m_min"].iloc[:n_ly].values
            if n_ly
            else [None] * n,
            "Chuva ano passado (mm)": last_year_df["precipitation_sum"].iloc[:n_ly].values
            if n_ly
            else [None] * n,
            f"Máx média {HIST_YEARS}a (°C)": hist_avg_df["temperature_2m_max"].values[:n],
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

st.caption(
    "Dados fornecidos por Open-Meteo.com. Este painel tem finalidade analítica/informativa "
    "e não substitui fontes oficiais de meteorologia."
)
