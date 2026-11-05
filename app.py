"""
Dashboard de Clima x Varejo — São Paulo x Rio de Janeiro

Compara o clima atual (últimos 14 dias observados + previsão) com o mesmo
período do ano passado e com a média histórica (10 anos), traduzindo os
desvios em leitura de fluxo de lojas e sazonalidade do varejo de vestuário.

Fonte de dados: Open-Meteo (https://open-meteo.com) — API pública e gratuita,
sem necessidade de chave/token.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
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
    initial_sidebar_state="auto",
)

CITIES = {
    "São Paulo (SP)": {"lat": -23.5505, "lon": -46.6333, "tz": "America/Sao_Paulo"},
    "Rio de Janeiro (RJ)": {"lat": -22.9068, "lon": -43.1729, "tz": "America/Sao_Paulo"},
}

PAST_DAYS = 14         # janela retroativa fixa (realizado recente)
HIST_YEARS = 10        # anos usados na média histórica
RAIN_MM = 1.0          # limiar de "dia de chuva" (mm)
BRT = ZoneInfo("America/Sao_Paulo")

WEATHER_CODES = {
    0: ("☀️", "Céu limpo"), 1: ("🌤️", "Poucas nuvens"), 2: ("⛅", "Parc. nublado"),
    3: ("☁️", "Nublado"), 45: ("🌫️", "Nevoeiro"), 48: ("🌫️", "Nevoeiro"),
    51: ("🌦️", "Garoa fraca"), 53: ("🌦️", "Garoa"), 55: ("🌦️", "Garoa forte"),
    56: ("🌦️", "Garoa gelada"), 57: ("🌦️", "Garoa gelada"),
    61: ("🌧️", "Chuva fraca"), 63: ("🌧️", "Chuva"), 65: ("🌧️", "Chuva forte"),
    66: ("🌧️", "Chuva gelada"), 67: ("🌧️", "Chuva gelada"),
    71: ("🌨️", "Neve"), 73: ("🌨️", "Neve"), 75: ("🌨️", "Neve"), 77: ("🌨️", "Neve"),
    80: ("🌧️", "Pancadas fracas"), 81: ("🌧️", "Pancadas"), 82: ("⛈️", "Pancadas fortes"),
    85: ("🌨️", "Neve"), 86: ("🌨️", "Neve"),
    95: ("⛈️", "Trovoada"), 96: ("⛈️", "Trovoada c/ granizo"), 99: ("⛈️", "Trovoada c/ granizo"),
}

WEEKDAY_PT = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weathercode"
DAILY_VARS_ARCHIVE = "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode"

TONE = {"good": "#4fd18c", "bad": "#ff6b6b", "warn": "#ffb454", "cold": "#4f8cff", "neutral": "#f2f4f8"}


# ---------------------------------------------------------------------------
# Acesso à API
# ---------------------------------------------------------------------------


def _get_json(url: str, params: dict, retries: int = 3) -> dict:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429:  # limite de requisições: espera e tenta de novo
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            last = e
            time.sleep(0.8 * (attempt + 1))
    raise last or RuntimeError("A API Open-Meteo não respondeu (limite de requisições).")


def _to_df(payload: dict, cols: list[str]) -> pd.DataFrame:
    daily = payload.get("daily") or {}
    if not daily.get("time"):
        return pd.DataFrame(columns=["time"] + cols)
    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"]).dt.date
    return df


def safe_replace_year(d: date, years_back: int) -> date:
    """Subtrai N anos de uma data, tratando 29/fev."""
    try:
        return d.replace(year=d.year - years_back)
    except ValueError:
        return d.replace(year=d.year - years_back, day=28)


def align(df: pd.DataFrame | None, dates: list[date], years_back: int) -> pd.DataFrame | None:
    """Reindexa por data (deslocada N anos) — robusto a anos bissextos."""
    if df is None or df.empty:
        return None
    keys = [safe_replace_year(d, years_back) for d in dates]
    return df.set_index("time").reindex(keys).reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_current(lat: float, lon: float, tz: str, past_days: int, forecast_days: int):
    payload = _get_json(
        FORECAST_URL,
        {
            "latitude": lat, "longitude": lon, "daily": DAILY_VARS, "timezone": tz,
            "past_days": past_days, "forecast_days": forecast_days,
        },
    )
    return _to_df(payload, DAILY_VARS.split(",")), datetime.now(BRT)


@st.cache_data(ttl=86400, show_spinner=False)
def load_history(lat: float, lon: float, tz: str, start: date, end: date, n_years: int) -> dict:
    """Baixa, em paralelo, a mesma janela de calendário dos últimos N anos."""

    def one(y: int):
        try:
            payload = _get_json(
                ARCHIVE_URL,
                {
                    "latitude": lat, "longitude": lon, "daily": DAILY_VARS_ARCHIVE, "timezone": tz,
                    "start_date": safe_replace_year(start, y).isoformat(),
                    "end_date": safe_replace_year(end, y).isoformat(),
                },
            )
            return y, _to_df(payload, DAILY_VARS_ARCHIVE.split(","))
        except Exception:
            return y, None

    # concorrência baixa: a API pública limita requisições simultâneas por IP
    with ThreadPoolExecutor(max_workers=3) as ex:
        result = {y: df for y, df in ex.map(one, range(1, n_years + 1)) if df is not None}
    if len(result) < max(1, n_years // 2):
        # exceção não é cacheada: evita guardar por 24h um histórico incompleto
        raise RuntimeError("A API Open-Meteo está limitando requisições no momento.")
    return result


def build_data(city: dict, today: date, horizon: int) -> dict:
    dates = [today + timedelta(days=o) for o in range(-PAST_DAYS, horizon)]
    cur_raw, fetched_at = load_current(city["lat"], city["lon"], city["tz"], PAST_DAYS, horizon)
    hist_raw = load_history(city["lat"], city["lon"], city["tz"], dates[0], dates[-1], HIST_YEARS)
    years = {y: align(df, dates, y) for y, df in hist_raw.items()}
    return {
        "dates": dates,
        "horizon": horizon,
        "current": align(cur_raw, dates, 0),
        "years": {y: d for y, d in years.items() if d is not None},
        "fetched_at": fetched_at,
    }


# ---------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------


def get_season(d: date) -> str:
    m = d.month
    if m in (12, 1, 2):
        return "Verão"
    if m in (3, 4, 5):
        return "Outono"
    if m in (6, 7, 8):
        return "Inverno"
    return "Primavera"


# Para cada estação: o que o calor / o frio acima do normal significa para o varejo de vestuário
SEASON_READ = {
    "Verão": {
        "warm": ("good", "calor acima do normal tende a **favorecer moda praia, peças leves e o fluxo em shoppings** com ar-condicionado."),
        "cold": ("bad", "verão mais ameno que o normal tende a **reduzir a procura por moda praia e roupas leves**."),
    },
    "Outono": {
        "warm": ("bad", "outono mais quente que o normal tende a **atrasar a transição para a coleção de frio** — risco de estoque de casacos parado e de liquidação precoce."),
        "cold": ("good", "frio antecipado tende a **impulsionar tricôs, jaquetas e meia-estação** — bom momento para antecipar reposição."),
    },
    "Inverno": {
        "warm": ("bad", "inverno mais quente que o normal tende a **reduzir a procura por casacos, botas e roupas de frio** — considere antecipar promoções ou ajustar o mix de vitrine."),
        "cold": ("good", "frio acima do normal tende a **impulsionar vendas de peças de inverno** — bom momento para reforçar estoque."),
    },
    "Primavera": {
        "warm": ("good", "primavera mais quente que o normal tende a **antecipar a demanda de verão** (moda praia, leves, sandálias) — oportunidade de antecipar a vitrine de verão."),
        "cold": ("bad", "primavera mais fria que o normal tende a **atrasar a coleção de verão** e manter a demanda por meia-estação — risco de estoque de verão encalhado."),
    },
}


def pct(a, b):
    if a is None or b is None or b == 0:
        return None
    return (a / b - 1) * 100


def compute_metrics(data: dict, params: dict) -> dict:
    P, F = PAST_DAYS, data["horizon"]
    dates, cur, yrs = data["dates"], data["current"], data["years"]
    ly = yrs.get(1)

    def A(df, col):
        return None if df is None else df[col].to_numpy(dtype=float)

    def smean(a, sl):
        if a is None:
            return None
        s = a[sl]
        return None if np.all(np.isnan(s)) else float(np.nanmean(s))

    past, fwd = slice(0, P), slice(P, P + F)
    tmax, tmin, rain = A(cur, "temperature_2m_max"), A(cur, "temperature_2m_min"), A(cur, "precipitation_sum")
    tmax_ly, tmin_ly, rain_ly = A(ly, "temperature_2m_max"), A(ly, "temperature_2m_min"), A(ly, "precipitation_sum")

    def hist_avg(col):
        if not yrs:
            return None
        return pd.DataFrame({y: A(d, col) for y, d in yrs.items()}).mean(axis=1).to_numpy()

    tmax_h, tmin_h = hist_avg("temperature_2m_max"), hist_avg("temperature_2m_min")

    def dev(a, b):
        return None if a is None or b is None else a - b

    m = {"arrays": {"tmax": tmax, "tmin": tmin, "rain": rain, "tmax_ly": tmax_ly, "tmin_ly": tmin_ly,
                    "rain_ly": rain_ly, "tmax_h": tmax_h, "tmin_h": tmin_h}}
    for key, a, a_ly, a_h in (("tmax", tmax, tmax_ly, tmax_h), ("tmin", tmin, tmin_ly, tmin_h)):
        m[f"{key}_fwd"] = smean(a, fwd)
        m[f"{key}_fwd_ly"] = smean(a_ly, fwd)
        m[f"{key}_fwd_h"] = smean(a_h, fwd)
        m[f"{key}_fwd_dev_ly"] = dev(m[f"{key}_fwd"], m[f"{key}_fwd_ly"])
        m[f"{key}_fwd_dev_h"] = dev(m[f"{key}_fwd"], m[f"{key}_fwd_h"])
    m["tmax_past"] = smean(tmax, past)
    m["tmax_past_ly"] = smean(tmax_ly, past)
    m["tmax_past_h"] = smean(tmax_h, past)
    m["tmax_past_dev_ly"] = dev(m["tmax_past"], m["tmax_past_ly"])
    m["tmax_past_dev_h"] = dev(m["tmax_past"], m["tmax_past_h"])

    # chuva na janela de previsão
    m["rain_days"] = int(np.sum(rain[fwd] > RAIN_MM))
    m["rain_days_ly"] = int(np.sum(rain_ly[fwd] > RAIN_MM)) if rain_ly is not None else None
    m["rain_mm"] = float(np.nansum(rain[fwd]))
    m["rain_mm_ly"] = float(np.nansum(rain_ly[fwd])) if rain_ly is not None else None

    # calor / frio / frente fria na previsão
    m["hot_days"] = int(np.sum(tmax[fwd] >= params["heat_c"]))
    m["cold_days"] = int(np.sum(tmin[fwd] <= params["cold_c"]))
    run = best = 0
    for v in tmax[fwd]:
        run = run + 1 if (not np.isnan(v) and v >= params["heat_c"]) else 0
        best = max(best, run)
    m["hot_run"] = best
    m["front"] = None
    for i in range(P + 1, P + F):
        drop = tmax[i - 1] - tmax[i]
        if not np.isnan(drop) and drop >= params["front_c"] and (m["front"] is None or drop > m["front"][1]):
            m["front"] = (dates[i], float(drop), float(tmax[i]))

    # fins de semana dentro do horizonte (chuva este ano x ano passado)
    wk = []
    for i in range(P, P + F):
        if dates[i].weekday() >= 5:
            wk.append({
                "date": dates[i],
                "rain": bool(rain[i] > RAIN_MM),
                "rain_ly": bool(rain_ly[i] > RAIN_MM) if rain_ly is not None and not np.isnan(rain_ly[i]) else None,
                "tmax": float(tmax[i]),
            })
    m["weekend"] = wk

    # sinal climático p/ a estação (calor/frio vs. normal): prioriza a média histórica
    ref = m["tmax_fwd_dev_h"] if m["tmax_fwd_dev_h"] is not None else m["tmax_fwd_dev_ly"]
    m["signal"] = None if ref is None else ("warm" if ref >= params["dev_c"] else "cold" if ref <= -params["dev_c"] else None)
    m["signal_ref"] = ref
    return m


# ---------------------------------------------------------------------------
# Helpers de apresentação
# ---------------------------------------------------------------------------


def md_bold_to_html(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def fmt_dev(v, unit="°C"):
    return None if v is None else f"{v:+.1f}{unit}"


def dev_tone(v):
    """Vermelho = mais quente, azul = mais frio (neutro se ~igual)."""
    if v is None or abs(v) < 0.5:
        return "neutral"
    return "bad" if v > 0 else "cold"


def kpi_card(label, value, delta=None, sub=None, tone="neutral", delta_tone="neutral"):
    parts = [f'<div class="kpi"><div class="kpi-label">{label}</div>']
    parts.append(f'<div class="kpi-value" style="color:{TONE[tone]}">{value}</div>')
    if delta:
        parts.append(f'<div class="kpi-delta" style="color:{TONE[delta_tone]}">{delta}</div>')
    if sub:
        parts.append(f'<div class="kpi-sub">{sub}</div>')
    parts.append("</div>")
    return "".join(parts)


def style_fig(fig, labels, height=440, ytitle=""):
    step = max(1, len(labels) // 12)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", dragmode=False, height=height,
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0, font=dict(size=11)),
        margin=dict(t=30, l=8, r=8, b=8), yaxis_title=ytitle,
        xaxis=dict(tickmode="array", tickvals=labels[::step], tickangle=-45),
    )
    return fig


def mark_today(fig, label):
    fig.add_shape(type="line", x0=label, x1=label, y0=0, y1=1, yref="paper",
                  line=dict(color="#f2f4f8", width=1, dash="dot"))
    fig.add_annotation(x=label, y=1, yref="paper", text="hoje", showarrow=False, yshift=10,
                       font=dict(color="#f2f4f8", size=11))


PLOT_CFG = {"displayModeBar": False}

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .block-container { padding-top: 2.2rem; }
    .kpi-grid, .day-grid { display: grid; gap: 10px; margin: 6px 0 14px 0; }
    .kpi-grid { grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }
    .day-grid { grid-template-columns: repeat(auto-fit, minmax(108px, 1fr)); }
    .kpi { background:#161b26; border:1px solid #262d3d; border-radius:10px; padding:14px 16px; }
    .kpi-label { font-size:.74rem; color:#8b93a7; text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }
    .kpi-value { font-size:1.75rem; font-weight:700; line-height:1.15; }
    .kpi-delta { font-size:.88rem; font-weight:600; margin-top:3px; }
    .kpi-sub { font-size:.78rem; color:#8b93a7; margin-top:3px; }
    .insight { background:linear-gradient(135deg,#161b26 0%,#1c2233 100%); border-left:4px solid #4f8cff;
               border-radius:10px; padding:18px 22px; margin:10px 0 16px 0; line-height:1.55; }
    .insight.alert { border-left-color:#ff6b6b; background:linear-gradient(135deg,#161b26 0%,#2a1c1f 100%); }
    .insight.positive { border-left-color:#4fd18c; background:linear-gradient(135deg,#161b26 0%,#1c2a1f 100%); }
    .insight .headline { font-size:1.05rem; font-weight:700; margin-bottom:8px; }
    .day { background:#161b26; border:1px solid #262d3d; border-radius:10px; padding:10px 6px; text-align:center; }
    .day .n { font-weight:600; color:#f2f4f8; font-size:.9rem; }
    .day .i { font-size:1.6rem; margin:4px 0; }
    .day .t { color:#f2f4f8; font-size:.98rem; } .day .t .mn { color:#8b93a7; }
    .day .r { font-size:.76rem; color:#6fb1ff; margin-top:2px; }
    .day.wk { border-color:#4f8cff; }
    @media (max-width: 640px) {
      h1 { font-size: 1.45rem !important; }
      .kpi-grid { grid-template-columns: repeat(2, 1fr); }
      .kpi { padding: 12px 12px; }
      .kpi-value { font-size: 1.3rem; }
      .block-container { padding-left: .9rem; padding-right: .9rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🌦️ Clima x Varejo")
st.sidebar.caption("Monitoramento climático para análise de fluxo e vendas")

city_names = list(CITIES.keys())
qp_city = st.query_params.get("cidade")
default_idx = city_names.index(qp_city) if qp_city in city_names else 0
city_name = st.sidebar.selectbox("Capital", city_names, index=default_idx)
st.query_params["cidade"] = city_name  # permite compartilhar o link já na cidade escolhida
city = CITIES[city_name]

horizon = st.sidebar.radio("Horizonte de previsão (dias)", [5, 7, 10], index=1, horizontal=True)

with st.sidebar.expander("⚙️ Limiares de leitura"):
    dev_c = st.slider("Desvio relevante (°C)", 0.5, 4.0, 1.0, 0.5)
    heat_c = st.slider("Dia quente: Tmax ≥ (°C)", 28, 38, 32, 1)
    cold_c = st.slider("Noite fria: Tmin ≤ (°C)", 8, 20, 14, 1)
    front_c = st.slider("Frente fria: queda de Tmax ≥ (°C)", 3, 10, 5, 1)
params = {"dev_c": dev_c, "heat_c": heat_c, "cold_c": cold_c, "front_c": front_c}

if st.sidebar.button("🔄 Atualizar dados"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Dados: [Open-Meteo](https://open-meteo.com) (gratuito, sem chave). Normal climatológica = média dos "
    f"últimos {HIST_YEARS} anos. Os limiares acima são ajustáveis."
)

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

today = datetime.now(ZoneInfo(city["tz"])).date()  # fuso da cidade (o servidor da nuvem roda em UTC)

with st.spinner(f"Buscando dados climáticos para {city_name}..."):
    try:
        data = build_data(city, today, horizon)
    except Exception as e:
        st.error(f"Não foi possível consultar a API Open-Meteo agora ({e}). Tente atualizar em instantes.")
        st.stop()

if data["current"] is None or data["current"]["temperature_2m_max"].isna().all():
    st.error("A API não retornou dados para este período. Tente atualizar em instantes.")
    st.stop()

dates, cur, P, F = data["dates"], data["current"], PAST_DAYS, horizon
m = compute_metrics(data, params)
A = m["arrays"]
n_years_hist = len(data["years"])
season = get_season(today)
labels = [d.strftime("%d/%m") for d in dates]
today_label = today.strftime("%d/%m")
ly_year = today.year - 1

# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------

st.title(f"📍 {city_name} — Clima para Varejo")
st.caption(
    f"{P} dias observados + {F} de previsão · comparado a {ly_year} e à média de {n_years_hist} anos · "
    f"Estação: **{season}** · Dados de {data['fetched_at'].strftime('%d/%m %H:%M')} (Brasília)"
)

# ---------------------------------------------------------------------------
# Previsão — cards responsivos
# ---------------------------------------------------------------------------

st.subheader(f"Previsão dos próximos {F} dias")
cards = []
for i in range(P, P + F):
    row, d = cur.iloc[i], dates[i]
    icon, desc = WEATHER_CODES.get(int(row["weathercode"]) if pd.notna(row["weathercode"]) else -1, ("❓", "N/D"))
    prob = row["precipitation_probability_max"]
    prob_txt = f"{prob:.0f}%" if pd.notna(prob) else "–"
    wk = " wk" if d.weekday() >= 5 else ""
    cards.append(
        f'<div class="day{wk}"><div class="n">{WEEKDAY_PT[d.weekday()]} {d.strftime("%d/%m")}</div>'
        f'<div class="i">{icon}</div>'
        f'<div class="t"><b>{row["temperature_2m_max"]:.0f}°</b> / <span class="mn">{row["temperature_2m_min"]:.0f}°</span></div>'
        f'<div class="r">💧 {prob_txt}</div><div class="r">{desc}</div></div>'
    )
st.markdown(f'<div class="day-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
st.caption("Fins de semana destacados em azul.")

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------


def sub_h(label, v, dev):
    return None if v is None else f"vs média {n_years_hist}a: {v:.1f}°C ({fmt_dev(dev)})"


kpis = [
    kpi_card(f"Tmax média prevista ({F}d)", f"{m['tmax_fwd']:.1f}°C" if m["tmax_fwd"] is not None else "N/D",
             m["tmax_fwd_dev_ly"] is not None and f"{fmt_dev(m['tmax_fwd_dev_ly'])} vs {ly_year}" or None,
             sub_h("", m["tmax_fwd_h"], m["tmax_fwd_dev_h"]), delta_tone=dev_tone(m["tmax_fwd_dev_ly"])),
    kpi_card(f"Realizado: Tmax média ({P}d)", f"{m['tmax_past']:.1f}°C" if m["tmax_past"] is not None else "N/D",
             m["tmax_past_dev_ly"] is not None and f"{fmt_dev(m['tmax_past_dev_ly'])} vs {ly_year}" or None,
             sub_h("", m["tmax_past_h"], m["tmax_past_dev_h"]), delta_tone=dev_tone(m["tmax_past_dev_ly"])),
    kpi_card(f"Tmin média prevista ({F}d)", f"{m['tmin_fwd']:.1f}°C" if m["tmin_fwd"] is not None else "N/D",
             m["tmin_fwd_dev_ly"] is not None and f"{fmt_dev(m['tmin_fwd_dev_ly'])} vs {ly_year}" or None,
             sub_h("", m["tmin_fwd_h"], m["tmin_fwd_dev_h"]), delta_tone=dev_tone(m["tmin_fwd_dev_ly"])),
    kpi_card(f"Dias de chuva ({F}d)", f"{m['rain_days']} / {F}",
             f"{m['rain_mm']:.0f} mm previstos",
             (f"{ly_year}: {m['rain_days_ly']} dias · {m['rain_mm_ly']:.0f} mm" if m["rain_days_ly"] is not None else None)),
    kpi_card("Calor × frio na previsão", f"{m['hot_days']} · {m['cold_days']}",
             f"dias ≥ {heat_c}°C · noites ≤ {cold_c}°C",
             (f"frente fria em {WEEKDAY_PT[m['front'][0].weekday()]} {m['front'][0].strftime('%d/%m')} "
              f"({-m['front'][1]:.0f}°C)") if m["front"] else "sem frente fria no período",
             tone="bad" if m["hot_run"] >= 3 else "neutral"),
]
if m["weekend"]:
    wk_rain = sum(1 for w in m["weekend"] if w["rain"])
    wk_known = [w for w in m["weekend"] if w["rain_ly"] is not None]
    wk_rain_ly = sum(1 for w in wk_known if w["rain_ly"])
    wk_avg = np.mean([w["tmax"] for w in m["weekend"]])
    kpis.append(kpi_card(
        "Fim de semana", f"{wk_rain} de {len(m['weekend'])} com chuva",
        f"Tmax média {wk_avg:.0f}°C",
        f"{ly_year}: {wk_rain_ly} de {len(wk_known)} com chuva" if wk_known else None,
        tone="bad" if wk_rain else "good",
    ))
else:
    kpis.append(kpi_card("Fim de semana", "fora do horizonte", None, "aumente o horizonte de previsão"))
st.subheader("Painel de Insights de Varejo")
st.markdown(f'<div class="kpi-grid">{"".join(kpis)}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Insight textual com semáforo
# ---------------------------------------------------------------------------

lines, impact = [], None  # impact: 'good' | 'bad' | None

if m["tmax_fwd_dev_ly"] is not None:
    d_ly = m["tmax_fwd_dev_ly"]
    word = "mais quentes" if d_ly > 0 else "mais frios"
    lines.append(
        f"🌡️ Os próximos **{F} dias** em **{city_name}** devem ficar, em média, **{abs(d_ly):.1f}°C {word}** "
        f"que o mesmo período de {ly_year}."
    )
if m["tmax_fwd_dev_h"] is not None:
    d_h = m["tmax_fwd_dev_h"]
    lines.append(
        f"📊 Frente à média dos últimos {n_years_hist} anos: **{abs(d_h):.1f}°C {'acima' if d_h > 0 else 'abaixo'}** do normal para essas datas."
    )
if m["tmax_past_dev_h"] is not None:
    d_p = m["tmax_past_dev_h"]
    if abs(d_p) >= dev_c:
        lines.append(
            f"📈 O **realizado** dos últimos {P} dias já veio **{abs(d_p):.1f}°C {'acima' if d_p > 0 else 'abaixo'}** da média — "
            f"o efeito sobre fluxo e vendas recentes pode já estar nos números."
        )

if m["signal"]:
    tone_key, text = SEASON_READ[season][m["signal"]]
    impact = tone_key
    tag = "Oportunidade varejo" if tone_key == "good" else "Alerta varejo"
    lines.append(f"👕 **{tag} ({season.lower()}):** {text}")

if m["front"]:
    d_f, drop, tmax_after = m["front"]
    lines.append(
        f"❄️ **Frente fria:** queda de **{drop:.0f}°C** na máxima em **{WEEKDAY_PT[d_f.weekday()]} {d_f.strftime('%d/%m')}** "
        f"(máx. {tmax_after:.0f}°C) — tende a puxar a procura por agasalhos e meia-estação nos dias seguintes."
    )
if m["hot_run"] >= 3:
    lines.append(
        f"🔥 **Onda de calor:** {m['hot_run']} dias seguidos com máxima ≥ {heat_c}°C — favorece leves/moda praia; "
        f"atenção ao fluxo de rua nas horas mais quentes."
    )

for w in m["weekend"]:
    if w["rain_ly"] is None or w["rain"] == w["rain_ly"]:
        continue
    dstr = f"{WEEKDAY_PT[w['date'].weekday()]} {w['date'].strftime('%d/%m')}"
    if w["rain"]:
        impact = impact or "bad"
        lines.append(f"☔ **Fim de semana:** chuva prevista para **{dstr}** (no ano passado foi seco) — possível **queda no fluxo de lojas físicas**.")
    else:
        lines.append(f"☀️ **Fim de semana:** **{dstr}** deve ficar seco (no ano passado choveu) — possível **alta no fluxo de lojas físicas**.")

headline = {"good": ("🟢 Cenário favorável", " positive"), "bad": ("🔴 Pontos de atenção", " alert"),
            None: ("⚪ Sem desvios relevantes", "")}[impact]
body = "<br>".join(md_bold_to_html(x) for x in lines) or "Sem desvios relevantes identificados para o período."
st.markdown(
    f'<div class="insight{headline[1]}"><div class="headline">{headline[0]}</div>{body}</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

tab_t, tab_dev, tab_r, tab_cmp = st.tabs(["🌡️ Temperatura", "📉 Desvio diário", "🌧️ Chuva", "🏙️ SP x RJ"])

with tab_t:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=A["tmax"], mode="lines", name=f"Máx. {today.year}",
                             line=dict(color="#ff6b6b", width=3)))
    fig.add_trace(go.Scatter(x=labels, y=A["tmin"], mode="lines", name=f"Mín. {today.year}",
                             line=dict(color="#4f8cff", width=3)))
    if A["tmax_ly"] is not None:
        fig.add_trace(go.Scatter(x=labels, y=A["tmax_ly"], mode="lines", name=f"Máx. {ly_year}",
                                 line=dict(color="#ff6b6b", width=1.6, dash="dot")))
        fig.add_trace(go.Scatter(x=labels, y=A["tmin_ly"], mode="lines", name=f"Mín. {ly_year}",
                                 line=dict(color="#4f8cff", width=1.6, dash="dot")))
    if A["tmax_h"] is not None:
        fig.add_trace(go.Scatter(x=labels, y=A["tmax_h"], mode="lines", name=f"Máx. média {n_years_hist}a",
                                 line=dict(color="#8b93a7", width=1.6, dash="dash")))
    mark_today(fig, today_label)
    st.plotly_chart(style_fig(fig, labels, 450, "Temperatura (°C)"), width="stretch", config=PLOT_CFG)

with tab_dev:
    fig = go.Figure()
    if A["tmax_h"] is not None:
        d_h = A["tmax"] - A["tmax_h"]
        fig.add_trace(go.Bar(x=labels, y=d_h, name=f"Tmax vs média {n_years_hist}a",
                             marker_color=["#ff6b6b" if v > 0 else "#4f8cff" for v in np.nan_to_num(d_h)]))
    if A["tmax_ly"] is not None:
        fig.add_trace(go.Scatter(x=labels, y=A["tmax"] - A["tmax_ly"], mode="lines+markers", name=f"Tmax vs {ly_year}",
                                 line=dict(color="#f2f4f8", width=1.6), marker=dict(size=5)))
    fig.add_hline(y=0, line_color="#8b93a7", line_width=1)
    mark_today(fig, today_label)
    st.plotly_chart(style_fig(fig, labels, 420, "Desvio da temperatura máxima (°C)"), width="stretch", config=PLOT_CFG)
    st.caption("Barras: desvio da máxima em relação à média histórica (vermelho = mais quente, azul = mais frio). "
               "Linha: desvio em relação ao mesmo dia do ano passado.")

with tab_r:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=A["rain"], name=str(today.year), marker_color="#4f8cff"))
    if A["rain_ly"] is not None:
        fig.add_trace(go.Bar(x=labels, y=A["rain_ly"], name=str(ly_year), marker_color="#8b93a7"))
    fig.update_layout(barmode="group")
    mark_today(fig, today_label)
    st.plotly_chart(style_fig(fig, labels, 400, "Precipitação (mm)"), width="stretch", config=PLOT_CFG)

with tab_cmp:
    other = [c for c in CITIES if c != city_name][0]
    load_cmp = st.session_state.get("cmp_loaded", False)
    if not load_cmp:
        st.write("Compara São Paulo e Rio de Janeiro lado a lado. Consulta o histórico da outra capital "
                 "(até cerca de 40 segundos no primeiro carregamento).")
        if st.button("Carregar comparação SP x RJ"):
            st.session_state["cmp_loaded"] = True
            load_cmp = True
    if load_cmp:
        with st.spinner("Comparando as capitais..."):
            try:
                results = {city_name: (data, m)}
                oc = CITIES[other]
                od = build_data(oc, datetime.now(ZoneInfo(oc["tz"])).date(), horizon)
                results[other] = (od, compute_metrics(od, params))
            except Exception:
                results = None
        if results:
            rows = []
            for name in CITIES:
                d_c, m_c = results[name]
                rows.append({
                    "Capital": name,
                    f"Tmax prev. (°C)": m_c["tmax_fwd"],
                    f"vs {ly_year} (°C)": m_c["tmax_fwd_dev_ly"],
                    "vs média (°C)": m_c["tmax_fwd_dev_h"],
                    f"Realizado {P}d vs média (°C)": m_c["tmax_past_dev_h"],
                    "Dias de chuva": m_c["rain_days"],
                    "Dias quentes": m_c["hot_days"],
                    "Noites frias": m_c["cold_days"],
                })
            st.dataframe(
                pd.DataFrame(rows), hide_index=True, width="stretch",
                column_config={
                    "Tmax prev. (°C)": st.column_config.NumberColumn(format="%.1f"),
                    f"vs {ly_year} (°C)": st.column_config.NumberColumn(format="%+.1f"),
                    "vs média (°C)": st.column_config.NumberColumn(format="%+.1f"),
                    f"Realizado {P}d vs média (°C)": st.column_config.NumberColumn(format="%+.1f"),
                },
            )
            fig = go.Figure()
            for name, color in zip(CITIES, ("#ff6b6b", "#4f8cff")):
                d_c, m_c = results[name]
                fig.add_trace(go.Scatter(x=labels, y=m_c["arrays"]["tmax"], mode="lines", name=f"Máx. {name}",
                                         line=dict(color=color, width=3)))
                fig.add_trace(go.Scatter(x=labels, y=m_c["arrays"]["tmin"], mode="lines", name=f"Mín. {name}",
                                         line=dict(color=color, width=1.6, dash="dot")))
            mark_today(fig, today_label)
            st.plotly_chart(style_fig(fig, labels, 420, "Temperatura (°C)"), width="stretch", config=PLOT_CFG)
        else:
            st.info("Não foi possível carregar a comparação agora (limite da API). Tente novamente em alguns minutos.")

# ---------------------------------------------------------------------------
# Dados brutos + download
# ---------------------------------------------------------------------------

table = pd.DataFrame({
    "Data": [d.isoformat() for d in dates],
    "Tmax atual (°C)": A["tmax"],
    "Tmin atual (°C)": A["tmin"],
    "Chuva atual (mm)": A["rain"],
    f"Tmax {ly_year} (°C)": A["tmax_ly"] if A["tmax_ly"] is not None else np.nan,
    f"Tmin {ly_year} (°C)": A["tmin_ly"] if A["tmin_ly"] is not None else np.nan,
    f"Chuva {ly_year} (mm)": A["rain_ly"] if A["rain_ly"] is not None else np.nan,
    f"Tmax média {n_years_hist}a (°C)": A["tmax_h"] if A["tmax_h"] is not None else np.nan,
}).round(1)

with st.expander("📄 Dados brutos"):
    st.dataframe(table, hide_index=True, width="stretch")
st.download_button(
    "⬇️ Baixar dados (CSV)",
    table.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"clima_varejo_{city_name.split(' (')[0].lower().replace(' ', '_')}_{today.isoformat()}.csv",
    mime="text/csv",
)

st.caption(
    "Dados: Open-Meteo.com. As leituras de varejo são hipóteses sazonais baseadas em desvios climáticos, "
    "não previsões de vendas. Painel de caráter analítico/informativo."
)
