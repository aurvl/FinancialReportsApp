# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import date, timedelta

st.set_page_config(page_title="Variation moyenne - Analyse rapide", layout="wide")

# ---------- Fonction fournie (legeres verifs et messages d'erreurs preserves) ----------
def variation_moyenne(
    df: pd.DataFrame,
    price_type: str = "close",          # "close" ou "ohlc"
    variation: str = "daily",           # "daily","weekly","monthly","annual"
    method: str = "mean",               # "mean","mean_abs","volatility"
    n_days: int | None = None,          # k jours perso; si renseigne, il prend le dessus
    aggregation: str = "auto"           # "rolling" ou "calendar" ou "auto"
) -> dict:
    if not {"Close","High","Low","Open"}.issubset(df.columns):
        raise ValueError("df doit contenir les colonnes: Open, High, Low, Close")

    df = df.copy().sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("L’index doit etre un DatetimeIndex (daily).")

    map_k = {"daily": 1, "weekly": 5, "monthly": 21, "annual": 252}
    if n_days is not None:
        k = int(n_days)
        if k < 1:
            raise ValueError("n_days doit etre >= 1")
        freq = None
        agg_mode = "rolling"
    else:
        variation = variation.lower()
        if variation not in map_k:
            raise ValueError("variation doit etre 'daily','weekly','monthly','annual'")
        k = map_k[variation]
        freq = {"daily": None, "weekly": "W", "monthly": "M", "annual": "Y"}[variation]
        if aggregation == "auto":
            agg_mode = "rolling" if variation == "daily" else "calendar"
        else:
            agg_mode = aggregation.lower()
            if agg_mode not in {"rolling","calendar"}:
                raise ValueError("aggregation doit etre 'rolling','calendar' ou 'auto'")

    last_price = float(df["Close"].iloc[-1])

    if price_type == "close":
        x = np.log(df["Close"] / df["Close"].shift(1)).dropna()

        if agg_mode == "calendar" and freq is not None and k > 1:
            closes = df["Close"].resample(freq).last().dropna()
            x_k = np.log(closes / closes.shift(1)).dropna()
        else:
            if k == 1:
                x_k = x.copy()
            else:
                x_k = x.rolling(window=k, min_periods=k).sum().dropna()

        R_k = np.exp(x_k) - 1.0

        if method == "mean":
            stat_pct = R_k.mean()
        elif method == "mean_abs":
            stat_pct = R_k.abs().mean()
        elif method == "volatility":
            sigma_x = x_k.std(ddof=1)
            stat_pct = np.exp(sigma_x) - 1.0
        else:
            raise ValueError("method doit etre 'mean','mean_abs' ou 'volatility'")

    elif price_type == "ohlc":
        v = (np.log(df["High"] / df["Low"])**2) / (4.0 * np.log(2.0))
        v = v.dropna()

        if agg_mode == "calendar" and freq is not None and k > 1:
            v_k = v.resample(freq).sum().dropna()
        else:
            if k == 1:
                v_k = v.copy()
            else:
                v_k = v.rolling(window=k, min_periods=k).sum().dropna()

        sigma_k = np.sqrt(v_k)

        if method == "mean":
            raise ValueError("method='mean' n’est pas defini pour price_type='ohlc'.")
        elif method == "mean_abs":
            stat_pct = (sigma_k * np.sqrt(2/np.pi)).mean()
        elif method == "volatility":
            stat_pct = sigma_k.mean()
        else:
            raise ValueError("method doit etre 'mean','mean_abs' ou 'volatility'")

        stat_pct = np.expm1(stat_pct)
    else:
        raise ValueError("price_type doit etre 'close' ou 'ohlc'")

    pct = float(np.round(stat_pct * 100.0, 6))
    value = float(np.round(stat_pct * last_price, 6))
    return {"pct": pct, "value": value}

# ---------- Utilitaires ----------
@st.cache_data(show_spinner=False)
def fetch_ohlc(ticker: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        return df
    
    if isinstance(df.columns, pd.MultiIndex):
        # si le niveau 0 est ('Open','High','Low','Close','Adj Close','Volume')
        # on garde ce niveau
        if set(df.columns.get_level_values(0)) >= {"Open","High","Low","Close"}:
            df.columns = df.columns.get_level_values(0)
        else:
            # fallback: prend le premier element du tuple
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # Normalise les colonnes attendues
    # st.write(df.columns)
    cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    
    df = df[cols].copy()
    df.index = pd.to_datetime(df.index)
    return df

def fmt_pct(x: float) -> str:
    return f"{x:.4f}%"

def fmt_price(x: float) -> str:
    return f"{x:,.4f}".replace(",", " ")

# ---------- UI (sans sidebar) ----------
st.markdown(
    """
    # Variation moyenne & analyse rapide
    Entrez un **ticker** Yahoo Finance, choisissez l’horizon et la methode.  
    L’app calcule la **variation moyenne** et affiche le **dernier cours** avec un graphique.
    """
)

with st.container():
    c1, c2, c3 = st.columns([2, 1.2, 1.2])
    with c1:
        ticker = st.text_input(
            "Ticker (ex: AAPL, MSFT, ^GSPC, EURUSD=X, BTC-USD, AIR.PA)",
            value="AAPL"
        ).strip()

    today = date.today()
    default_start = today - timedelta(days=365*2)
    with c2:
        start_d = st.date_input("Debut", value=default_start, max_value=today)
    with c3:
        end_d = st.date_input("Fin", value=today, max_value=today)

    c4, c5, c6, c7 = st.columns([1.4, 1.4, 1.2, 1.2])
    with c4:
        price_type = st.selectbox("Type de prix", options=["close", "ohlc"], index=0)
    with c5:
        method = st.selectbox("Methode", options=["mean", "mean_abs", "volatility"], index=0)
    with c6:
        mode_horizon = st.radio("Choix horizon", options=["variation", "n_days"], index=0, horizontal=True)
    with c7:
        aggregation = st.selectbox("Aggregation", options=["auto", "rolling", "calendar"], index=0)

    c8, c9 = st.columns([1.2, 2])
    with c8:
        if mode_horizon == "variation":
            variation = st.selectbox("Variation", options=["daily", "weekly", "monthly", "annual"], index=0)
            n_days = None
        else:
            n_days = st.number_input("n_days (>=1)", min_value=1, value=5, step=1)
            variation = "daily"  # non utilise si n_days est defini

    run = st.button("Calculer", type="primary", use_container_width=True)

# ---------- Execution ----------
if run:
    if not ticker:
        st.warning("Entrez un ticker.")
        st.stop()

    with st.spinner("Telechargement des donnees..."):
        df = fetch_ohlc(ticker, start_d, end_d)

    if df.empty or len(df) < 2:
        st.error("Aucune donnee ou serie trop courte pour ce ticker / cette periode.")
        st.stop()

    # Assure frequence daily pour le resampling calendrier si besoin
    df = df.asfreq("B", method=None)  # jours ouvrables; evite les doublons
    df["Close"] = df["Close"].fillna(method="ffill")
    for c in ["Open", "High", "Low"]:
        if c in df.columns:
            df[c] = df[c].fillna(df["Close"])

    # Dernier cours et delta jour
    last_dt = df.index.max()
    last_close = float(df.loc[last_dt, "Close"])
    prev_idx = df.index[df.index < last_dt]
    prev_close = float(df.loc[prev_idx.max(), "Close"]) if len(prev_idx) else np.nan
    day_delta = (last_close / prev_close - 1.0) * 100.0 if prev_close > 0 else np.nan

    # Calcul variation_moyenne
    try:
        df_for_calc = df[["Open","High","Low","Close"]].dropna()
        out = variation_moyenne(
            df=df_for_calc,
            price_type=price_type,
            variation=variation,
            method=method,
            n_days=n_days,
            aggregation=aggregation
        )
    except Exception as e:
        st.error(f"Erreur calcul: {e}")
        st.stop()

    # ---------- Affichage metriques ----------
    m1, m2, m3 = st.columns(3)
    m1.metric(
        label="Dernier cours (Close)",
        value=fmt_price(last_close),
        delta=f"{day_delta:+.2f}%" if not np.isnan(day_delta) else None
    )
    m2.metric(
        label="Variation moyenne (pct)",
        value=fmt_pct(out["pct"])
    )
    m3.metric(
        label="Variation moyenne (valeur absolue)",
        value=fmt_price(out["value"])
    )

    # ---------- Graphique ----------
    st.subheader("Historique des prix (Close)")
    st.line_chart(
        df[["Close"]].rename(columns={"Close": f"{ticker} - Close"})
    )

    # ---------- Details & telechargement ----------
    with st.expander("Voir les donnees / telecharger"):
        st.dataframe(df.tail(250))
        csv = df.to_csv().encode("utf-8")
        st.download_button(
            "Telecharger CSV",
            data=csv,
            file_name=f"{ticker}_ohlc.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.caption(
        "Note: les donnees proviennent de Yahoo Finance via yfinance. "
        "Les pourcentages utilisent un point decimal. Application a but informatif."
    )
else:
    st.info(
        "Entrez un ticker, ajustez les parametres, puis cliquez sur **Calculer**. "
        "Exemples: AAPL, TSLA, ^GSPC, EURUSD=X, BTC-USD, AIR.PA."
    )