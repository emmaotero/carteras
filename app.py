import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta as ta_lib
from supabase import create_client
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Tracker",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #18181c; }
  .block-container { padding-top: 1.5rem; }
  .metric-card {
    background: #18181c; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 1rem 1.25rem;
  }
  .sig-buy  { color: #34d399; font-weight: 600; }
  .sig-sell { color: #f87171; font-weight: 600; }
  .sig-neu  { color: #888898; font-weight: 600; }
  .stDataFrame { font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ── Supabase ──────────────────────────────────────────────
@st.cache_resource
def get_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

def load_clients(sb):
    r = sb.table("clients").select("*").order("created_at").execute()
    return pd.DataFrame(r.data) if r.data else pd.DataFrame()

def load_assets(sb):
    r = sb.table("assets").select("*").order("created_at").execute()
    return pd.DataFrame(r.data) if r.data else pd.DataFrame()

# ── Calculations ──────────────────────────────────────────
def patrimonio(assets_df, client_id):
    cl = assets_df[assets_df["client_id"] == client_id]
    return (cl["qty"] * cl["price_current"]).sum()

def rend_pct(assets_df, client_id):
    cl = assets_df[assets_df["client_id"] == client_id]
    if cl.empty: return 0.0
    cost = (cl["qty"] * cl["price_buy"]).sum()
    curr = (cl["qty"] * cl["price_current"]).sum()
    return ((curr - cost) / cost * 100) if cost else 0.0

def get_alerts(assets_df, clients_df, client_id):
    alerts = []
    cl_row = clients_df[clients_df["id"] == client_id]
    perfil = cl_row["perfil"].values[0] if not cl_row.empty else ""
    cl_assets = assets_df[assets_df["client_id"] == client_id]
    for _, a in cl_assets.iterrows():
        r = ((a["price_current"] - a["price_buy"]) / a["price_buy"] * 100) if a["price_buy"] else 0
        if r < -10:
            alerts.append(("🔴", f"{a['ticker']} cayó {r:.1f}% vs precio de compra"))
        elif r < -5:
            alerts.append(("🟡", f"{a['ticker']} bajó {abs(r):.1f}% vs compra"))
        if perfil == "Conservador" and a["tipo"] == "Cripto":
            alerts.append(("🟡", f"{a['ticker']} (Cripto) no es ideal para perfil Conservador"))
    return alerts

def fmt_usd(n):
    if n >= 1_000_000: return f"${n/1_000_000:.2f}M"
    if n >= 1_000:     return f"${n/1_000:.1f}K"
    return f"${n:.0f}"

# ── Indicators ────────────────────────────────────────────
AR_MAP = {
    "YPF":"YPF","GGAL":"GGAL","BMA":"BMA","PAMP":"PAM","TECO2":"TEO",
    "CEPU":"CEPU","SUPV":"SUPV","LOMA":"LOMA.BA","TXAR":"TXAR.BA",
    "ALUA":"ALUA.BA","BBAR":"BBAR.BA"
}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_indicators(ticker: str):
    yt = AR_MAP.get(ticker.upper(), ticker)
    try:
        df = yf.download(yt, period="6mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 20:
            return None, None

        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"].dropna()
        high  = df["High"].dropna()
        low   = df["Low"].dropna()
        vol   = df["Volume"].dropna()

        # Technical indicators via ta library
        df["RSI"]    = ta_lib.momentum.RSIIndicator(close, window=14).rsi()
        df["EMA50"]  = ta_lib.trend.EMAIndicator(close, window=50).ema_indicator()
        df["EMA200"] = ta_lib.trend.EMAIndicator(close, window=200).ema_indicator()

        macd_obj = ta_lib.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        df["MACD"]       = macd_obj.macd()
        df["MACD_signal"] = macd_obj.macd_signal()

        boll_obj = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
        df["BB_upper"] = boll_obj.bollinger_hband()
        df["BB_mid"]   = boll_obj.bollinger_mavg()
        df["BB_lower"] = boll_obj.bollinger_lband()

        last = df.iloc[-1]
        price = float(last["Close"])

        # Support / Resistance (20-day window)
        support    = float(low.tail(20).min())
        resistance = float(high.tail(20).max())

        # Volume ratio
        vol_recent = float(vol.tail(5).mean())
        vol_avg    = float(vol.tail(20).mean())
        vol_ratio  = round(vol_recent / vol_avg, 2) if vol_avg else None

        tech = {
            "price":      price,
            "rsi":        round(float(last["RSI"]), 1)   if pd.notna(last.get("RSI"))    else None,
            "ema50":      round(float(last["EMA50"]), 2)  if pd.notna(last.get("EMA50"))  else None,
            "ema200":     round(float(last["EMA200"]), 2) if pd.notna(last.get("EMA200")) else None,
            "macd":       round(float(last["MACD"]), 3)   if pd.notna(last.get("MACD"))   else None,
            "macd_sig":   round(float(last["MACD_signal"]), 3) if pd.notna(last.get("MACD_signal")) else None,
            "bb_upper":   round(float(last["BB_upper"]), 2) if pd.notna(last.get("BB_upper")) else None,
            "bb_lower":   round(float(last["BB_lower"]), 2) if pd.notna(last.get("BB_lower")) else None,
            "bb_mid":     round(float(last["BB_mid"]), 2)   if pd.notna(last.get("BB_mid"))   else None,
            "support":    round(support, 2),
            "resistance": round(resistance, 2),
            "vol_ratio":  vol_ratio,
        }

        # Fundamentals
        info = yf.Ticker(yt).info
        fund = {
            "pe":        info.get("trailingPE"),
            "eps":       info.get("trailingEps"),
            "div_yield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else None,
            "mkt_cap":   info.get("marketCap"),
            "beta":      info.get("beta"),
        }

        return tech, fund, df

    except Exception as e:
        return None, None, None

def signal_rsi(rsi):
    if rsi is None: return "—", "sig-neu"
    if rsi < 30:    return "↑ Compra", "sig-buy"
    if rsi > 70:    return "↓ Venta",  "sig-sell"
    return "→ Neutral", "sig-neu"

def signal_ma(price, ema50, ema200):
    if not all([price, ema50, ema200]): return "→ Neutral", "sig-neu"
    if price > ema50 > ema200: return "↑ Alcista", "sig-buy"
    if price < ema50 < ema200: return "↓ Bajista", "sig-sell"
    return "→ Neutral", "sig-neu"

def signal_macd(macd, signal):
    if macd is None: return "—", "sig-neu"
    if macd > signal: return "↑ Alcista", "sig-buy"
    return "↓ Bajista", "sig-sell"

def signal_boll(price, bb_upper, bb_lower):
    if not all([price, bb_upper, bb_lower]): return "—", "sig-neu"
    if price >= bb_upper * 0.98: return "↓ Sobrecompra", "sig-sell"
    if price <= bb_lower * 1.02: return "↑ Sobreventa",  "sig-buy"
    return "→ Medio", "sig-neu"

def signal_vol(ratio):
    if ratio is None: return "—", "sig-neu"
    if ratio > 1.3: return f"↑ Alto ×{ratio}", "sig-buy"
    if ratio < 0.7: return f"↓ Bajo ×{ratio}", "sig-sell"
    return f"→ Normal ×{ratio}", "sig-neu"

def fmt_cap(n):
    if not n: return "—"
    if n >= 1e12: return f"${n/1e12:.1f}T"
    if n >= 1e9:  return f"${n/1e9:.1f}B"
    if n >= 1e6:  return f"${n/1e6:.0f}M"
    return f"${n:.0f}"

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ◈ Portfolio Tracker")
    st.markdown("---")
    page = st.radio("Navegación", ["Dashboard", "Clientes", "Screener", "Reportes"], label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"© {datetime.now().year} Portfolio Tracker")

# ── Init ──────────────────────────────────────────────────
try:
    sb = get_supabase()
    clients_df = load_clients(sb)
    assets_df  = load_assets(sb)
except Exception as e:
    st.error(f"Error conectando a Supabase: {e}")
    st.info("Configurá SUPABASE_URL y SUPABASE_KEY en `.streamlit/secrets.toml`")
    st.stop()

# ════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("Dashboard")
    st.caption(datetime.now().strftime("%A, %d de %B de %Y"))

    col1, col2, col3, col4 = st.columns(4)
    if not clients_df.empty:
        total_aum  = sum(patrimonio(assets_df, c) for c in clients_df["id"])
        total_al   = sum(len(get_alerts(assets_df, clients_df, c)) for c in clients_df["id"])
        avg_rend   = np.mean([rend_pct(assets_df, c) for c in clients_df["id"]])
        col1.metric("AUM Total",        fmt_usd(total_aum))
        col2.metric("Clientes",         len(clients_df))
        col3.metric("Alertas activas",  total_al)
        col4.metric("Rendim. promedio", f"{avg_rend:+.1f}%")
    else:
        col1.metric("AUM Total", "—")
        col2.metric("Clientes",  "0")
        col3.metric("Alertas",   "—")
        col4.metric("Rendimiento", "—")

    st.markdown("---")

    # ── Add client form ───────────────────────────────────
    with st.expander("➕ Nuevo cliente"):
        with st.form("new_client"):
            c1, c2 = st.columns(2)
            nombre   = c1.text_input("Nombre completo")
            perfil   = c2.selectbox("Perfil de riesgo", ["Conservador", "Moderado", "Agresivo"])
            objetivo = st.text_input("Objetivo de inversión")
            c3, c4   = st.columns(2)
            capital  = c3.number_input("Capital inicial (USD)", min_value=0.0, step=1000.0)
            horizonte = c4.selectbox("Horizonte", ["Corto plazo (<2 años)", "Mediano plazo (2-5 años)", "Largo plazo (>5 años)"])
            submitted = st.form_submit_button("Guardar cliente", type="primary")
            if submitted and nombre:
                sb.table("clients").insert({
                    "name": nombre, "perfil": perfil, "objetivo": objetivo,
                    "horizonte": horizonte, "capital": capital
                }).execute()
                st.success(f"✓ Cliente {nombre} guardado")
                st.rerun()

    # ── Clients table ──────────────────────────────────────
    st.subheader("Clientes")
    if clients_df.empty:
        st.info("Sin clientes. Agregá el primero con el formulario de arriba.")
    else:
        rows = []
        for _, c in clients_df.iterrows():
            pat  = patrimonio(assets_df, c["id"])
            rend = rend_pct(assets_df, c["id"])
            als  = get_alerts(assets_df, clients_df, c["id"])
            rows.append({
                "Cliente":      c["name"],
                "Perfil":       c["perfil"],
                "Patrimonio":   fmt_usd(pat),
                "Rendimiento":  f"{rend:+.1f}%",
                "Alertas":      f"🔴 {len(als)}" if als else "✅ Ok",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# CLIENTES
# ════════════════════════════════════════════════════════════
elif page == "Clientes":
    st.title("Clientes")

    if clients_df.empty:
        st.info("Sin clientes registrados.")
    else:
        client_names = clients_df["name"].tolist()
        selected = st.selectbox("Seleccionar cliente", client_names)
        c = clients_df[clients_df["name"] == selected].iloc[0]
        cid = c["id"]

        pat  = patrimonio(assets_df, cid)
        rend = rend_pct(assets_df, cid)
        als  = get_alerts(assets_df, clients_df, cid)
        cas  = assets_df[assets_df["client_id"] == cid]

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Patrimonio actual", fmt_usd(pat))
        col2.metric("Rendimiento total", f"{rend:+.1f}%")
        col3.metric("Posiciones",        len(cas))
        col4.metric("Alertas",           len(als))

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Perfil**")
            st.markdown(f"- **Objetivo:** {c.get('objetivo','—')}")
            st.markdown(f"- **Perfil:** {c.get('perfil','—')}")
            st.markdown(f"- **Horizonte:** {c.get('horizonte','—')}")
            st.markdown(f"- **Capital inicial:** {fmt_usd(c.get('capital',0))}")

        with col_r:
            if not cas.empty:
                by_type = cas.groupby("tipo").apply(lambda x: (x["qty"]*x["price_current"]).sum())
                fig = px.pie(values=by_type.values, names=by_type.index,
                             hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(margin=dict(t=0,b=0,l=0,r=0), showlegend=True,
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        # Alertas
        if als:
            st.markdown("**⚠ Alertas**")
            for icon, msg in als:
                st.warning(f"{icon} {msg}")

        st.markdown("---")
        st.subheader("Posiciones")

        # Add asset form
        with st.expander("➕ Agregar activo"):
            with st.form(f"add_asset_{cid}"):
                a1, a2 = st.columns(2)
                ticker   = a1.text_input("Ticker", placeholder="AAPL, BTC, AL30...")
                tipo     = a2.selectbox("Tipo", ["Acción","Bono","FCI","CEDEAR","Cripto"])
                a3, a4, a5 = st.columns(3)
                qty      = a3.number_input("Cantidad", min_value=0.0, step=1.0)
                price_buy = a4.number_input("Precio compra (USD)", min_value=0.0, step=0.01, format="%.4f")
                price_cur = a5.number_input("Precio actual (USD)", min_value=0.0, step=0.01, format="%.4f")
                if st.form_submit_button("Agregar", type="primary") and ticker:
                    sb.table("assets").insert({
                        "client_id": cid, "ticker": ticker.upper(), "tipo": tipo,
                        "qty": qty, "price_buy": price_buy, "price_current": price_cur
                    }).execute()
                    st.success(f"✓ {ticker.upper()} agregado")
                    st.rerun()

        if not cas.empty:
            pos_rows = []
            for _, a in cas.iterrows():
                r = ((a["price_current"] - a["price_buy"]) / a["price_buy"] * 100) if a["price_buy"] else 0
                pos_rows.append({
                    "Ticker":    a["ticker"],
                    "Tipo":      a["tipo"],
                    "Cantidad":  a["qty"],
                    "P. Compra": f"${a['price_buy']:.4f}",
                    "P. Actual": f"${a['price_current']:.4f}",
                    "Valor":     fmt_usd(a["qty"] * a["price_current"]),
                    "Rend.":     f"{r:+.1f}%",
                })
            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

            # Delete asset
            with st.expander("🗑 Eliminar activo"):
                tickers = cas["ticker"].tolist()
                to_del = st.selectbox("Seleccionar activo a eliminar", tickers)
                if st.button("Eliminar", type="secondary"):
                    aid = cas[cas["ticker"] == to_del]["id"].values[0]
                    sb.table("assets").delete().eq("id", aid).execute()
                    st.success(f"✓ {to_del} eliminado")
                    st.rerun()
        else:
            st.info("Sin activos. Agregá uno con el formulario de arriba.")

        # Delete client
        with st.expander("⚠ Zona de peligro"):
            st.warning("Esto eliminará el cliente y todos sus activos.")
            if st.button("🗑 Eliminar cliente", type="secondary"):
                sb.table("assets").delete().eq("client_id", cid).execute()
                sb.table("clients").delete().eq("id", cid).execute()
                st.success("Cliente eliminado")
                st.rerun()

# ════════════════════════════════════════════════════════════
# SCREENER
# ════════════════════════════════════════════════════════════
elif page == "Screener":
    st.title("Screener de posiciones")
    st.caption("Indicadores técnicos y fundamentales calculados con datos reales de Yahoo Finance")

    if assets_df.empty:
        st.info("Sin posiciones registradas. Agregá activos desde la sección Clientes.")
    else:
        # Build unique tickers with owners
        tickers_info = {}
        for _, a in assets_df.iterrows():
            t = a["ticker"]
            if t not in tickers_info:
                tickers_info[t] = {"tipo": a["tipo"], "owners": [], "price_current": a["price_current"]}
            c = clients_df[clients_df["id"] == a["client_id"]]
            if not c.empty:
                tickers_info[t]["owners"].append(c.iloc[0]["name"])

        # Filters
        col_f1, col_f2, col_f3 = st.columns([2,1,1])
        filter_tipo = col_f1.multiselect("Filtrar por tipo", ["Acción","Bono","FCI","CEDEAR","Cripto"])
        filter_sig  = col_f2.selectbox("Señal", ["Todas", "Compra", "Venta", "Neutral"])
        if col_f3.button("↻ Actualizar datos", type="primary"):
            fetch_indicators.clear()
            st.rerun()

        st.markdown("---")

        results = []
        tickers_list = list(tickers_info.keys())
        if filter_tipo:
            tickers_list = [t for t in tickers_list if tickers_info[t]["tipo"] in filter_tipo]

        progress = st.progress(0, text="Descargando datos...")
        for i, ticker in enumerate(tickers_list):
            progress.progress((i+1)/len(tickers_list), text=f"Cargando {ticker}...")
            info = tickers_info[ticker]
            result = fetch_indicators(ticker)
            tech, fund = (result[0], result[1]) if result and result[0] else (None, None)
            price = info["price_current"]

            if tech:
                sig_rsi_lbl,  sig_rsi_cls  = signal_rsi(tech["rsi"])
                sig_ma_lbl,   sig_ma_cls   = signal_ma(price, tech["ema50"], tech["ema200"])
                sig_macd_lbl, sig_macd_cls = signal_macd(tech["macd"], tech["macd_sig"])
                sig_bb_lbl,   sig_bb_cls   = signal_boll(price, tech["bb_upper"], tech["bb_lower"])
                sig_vol_lbl,  sig_vol_cls  = signal_vol(tech["vol_ratio"])

                # Overall signal (majority vote)
                sigs = [sig_rsi_cls, sig_ma_cls, sig_macd_cls, sig_bb_cls]
                buys  = sigs.count("sig-buy")
                sells = sigs.count("sig-sell")
                overall = "↑ Compra" if buys >= 3 else "↓ Venta" if sells >= 3 else "→ Neutral"

                row = {
                    "Ticker":      ticker,
                    "Tipo":        info["tipo"],
                    "Clientes":    ", ".join(set(info["owners"])),
                    "Precio":      f"${price:.2f}",
                    "Señal":       overall,
                    "RSI":         f"{tech['rsi']} — {sig_rsi_lbl}" if tech["rsi"] else "—",
                    "EMA 50/200":  f"{tech['ema50']}/{tech['ema200']} — {sig_ma_lbl}",
                    "MACD":        f"{tech['macd']} — {sig_macd_lbl}" if tech["macd"] else "—",
                    "Bollinger":   sig_bb_lbl,
                    "Volumen":     sig_vol_lbl,
                    "Soporte":     f"${tech['support']}",
                    "Resistencia": f"${tech['resistance']}",
                    "P/E":         f"{fund['pe']:.1f}" if fund and fund.get("pe") else "—",
                    "EPS":         f"${fund['eps']:.2f}" if fund and fund.get("eps") else "—",
                    "Div. Yield":  f"{fund['div_yield']}%" if fund and fund.get("div_yield") else "—",
                    "Mkt Cap":     fmt_cap(fund["mkt_cap"]) if fund else "—",
                    "Beta":        f"{fund['beta']:.2f}" if fund and fund.get("beta") else "—",
                }

                if filter_sig == "Todas" or filter_sig in overall:
                    results.append(row)
            else:
                results.append({
                    "Ticker": ticker, "Tipo": info["tipo"],
                    "Clientes": ", ".join(set(info["owners"])),
                    "Precio": f"${price:.2f}",
                    "Señal": "⚠ Sin datos",
                    **{k: "—" for k in ["RSI","EMA 50/200","MACD","Bollinger","Volumen","Soporte","Resistencia","P/E","EPS","Div. Yield","Mkt Cap","Beta"]}
                })

        progress.empty()

        if results:
            df_res = pd.DataFrame(results)

            # Color señal column
            def color_signal(val):
                if "Compra" in str(val): return "color: #34d399; font-weight: 600"
                if "Venta"  in str(val): return "color: #f87171; font-weight: 600"
                if "Sin datos" in str(val): return "color: #fbbf24"
                return "color: #888898"

            styled = df_res.style.applymap(color_signal, subset=["Señal"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Detail chart for selected ticker
            st.markdown("---")
            st.subheader("Gráfico de precio + indicadores")
            sel_ticker = st.selectbox("Ver gráfico de:", tickers_list)
            result = fetch_indicators(sel_ticker)
            if result and result[2] is not None:
                _, _, hist_df = result
                if isinstance(hist_df.columns, pd.MultiIndex):
                    hist_df.columns = hist_df.columns.get_level_values(0)

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=hist_df.index, open=hist_df["Open"], high=hist_df["High"],
                    low=hist_df["Low"], close=hist_df["Close"], name="Precio",
                    increasing_line_color="#34d399", decreasing_line_color="#f87171"
                ))
                if "EMA50" in hist_df.columns:
                    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["EMA50"],
                        name="EMA 50", line=dict(color="#7c6fff", width=1.5)))
                if "EMA200" in hist_df.columns:
                    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["EMA200"],
                        name="EMA 200", line=dict(color="#fbbf24", width=1.5)))
                if "BB_upper" in hist_df.columns:
                    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["BB_upper"],
                        name="BB Superior", line=dict(color="#888898", width=1, dash="dash")))
                    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["BB_lower"],
                        name="BB Inferior", line=dict(color="#888898", width=1, dash="dash"),
                        fill="tonexty", fillcolor="rgba(136,136,152,0.05)"))
                fig.update_layout(
                    xaxis_rangeslider_visible=False,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f0f0f2"), height=420,
                    legend=dict(orientation="h", y=1.02),
                    margin=dict(t=20, b=10)
                )
                fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
                fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
                st.plotly_chart(fig, use_container_width=True)

                # RSI chart
                if "RSI" in hist_df.columns:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=hist_df.index, y=hist_df["RSI"],
                        name="RSI", line=dict(color="#a78bfa", width=2)))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#f87171", annotation_text="Sobrecompra 70")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#34d399", annotation_text="Sobreventa 30")
                    fig_rsi.update_layout(
                        height=160, margin=dict(t=10,b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#f0f0f2"), showlegend=False
                    )
                    fig_rsi.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
                    fig_rsi.update_yaxes(gridcolor="rgba(255,255,255,0.05)", range=[0,100])
                    st.plotly_chart(fig_rsi, use_container_width=True)
            else:
                st.warning(f"No se pudieron obtener datos históricos para {sel_ticker}")

# ════════════════════════════════════════════════════════════
# REPORTES
# ════════════════════════════════════════════════════════════
elif page == "Reportes":
    st.title("Reportes")

    if clients_df.empty:
        st.info("Sin clientes registrados.")
    else:
        selected = st.selectbox("Seleccionar cliente", clients_df["name"].tolist())
        c = clients_df[clients_df["name"] == selected].iloc[0]
        cid = c["id"]
        cas = assets_df[assets_df["client_id"] == cid]
        pat  = patrimonio(assets_df, cid)
        rend = rend_pct(assets_df, cid)
        als  = get_alerts(assets_df, clients_df, cid)
        today = datetime.now().strftime("%d de %B de %Y")

        if st.button("Generar reporte", type="primary"):
            st.markdown("---")
            st.markdown(f"## Informe de Cartera — {c['name']}")
            st.caption(f"Generado el {today}")
            st.markdown("---")

            st.markdown("### Perfil y objetivos")
            col1, col2 = st.columns(2)
            col1.markdown(f"**Perfil de riesgo:** {c.get('perfil','—')}")
            col1.markdown(f"**Objetivo:** {c.get('objetivo','—')}")
            col2.markdown(f"**Horizonte:** {c.get('horizonte','—')}")
            col2.markdown(f"**Capital inicial:** {fmt_usd(c.get('capital',0))}")

            st.markdown("### Resumen patrimonial")
            m1, m2, m3 = st.columns(3)
            m1.metric("Patrimonio actual", fmt_usd(pat))
            m2.metric("Rendimiento total", f"{rend:+.1f}%")
            m3.metric("Posiciones",        len(cas))

            if not cas.empty:
                st.markdown("### Composición")
                by_type = cas.groupby("tipo").apply(lambda x: (x["qty"]*x["price_current"]).sum())
                total = by_type.sum()
                for tipo, val in by_type.items():
                    pct = val/total*100
                    st.markdown(f"**{tipo}:** {fmt_usd(val)} ({pct:.0f}%)")
                    st.progress(pct/100)

                st.markdown("### Posiciones detalladas")
                pos_rows = []
                for _, a in cas.iterrows():
                    r = ((a["price_current"]-a["price_buy"])/a["price_buy"]*100) if a["price_buy"] else 0
                    pos_rows.append({
                        "Ticker": a["ticker"], "Tipo": a["tipo"],
                        "Cantidad": a["qty"],
                        "P. Compra": f"${a['price_buy']:.4f}",
                        "P. Actual": f"${a['price_current']:.4f}",
                        "Valor": fmt_usd(a["qty"]*a["price_current"]),
                        "Rendimiento": f"{r:+.1f}%"
                    })
                st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

            if als:
                st.markdown("### ⚠ Alertas")
                for icon, msg in als:
                    st.warning(f"{icon} {msg}")

            st.markdown("---")
            st.caption("Reporte generado automáticamente por Portfolio Tracker")
