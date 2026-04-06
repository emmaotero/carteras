import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta as ta_lib
import requests
from supabase import create_client
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Portfolio Tracker", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #18181c; }
  .block-container { padding-top: 1.5rem; }
  .stDataFrame { font-size: 13px; }
</style>
""", unsafe_allow_html=True)

DEFAULT_CURRENCY = {"Acción":"ARS","CEDEAR":"ARS","Bono":"USD","FCI":"ARS","Cripto":"USD"}

@st.cache_resource
def get_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

def load_clients(sb):
    r = sb.table("clients").select("*").order("created_at").execute()
    return pd.DataFrame(r.data) if r.data else pd.DataFrame()

def load_assets(sb):
    r = sb.table("assets").select("*").order("created_at").execute()
    return pd.DataFrame(r.data) if r.data else pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def get_mep():
    sources = [
        ("https://api.bluelytics.com.ar/v2/latest", lambda d: d.get("mep",{}).get("value_sell")),
        ("https://dolarapi.com/v1/dolares/bolsa",    lambda d: d.get("venta")),
    ]
    for url, extractor in sources:
        try:
            r = requests.get(url, timeout=5)
            val = extractor(r.json())
            if val: return float(str(val).replace(",","."))
        except Exception:
            continue
    return None

def to_usd(ars, mep): return ars / mep if mep else 0.0
def fmt_ars(n): return f"${n:,.0f}" if n < 1_000_000 else f"${n/1_000_000:.2f}M"
def fmt_usd(n):
    if n >= 1_000_000: return f"USD {n/1_000_000:.2f}M"
    if n >= 1_000:     return f"USD {n/1_000:.1f}K"
    return f"USD {n:,.0f}"
def fmt_cap(n):
    if not n: return "—"
    if n >= 1e12: return f"USD {n/1e12:.1f}T"
    if n >= 1e9:  return f"USD {n/1e9:.1f}B"
    return f"USD {n/1e6:.0f}M"

def asset_val_usd(row, mep):
    v = row["qty"] * row["price_current"]
    return to_usd(v, mep) if row.get("currency","ARS") == "ARS" else v

def asset_cost_usd(row, mep):
    v = row["qty"] * row["price_buy"]
    return to_usd(v, mep) if row.get("currency","ARS") == "ARS" else v

def patrimonio_usd(adf, cid, mep):
    cl = adf[adf["client_id"]==cid]
    return sum(asset_val_usd(r,mep) for _,r in cl.iterrows()) if not cl.empty else 0.0

def rend_pct(adf, cid, mep):
    cl = adf[adf["client_id"]==cid]
    if cl.empty: return 0.0
    cost = sum(asset_cost_usd(r,mep) for _,r in cl.iterrows())
    curr = sum(asset_val_usd(r,mep) for _,r in cl.iterrows())
    return (curr-cost)/cost*100 if cost else 0.0

def get_alerts(adf, cdf, cid):
    alerts = []
    row = cdf[cdf["id"]==cid]
    perfil = row["perfil"].values[0] if not row.empty else ""
    for _,a in adf[adf["client_id"]==cid].iterrows():
        if a["price_buy"]:
            r = (a["price_current"]-a["price_buy"])/a["price_buy"]*100
            if r < -10: alerts.append(("🔴", f"{a['ticker']} cayó {r:.1f}% vs compra"))
            elif r < -5: alerts.append(("🟡", f"{a['ticker']} bajó {abs(r):.1f}% vs compra"))
        if perfil=="Conservador" and a["tipo"]=="Cripto":
            alerts.append(("🟡", f"{a['ticker']} (Cripto) no recomendado para perfil Conservador"))
    return alerts

def auto_update_prices(sb, assets_df):
    """Trae el último precio de Yahoo Finance y actualiza Supabase. Devuelve (ok, fail)."""
    unique_tickers = assets_df.drop_duplicates(subset="ticker")[["id","ticker","currency"]]
    ok, fail = [], []
    for _, row in unique_tickers.iterrows():
        ticker = row["ticker"]
        try:
            data = yf.download(ticker, period="2d", interval="1d", progress=False, auto_adjust=True)
            if data.empty: fail.append(ticker); continue
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            price = float(data["Close"].dropna().iloc[-1])
            ids = assets_df[assets_df["ticker"]==ticker]["id"].tolist()
            for aid in ids:
                sb.table("assets").update({"price_current": round(price,4)}).eq("id", str(aid)).execute()
            ok.append(f"{ticker} → {price:,.2f}")
        except Exception:
            fail.append(ticker)
    return ok, fail

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_indicators(ticker: str):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 20: return None, None, None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna(); high = df["High"].dropna()
        low   = df["Low"].dropna();   vol  = df["Volume"].dropna()
        df["RSI"]    = ta_lib.momentum.RSIIndicator(close, window=14).rsi()
        df["EMA50"]  = ta_lib.trend.EMAIndicator(close, window=50).ema_indicator()
        df["EMA200"] = ta_lib.trend.EMAIndicator(close, window=200).ema_indicator()
        m = ta_lib.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = m.macd(); df["MACD_sig"] = m.macd_signal()
        b = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
        df["BB_upper"]=b.bollinger_hband(); df["BB_mid"]=b.bollinger_mavg(); df["BB_lower"]=b.bollinger_lband()
        last = df.iloc[-1]; price = float(last["Close"])
        vol_ratio = round(float(vol.tail(5).mean())/float(vol.tail(20).mean()),2) if float(vol.tail(20).mean()) else None
        tech = {
            "price": price,
            "rsi":    round(float(last["RSI"]),1)   if pd.notna(last.get("RSI"))    else None,
            "ema50":  round(float(last["EMA50"]),2)  if pd.notna(last.get("EMA50"))  else None,
            "ema200": round(float(last["EMA200"]),2) if pd.notna(last.get("EMA200")) else None,
            "macd":   round(float(last["MACD"]),3)   if pd.notna(last.get("MACD"))   else None,
            "macd_sig": round(float(last["MACD_sig"]),3) if pd.notna(last.get("MACD_sig")) else None,
            "bb_upper": round(float(last["BB_upper"]),2) if pd.notna(last.get("BB_upper")) else None,
            "bb_lower": round(float(last["BB_lower"]),2) if pd.notna(last.get("BB_lower")) else None,
            "support": round(float(low.tail(20).min()),2),
            "resistance": round(float(high.tail(20).max()),2),
            "vol_ratio": vol_ratio,
        }
        info = yf.Ticker(ticker).info
        fund = {
            "pe": info.get("trailingPE"), "eps": info.get("trailingEps"),
            "div_yield": round(info.get("dividendYield",0)*100,2) if info.get("dividendYield") else None,
            "mkt_cap": info.get("marketCap"), "beta": info.get("beta"),
        }
        return tech, fund, df
    except Exception:
        return None, None, None

def sig(rsi=None, price=None, ema50=None, ema200=None, macd=None, macd_s=None, bb_u=None, bb_l=None, vol_r=None):
    results = {}
    if rsi is not None:
        if rsi<30: results["RSI"] = (f"↑ Compra ({rsi})","buy")
        elif rsi>70: results["RSI"] = (f"↓ Venta ({rsi})","sell")
        else: results["RSI"] = (f"→ Neutral ({rsi})","neu")
    else: results["RSI"] = ("—","neu")
    if all([price,ema50,ema200]):
        if price>ema50>ema200: results["EMA"] = ("↑ Alcista","buy")
        elif price<ema50<ema200: results["EMA"] = ("↓ Bajista","sell")
        else: results["EMA"] = ("→ Neutral","neu")
    else: results["EMA"] = ("—","neu")
    if macd is not None and macd_s is not None:
        results["MACD"] = (f"↑ Alcista ({macd:.3f})","buy") if macd>macd_s else (f"↓ Bajista ({macd:.3f})","sell")
    else: results["MACD"] = ("—","neu")
    if all([price,bb_u,bb_l]):
        if price>=bb_u*0.98: results["Bollinger"] = ("↓ Sobrecompra","sell")
        elif price<=bb_l*1.02: results["Bollinger"] = ("↑ Sobreventa","buy")
        else: results["Bollinger"] = ("→ Medio","neu")
    else: results["Bollinger"] = ("—","neu")
    if vol_r:
        if vol_r>1.3: results["Volumen"] = (f"↑ Alto ×{vol_r}","buy")
        elif vol_r<0.7: results["Volumen"] = (f"↓ Bajo ×{vol_r}","sell")
        else: results["Volumen"] = (f"→ Normal ×{vol_r}","neu")
    else: results["Volumen"] = ("—","neu")
    return results

def color_signal(val):
    v = str(val)
    if any(x in v for x in ["Compra","Alcista","Alto","Sobreventa"]): return "color:#34d399;font-weight:600"
    if any(x in v for x in ["Venta","Bajista","Bajo","Sobrecompra"]): return "color:#f87171;font-weight:600"
    if "Sin datos" in v: return "color:#fbbf24"
    return "color:#888898"

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ◈ Portfolio Tracker")
    st.markdown("---")
    page = st.radio("Nav", ["Dashboard","Clientes","Screener","Reportes"], label_visibility="collapsed")
    st.markdown("---")
    mep = get_mep()
    if mep:
        st.metric("Dólar MEP", f"${mep:,.0f}", help="Actualizado cada 30 min")
    else:
        mep = st.number_input("MEP manual (no se obtuvo auto)", min_value=1.0, value=1200.0, step=10.0)
        st.caption("⚠ Cotización manual")
    st.caption(f"© {datetime.now().year} Portfolio Tracker")

# ── Init ──────────────────────────────────────────────────
try:
    sb = get_supabase()
    clients_df = load_clients(sb)
    assets_df  = load_assets(sb)
    if not assets_df.empty and "currency" not in assets_df.columns:
        assets_df["currency"] = "ARS"
except Exception as e:
    st.error(f"Error Supabase: {e}"); st.stop()

# ════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════
if page == "Dashboard":
    st.title("Dashboard")
    st.caption(datetime.now().strftime("%A, %d de %B de %Y"))
    c1,c2,c3,c4 = st.columns(4)
    if not clients_df.empty:
        aum  = sum(patrimonio_usd(assets_df,c,mep) for c in clients_df["id"])
        al   = sum(len(get_alerts(assets_df,clients_df,c)) for c in clients_df["id"])
        rend = np.mean([rend_pct(assets_df,c,mep) for c in clients_df["id"]])
        c1.metric("AUM Total",fmt_usd(aum)); c2.metric("Clientes",len(clients_df))
        c3.metric("Alertas",al);             c4.metric("Rendim. prom.",f"{rend:+.1f}%")
    else:
        c1.metric("AUM","—"); c2.metric("Clientes","0"); c3.metric("Alertas","—"); c4.metric("Rend.","—")
    st.markdown("---")

    # Auto-update prices button
    col_upd, col_info = st.columns([1,3])
    if col_upd.button("↻ Actualizar precios", type="primary", help="Trae el último precio de Yahoo Finance para todos los activos"):
        if not assets_df.empty:
            with st.spinner("Actualizando precios..."):
                ok, fail = auto_update_prices(sb, assets_df)
            if ok:
                st.success(f"✓ Actualizados: {', '.join(ok)}")
            if fail:
                st.warning(f"⚠ Sin datos (actualizar manual): {', '.join(fail)}")
            st.rerun()
        else:
            st.info("Sin activos para actualizar.")

    with st.expander("➕ Nuevo cliente"):
        with st.form("nc"):
            x1,x2 = st.columns(2)
            nombre = x1.text_input("Nombre"); perfil = x2.selectbox("Perfil",["Conservador","Moderado","Agresivo"])
            objetivo = st.text_input("Objetivo")
            x3,x4 = st.columns(2)
            capital = x3.number_input("Capital inicial (ARS)",min_value=0.0,step=100000.0)
            horizonte = x4.selectbox("Horizonte",["Corto (<2 años)","Mediano (2-5 años)","Largo (>5 años)"])
            if st.form_submit_button("Guardar",type="primary") and nombre:
                sb.table("clients").insert({"name":nombre,"perfil":perfil,"objetivo":objetivo,"horizonte":horizonte,"capital":capital}).execute()
                st.success("✓ Guardado"); st.rerun()
    st.subheader("Clientes")
    if clients_df.empty:
        st.info("Sin clientes aún.")
    else:
        rows = []
        for _,c in clients_df.iterrows():
            rows.append({"Cliente":c["name"],"Perfil":c["perfil"],
                "Patrimonio":fmt_usd(patrimonio_usd(assets_df,c["id"],mep)),
                "Rendimiento":f"{rend_pct(assets_df,c['id'],mep):+.1f}%",
                "Alertas":f"🔴 {len(get_alerts(assets_df,clients_df,c['id']))}" if get_alerts(assets_df,clients_df,c["id"]) else "✅ Ok"})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

# ════════════════════════════════════
# CLIENTES
# ════════════════════════════════════
elif page == "Clientes":
    st.title("Clientes")
    if clients_df.empty:
        st.info("Sin clientes.")
    else:
        selected = st.selectbox("Cliente", clients_df["name"].tolist())
        c = clients_df[clients_df["name"]==selected].iloc[0]; cid = c["id"]
        cas = assets_df[assets_df["client_id"]==cid].copy()
        if "currency" not in cas.columns: cas["currency"] = "ARS"
        pat = patrimonio_usd(assets_df,cid,mep)
        rend = rend_pct(assets_df,cid,mep)
        als  = get_alerts(assets_df,clients_df,cid)

        # Load extended data
        cf_r  = sb.table("cash_flow").select("*").eq("client_id",cid).order("created_at").execute()
        ph_r  = sb.table("profile_history").select("*").eq("client_id",cid).order("fecha",desc=True).execute()
        cf_df = pd.DataFrame(cf_r.data) if cf_r.data else pd.DataFrame()
        ph_df = pd.DataFrame(ph_r.data) if ph_r.data else pd.DataFrame()

        # ── Summary metrics ──────────────────────────────
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Patrimonio (USD)", fmt_usd(pat))
        m2.metric("Rendimiento",      f"{rend:+.1f}%")
        m3.metric("Posiciones",       len(cas))
        m4.metric("Alertas",          len(als))
        if als:
            for icon,msg in als: st.warning(f"{icon} {msg}")
        st.markdown("---")

        # ── Tabs ─────────────────────────────────────────
        tab_perf, tab_pos, tab_cf, tab_hist = st.tabs(["👤 Perfil","📊 Posiciones","💰 Flujo de caja","📋 Historial"])

        # ════════════════════════════════════
        # TAB: PERFIL
        # ════════════════════════════════════
        with tab_perf:
            col_l, col_r = st.columns(2)

            with col_l:
                # Donut chart
                if not cas.empty:
                    bt = {}
                    for _,a in cas.iterrows(): bt[a["tipo"]] = bt.get(a["tipo"],0)+asset_val_usd(a,mep)
                    fig = px.pie(values=list(bt.values()),names=list(bt.keys()),hole=0.4,
                                 color_discrete_sequence=px.colors.qualitative.Set3)
                    fig.update_layout(margin=dict(t=0,b=0,l=0,r=0),
                                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig,use_container_width=True)

            with col_r:
                cap_ars = c.get("capital",0) or 0
                st.markdown(f"**Email:** {c.get('email','—') or '—'}")
                st.markdown(f"**Teléfono:** {c.get('telefono','—') or '—'}")
                st.markdown(f"**Ocupación:** {c.get('ocupacion','—') or '—'}")
                st.markdown(f"**Fecha nac.:** {c.get('fecha_nac','—') or '—'}")
                st.markdown(f"**Estado:** {c.get('estado','Activo') or 'Activo'}")
                st.markdown(f"**Última reunión:** {c.get('ultima_reunion','—') or '—'}")

            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Inversión**")
                cap_ars = c.get("capital",0) or 0
                st.markdown(f"- **Perfil de riesgo:** {c.get('perfil','—')}")
                st.markdown(f"- **Horizonte:** {c.get('horizonte','—') or '—'}")
                st.markdown(f"- **Objetivo:** {c.get('objetivo','—') or '—'}")
                st.markdown(f"- **Monto objetivo:** {fmt_ars(c.get('monto_objetivo',0) or 0)}")
                st.markdown(f"- **Fecha objetivo:** {c.get('fecha_objetivo','—') or '—'}")
                st.markdown(f"- **Capital inicial:** {fmt_ars(cap_ars)} / {fmt_usd(to_usd(cap_ars,mep))}")
            with col_b:
                st.markdown("**Perfil inversor**")
                st.markdown(f"- **Tolerancia volatilidad:** {c.get('tolerancia_vol','—') or '—'}")
                st.markdown(f"- **Experiencia:** {c.get('experiencia','—') or '—'}")
                st.markdown(f"- **Tipo de capital:** {c.get('tipo_capital','—') or '—'}")

            if c.get("notas"):
                st.markdown("**Notas**")
                st.info(c["notas"])

            # ── Editar perfil ─────────────────────────────
            with st.expander("✏ Editar perfil completo"):
                with st.form(f"edit_profile_{cid}"):
                    st.markdown("**Datos de contacto**")
                    e1,e2,e3 = st.columns(3)
                    new_email   = e1.text_input("Email",    value=c.get("email","") or "")
                    new_tel     = e2.text_input("Teléfono", value=c.get("telefono","") or "")
                    new_ocup    = e3.text_input("Ocupación",value=c.get("ocupacion","") or "")
                    e4,e5 = st.columns(2)
                    raw_fn = c.get("fecha_nac")
                    new_fnac = e4.date_input("Fecha nacimiento",
                        value=datetime.strptime(str(raw_fn),"%Y-%m-%d").date() if raw_fn else None,
                        format="DD/MM/YYYY")
                    estado_opts = ["Activo","Prospecto","Pausado","Dado de baja"]
                    cur_estado  = c.get("estado","Activo") or "Activo"
                    new_estado  = e5.selectbox("Estado", estado_opts,
                        index=estado_opts.index(cur_estado) if cur_estado in estado_opts else 0)

                    raw_ur = c.get("ultima_reunion")
                    new_reunion = st.date_input("Última reunión",
                        value=datetime.strptime(str(raw_ur),"%Y-%m-%d").date() if raw_ur else None,
                        format="DD/MM/YYYY")

                    st.markdown("**Inversión**")
                    f1,f2 = st.columns(2)
                    perfil_opts = ["Conservador","Moderado","Agresivo"]
                    cur_perf    = c.get("perfil","Moderado") or "Moderado"
                    new_perfil  = f1.selectbox("Perfil de riesgo", perfil_opts,
                        index=perfil_opts.index(cur_perf) if cur_perf in perfil_opts else 1)
                    horiz_opts  = ["Corto (<2 años)","Mediano (2-5 años)","Largo (>5 años)"]
                    cur_horiz   = c.get("horizonte","Mediano (2-5 años)") or "Mediano (2-5 años)"
                    new_horiz   = f2.selectbox("Horizonte", horiz_opts,
                        index=horiz_opts.index(cur_horiz) if cur_horiz in horiz_opts else 1)
                    new_obj     = st.text_input("Objetivo", value=c.get("objetivo","") or "")
                    g1,g2,g3 = st.columns(3)
                    new_cap     = g1.number_input("Capital inicial (ARS)",
                        value=float(c.get("capital",0) or 0), min_value=0.0, step=100000.0)
                    new_monto_obj = g2.number_input("Monto objetivo (ARS)",
                        value=float(c.get("monto_objetivo",0) or 0), min_value=0.0, step=100000.0)
                    raw_fo = c.get("fecha_objetivo")
                    new_fobj = g3.date_input("Fecha objetivo",
                        value=datetime.strptime(str(raw_fo),"%Y-%m-%d").date() if raw_fo else None,
                        format="DD/MM/YYYY")

                    st.markdown("**Perfil inversor**")
                    h1,h2,h3 = st.columns(3)
                    tol_opts = ["Alta","Media","Baja"]
                    cur_tol  = c.get("tolerancia_vol","Media") or "Media"
                    new_tol  = h1.selectbox("Tolerancia volatilidad", tol_opts,
                        index=tol_opts.index(cur_tol) if cur_tol in tol_opts else 1)
                    exp_opts = ["Sin experiencia","Básica","Intermedia","Avanzada"]
                    cur_exp  = c.get("experiencia","Básica") or "Básica"
                    new_exp  = h2.selectbox("Experiencia", exp_opts,
                        index=exp_opts.index(cur_exp) if cur_exp in exp_opts else 1)
                    cap_opts = ["Excedente","Único ahorro","Mixto"]
                    cur_cap_t = c.get("tipo_capital","Excedente") or "Excedente"
                    new_cap_t = h3.selectbox("Tipo de capital", cap_opts,
                        index=cap_opts.index(cur_cap_t) if cur_cap_t in cap_opts else 0)

                    new_notas   = st.text_area("Notas libres", value=c.get("notas","") or "")
                    nota_cambio = st.text_input("Nota del cambio (para historial)", placeholder="Ej: Revisión anual de perfil")

                    if st.form_submit_button("Guardar cambios", type="primary"):
                        updates = {
                            "email": new_email, "telefono": new_tel, "ocupacion": new_ocup,
                            "fecha_nac": str(new_fnac) if new_fnac else None,
                            "estado": new_estado,
                            "ultima_reunion": str(new_reunion) if new_reunion else None,
                            "perfil": new_perfil, "horizonte": new_horiz,
                            "objetivo": new_obj, "capital": new_cap,
                            "monto_objetivo": new_monto_obj,
                            "fecha_objetivo": str(new_fobj) if new_fobj else None,
                            "tolerancia_vol": new_tol, "experiencia": new_exp,
                            "tipo_capital": new_cap_t, "notas": new_notas,
                        }
                        # Detect changes and log history
                        campos_log = {"perfil":"Perfil","horizonte":"Horizonte","objetivo":"Objetivo",
                                      "estado":"Estado","tolerancia_vol":"Tolerancia vol.",
                                      "experiencia":"Experiencia","tipo_capital":"Tipo capital"}
                        for campo, label in campos_log.items():
                            antes = str(c.get(campo,"") or "")
                            nuevo = str(updates.get(campo,"") or "")
                            if antes != nuevo:
                                sb.table("profile_history").insert({
                                    "client_id": cid, "campo": label,
                                    "valor_antes": antes, "valor_nuevo": nuevo,
                                    "nota": nota_cambio
                                }).execute()
                        sb.table("clients").update(updates).eq("id",cid).execute()
                        st.success("✓ Perfil actualizado")
                        st.rerun()

        # ════════════════════════════════════
        # TAB: POSICIONES
        # ════════════════════════════════════
        with tab_pos:
            ca1, ca2 = st.columns([1,3])
            if ca1.button("↻ Actualizar precios", type="primary", key=f"upd_auto_{cid}"):
                with st.spinner("Actualizando..."):
                    ok, fail = auto_update_prices(sb, cas)
                if ok:   st.success(f"✓ {', '.join(ok)}")
                if fail: st.warning(f"⚠ Sin datos Yahoo (manual): {', '.join(fail)}")
                st.rerun()

            with st.expander("➕ Agregar activo"):
                with st.form(f"aa_{cid}"):
                    a1,a2,a3 = st.columns(3)
                    ticker   = a1.text_input("Ticker",placeholder="GGAL.BA, AAPL...")
                    tipo     = a2.selectbox("Tipo",["Acción","CEDEAR","Bono","FCI","Cripto"])
                    def_cur  = DEFAULT_CURRENCY.get(tipo,"ARS")
                    currency = a3.selectbox("Moneda",["ARS","USD"],index=0 if def_cur=="ARS" else 1)
                    b1,b2,b3 = st.columns(3)
                    qty       = b1.number_input("Cantidad",min_value=0.0,step=1.0)
                    price_buy = b2.number_input(f"P. Compra ({currency})",min_value=0.0,step=0.01,format="%.2f")
                    price_cur_inp = b3.number_input(f"P. Actual ({currency})",min_value=0.0,step=0.01,format="%.2f")
                    if price_cur_inp and currency=="ARS" and mep:
                        st.caption(f"≈ {fmt_usd(to_usd(price_cur_inp*qty,mep))} al MEP ${mep:,.0f}")
                    if st.form_submit_button("Agregar",type="primary") and ticker:
                        sb.table("assets").insert({"client_id":cid,"ticker":ticker.upper(),
                            "tipo":tipo,"currency":currency,"qty":qty,
                            "price_buy":price_buy,"price_current":price_cur_inp}).execute()
                        st.success(f"✓ {ticker.upper()} en {currency}"); st.rerun()

            if not cas.empty:
                rows = []
                for _,a in cas.iterrows():
                    cur = a.get("currency","ARS")
                    r   = ((a["price_current"]-a["price_buy"])/a["price_buy"]*100) if a["price_buy"] else 0
                    rows.append({"Ticker":a["ticker"],"Tipo":a["tipo"],"Moneda":cur,"Qty":a["qty"],
                        "P.Compra": fmt_ars(a["price_buy"]) if cur=="ARS" else f"USD {a['price_buy']:.2f}",
                        "P.Actual": fmt_ars(a["price_current"]) if cur=="ARS" else f"USD {a['price_current']:.2f}",
                        "Valor USD": fmt_usd(asset_val_usd(a,mep)), "Rend.":f"{r:+.1f}%"})
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

                with st.expander("✏ Actualizar precio manual"):
                    with st.form(f"upd_{cid}"):
                        upd_t = st.selectbox("Activo",cas["ticker"].tolist())
                        upd_r = cas[cas["ticker"]==upd_t].iloc[0]
                        upd_c = upd_r.get("currency","ARS")
                        new_p = st.number_input(f"Nuevo precio ({upd_c})",
                            value=float(upd_r["price_current"]),min_value=0.0,step=0.01,format="%.2f")
                        if st.form_submit_button("Actualizar",type="primary"):
                            sb.table("assets").update({"price_current":new_p}).eq("id",str(upd_r["id"])).execute()
                            st.success("✓ Actualizado"); st.rerun()

                with st.expander("🗑 Eliminar activo"):
                    to_del = st.selectbox("Eliminar",cas["ticker"].tolist(),key="del")
                    if st.button("Eliminar activo",type="secondary"):
                        sb.table("assets").delete().eq("id",str(cas[cas["ticker"]==to_del]["id"].values[0])).execute()
                        st.success("✓ Eliminado"); st.rerun()
            else:
                st.info("Sin activos.")

            with st.expander("⚠ Zona de peligro"):
                st.warning("Elimina el cliente y todos sus activos.")
                if st.button("🗑 Eliminar cliente",type="secondary"):
                    sb.table("assets").delete().eq("client_id",cid).execute()
                    sb.table("cash_flow").delete().eq("client_id",cid).execute()
                    sb.table("profile_history").delete().eq("client_id",cid).execute()
                    sb.table("clients").delete().eq("id",cid).execute()
                    st.success("Eliminado"); st.rerun()

        # ════════════════════════════════════
        # TAB: FLUJO DE CAJA
        # ════════════════════════════════════
        with tab_cf:
            with st.expander("➕ Agregar concepto"):
                with st.form(f"cf_add_{cid}"):
                    cf1,cf2,cf3 = st.columns(3)
                    cf_concepto  = cf1.text_input("Concepto", placeholder="Sueldo, Alquiler, Cuota...")
                    cf_tipo      = cf2.selectbox("Tipo", ["Ingreso","Egreso","Compromiso"])
                    cf_freq      = cf3.selectbox("Frecuencia", ["Mensual","Bimestral","Trimestral","Semestral","Anual","Único"])
                    cf4,cf5,cf6  = st.columns(3)
                    cf_monto     = cf4.number_input("Monto", min_value=0.0, step=1000.0)
                    cf_moneda    = cf5.selectbox("Moneda", ["ARS","USD"])
                    cf_notas     = cf6.text_input("Notas", placeholder="Opcional")
                    if st.form_submit_button("Agregar", type="primary") and cf_concepto:
                        sb.table("cash_flow").insert({
                            "client_id": cid, "concepto": cf_concepto, "tipo": cf_tipo,
                            "monto": cf_monto, "moneda": cf_moneda,
                            "frecuencia": cf_freq, "notas": cf_notas
                        }).execute()
                        st.success("✓ Agregado"); st.rerun()

            if not cf_df.empty:
                # Summary by tipo
                s1,s2,s3 = st.columns(3)
                def cf_total(tipo):
                    sub = cf_df[cf_df["tipo"]==tipo]
                    if sub.empty: return 0
                    return sum(
                        r["monto"] if r["moneda"]=="ARS" else r["monto"]*mep
                        for _,r in sub.iterrows()
                    )
                ing  = cf_total("Ingreso")
                egr  = cf_total("Egreso")
                comp = cf_total("Compromiso")
                s1.metric("Ingresos / mes",    fmt_ars(ing),  delta=None)
                s2.metric("Egresos / mes",      fmt_ars(egr),  delta=None)
                s3.metric("Compromisos / mes",  fmt_ars(comp), delta=None)
                st.markdown("---")

                for tipo, icon in [("Ingreso","🟢"),("Egreso","🔴"),("Compromiso","🟡")]:
                    sub = cf_df[cf_df["tipo"]==tipo]
                    if sub.empty: continue
                    st.markdown(f"**{icon} {tipo}s**")
                    rows_cf = []
                    for _,r in sub.iterrows():
                        rows_cf.append({
                            "Concepto":   r["concepto"],
                            "Monto":      fmt_ars(r["monto"]) if r["moneda"]=="ARS" else f"USD {r['monto']:,.2f}",
                            "Frecuencia": r["frecuencia"],
                            "Notas":      r.get("notas","") or "",
                            "_id":        r["id"]
                        })
                    st.dataframe(pd.DataFrame(rows_cf).drop(columns=["_id"]),
                                 use_container_width=True, hide_index=True)

                with st.expander("🗑 Eliminar concepto"):
                    opts = {f"{r['concepto']} ({r['tipo']})": r["id"] for _,r in cf_df.iterrows()}
                    to_del_cf = st.selectbox("Seleccionar", list(opts.keys()), key="del_cf")
                    if st.button("Eliminar concepto", type="secondary"):
                        sb.table("cash_flow").delete().eq("id", str(opts[to_del_cf])).execute()
                        st.success("✓ Eliminado"); st.rerun()
            else:
                st.info("Sin conceptos registrados. Agregá ingresos, egresos o compromisos.")

        # ════════════════════════════════════
        # TAB: HISTORIAL
        # ════════════════════════════════════
        with tab_hist:
            if ph_df.empty:
                st.info("Sin cambios registrados aún. Cada vez que modifiques el perfil quedará registrado acá.")
            else:
                for _,h in ph_df.iterrows():
                    fecha = str(h.get("fecha",""))[:10]
                    nota_txt = f"  \n_{h['nota']}_" if h.get("nota") else ""
                    st.markdown(
                        f"**{fecha}** — **{h['campo']}**: "
                        f"`{h.get('valor_antes','—')}` → `{h.get('valor_nuevo','—')}`"
                        + nota_txt
                    )
                    st.divider()

# ════════════════════════════════════
# SCREENER
# ════════════════════════════════════
elif page == "Screener":
    st.title("Screener"); st.caption("Indicadores técnicos y fundamentales — Yahoo Finance")
    if assets_df.empty:
        st.info("Sin posiciones.")
    else:
        ti = {}
        for _,a in assets_df.iterrows():
            t = a["ticker"]
            if t not in ti: ti[t]={"tipo":a["tipo"],"currency":a.get("currency","ARS"),"owners":[],"price":a["price_current"]}
            cl = clients_df[clients_df["id"]==a["client_id"]]
            if not cl.empty: ti[t]["owners"].append(cl.iloc[0]["name"])

        f1,f2,f3 = st.columns([2,1,1])
        ftipo = f1.multiselect("Tipo",["Acción","Bono","FCI","CEDEAR","Cripto"])
        fsig  = f2.selectbox("Señal",["Todas","Compra","Venta","Neutral"])
        if f3.button("↻ Actualizar",type="primary"): fetch_indicators.clear(); st.rerun()
        st.markdown("---")

        tlist = [t for t in ti if not ftipo or ti[t]["tipo"] in ftipo]
        results = []; prog = st.progress(0,text="Descargando...")
        for i,ticker in enumerate(tlist):
            prog.progress((i+1)/len(tlist),text=f"Cargando {ticker}...")
            info = ti[ticker]; tech,fund,_ = fetch_indicators(ticker)
            price = info["price"]; cur = info["currency"]
            price_fmt = fmt_ars(price) if cur=="ARS" else f"USD {price:.2f}"
            if tech:
                sigs = sig(rsi=tech["rsi"],price=price,ema50=tech["ema50"],ema200=tech["ema200"],
                           macd=tech["macd"],macd_s=tech["macd_sig"],
                           bb_u=tech["bb_upper"],bb_l=tech["bb_lower"],vol_r=tech["vol_ratio"])
                votes = [v[1] for v in sigs.values()]
                overall = "↑ Compra" if votes.count("buy")>=3 else "↓ Venta" if votes.count("sell")>=3 else "→ Neutral"
                row = {"Ticker":ticker,"Tipo":info["tipo"],"Moneda":cur,
                       "Clientes":", ".join(set(info["owners"])),"Precio":price_fmt,"Señal":overall,
                       **{k:v[0] for k,v in sigs.items()},
                       "Soporte":f"${tech['support']:,.2f}","Resistencia":f"${tech['resistance']:,.2f}",
                       "P/E":f"{fund['pe']:.1f}" if fund and fund.get("pe") else "—",
                       "EPS":f"{fund['eps']:.2f}" if fund and fund.get("eps") else "—",
                       "Div.Yield":f"{fund['div_yield']}%" if fund and fund.get("div_yield") else "—",
                       "MktCap":fmt_cap(fund["mkt_cap"]) if fund else "—",
                       "Beta":f"{fund['beta']:.2f}" if fund and fund.get("beta") else "—"}
                if fsig=="Todas" or fsig in overall: results.append(row)
            else:
                results.append({"Ticker":ticker,"Tipo":info["tipo"],"Moneda":cur,
                    "Clientes":", ".join(set(info["owners"])),"Precio":price_fmt,"Señal":"⚠ Sin datos",
                    **{k:"—" for k in ["RSI","EMA","MACD","Bollinger","Volumen","Soporte","Resistencia","P/E","EPS","Div.Yield","MktCap","Beta"]}})
        prog.empty()

        if results:
            df_r = pd.DataFrame(results)
            sig_cols = [c for c in ["Señal","RSI","EMA","MACD","Bollinger","Volumen"] if c in df_r.columns]
            st.dataframe(df_r.style.map(color_signal,subset=sig_cols),use_container_width=True,hide_index=True)

        st.markdown("---"); st.subheader("Gráfico")
        sel = st.selectbox("Ticker",tlist)
        _,_,hist = fetch_indicators(sel)
        if hist is not None:
            if isinstance(hist.columns,pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=hist.index,open=hist["Open"],high=hist["High"],
                low=hist["Low"],close=hist["Close"],name="Precio",
                increasing_line_color="#34d399",decreasing_line_color="#f87171"))
            for col,color,name in [("EMA50","#7c6fff","EMA 50"),("EMA200","#fbbf24","EMA 200")]:
                if col in hist.columns:
                    fig.add_trace(go.Scatter(x=hist.index,y=hist[col],name=name,line=dict(color=color,width=1.5)))
            if "BB_upper" in hist.columns:
                fig.add_trace(go.Scatter(x=hist.index,y=hist["BB_upper"],name="BB Sup",line=dict(color="#888898",width=1,dash="dash")))
                fig.add_trace(go.Scatter(x=hist.index,y=hist["BB_lower"],name="BB Inf",line=dict(color="#888898",width=1,dash="dash"),fill="tonexty",fillcolor="rgba(136,136,152,0.06)"))
            fig.update_layout(xaxis_rangeslider_visible=False,height=420,paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#f0f0f2"),margin=dict(t=20,b=10),
                legend=dict(orientation="h",y=1.02))
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)"); fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig,use_container_width=True)
            if "RSI" in hist.columns:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=hist.index,y=hist["RSI"],line=dict(color="#a78bfa",width=2)))
                fig2.add_hline(y=70,line_dash="dash",line_color="#f87171",annotation_text="70")
                fig2.add_hline(y=30,line_dash="dash",line_color="#34d399",annotation_text="30")
                fig2.update_layout(height=160,margin=dict(t=10,b=10),showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#f0f0f2"))
                fig2.update_yaxes(range=[0,100],gridcolor="rgba(255,255,255,0.05)")
                fig2.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
                st.plotly_chart(fig2,use_container_width=True)
        else:
            st.warning(f"Sin datos para {sel}. Verificá el ticker (ej: GGAL.BA, no GGAL).")

# ════════════════════════════════════
# REPORTES
# ════════════════════════════════════
elif page == "Reportes":
    st.title("Reportes")
    if clients_df.empty:
        st.info("Sin clientes.")
    else:
        selected = st.selectbox("Cliente",clients_df["name"].tolist())
        c = clients_df[clients_df["name"]==selected].iloc[0]; cid = c["id"]
        cas = assets_df[assets_df["client_id"]==cid].copy()
        if "currency" not in cas.columns: cas["currency"]="ARS"
        pat = patrimonio_usd(assets_df,cid,mep); rend = rend_pct(assets_df,cid,mep)
        als = get_alerts(assets_df,clients_df,cid)
        today = datetime.now().strftime("%d de %B de %Y")
        if st.button("Generar reporte",type="primary"):
            st.markdown("---")
            st.markdown(f"## Informe — {c['name']}")
            st.caption(f"{today}  |  Dólar MEP: ${mep:,.0f}")
            st.markdown("---")
            st.markdown("### Perfil")
            r1,r2 = st.columns(2)
            cap_ars = c.get("capital",0)
            r1.markdown(f"**Perfil:** {c.get('perfil','—')}  \n**Objetivo:** {c.get('objetivo','—')}")
            r2.markdown(f"**Horizonte:** {c.get('horizonte','—')}  \n**Capital:** {fmt_ars(cap_ars)} / {fmt_usd(to_usd(cap_ars,mep))}")
            st.markdown("### Resumen")
            m1,m2,m3 = st.columns(3)
            m1.metric("Patrimonio",fmt_usd(pat)); m2.metric("Rendimiento",f"{rend:+.1f}%"); m3.metric("Posiciones",len(cas))
            if not cas.empty:
                st.markdown("### Composición")
                bt = {}
                for _,a in cas.iterrows(): bt[a["tipo"]]=bt.get(a["tipo"],0)+asset_val_usd(a,mep)
                total = sum(bt.values()) or 1
                for tipo,val in bt.items():
                    pct = val/total*100; st.markdown(f"**{tipo}:** {fmt_usd(val)} ({pct:.0f}%)"); st.progress(pct/100)
                st.markdown("### Posiciones")
                rows = []
                for _,a in cas.iterrows():
                    cur=a.get("currency","ARS"); r=((a["price_current"]-a["price_buy"])/a["price_buy"]*100) if a["price_buy"] else 0
                    rows.append({"Ticker":a["ticker"],"Tipo":a["tipo"],"Moneda":cur,"Qty":a["qty"],
                        "P.Compra":fmt_ars(a["price_buy"]) if cur=="ARS" else f"USD {a['price_buy']:.2f}",
                        "P.Actual":fmt_ars(a["price_current"]) if cur=="ARS" else f"USD {a['price_current']:.2f}",
                        "Valor USD":fmt_usd(asset_val_usd(a,mep)),"Rend.":f"{r:+.1f}%"})
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
            if als:
                st.markdown("### ⚠ Alertas")
                for icon,msg in als: st.warning(f"{icon} {msg}")
            st.markdown("---"); st.caption("Reporte generado por Portfolio Tracker")
