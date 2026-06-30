# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Sales Prediction", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[class*="css"],.stApp{font-family:'Space Grotesk',sans-serif !important;background:#030d0a !important;color:#cce0d8}
.block-container{padding:0 !important;max-width:100% !important}
#MainMenu,footer,header,[data-testid="stToolbar"]{visibility:hidden;height:0}
section[data-testid="stSidebar"]{display:none}
.stButton>button{background:#071a10 !important;border:1px solid #1a6a40 !important;color:#2aaa70 !important;border-radius:8px !important;font-family:'Space Grotesk',sans-serif !important;font-size:.8rem !important;letter-spacing:.08em !important;padding:.6rem 1.4rem !important;transition:all .2s !important}
.stButton>button:hover{background:#0a2818 !important;border-color:#2aaa70 !important;color:#4acc90 !important}
[data-testid="stSelectbox"]>div>div{background:#071a10 !important;border:1px solid #0e2e1e !important;border-radius:8px !important;color:#cce0d8 !important}
[data-testid="stFileUploadDropzone"]{background:#071a10 !important;border:1px dashed #0e2e1e !important;border-radius:16px !important;padding:1.5rem !important}
section[data-testid="stFileUploaderDropzone"] button{background:#071a10 !important;border:1px solid #1a6a40 !important;color:#2aaa70 !important;border-radius:8px !important}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:#030d0a}::-webkit-scrollbar-thumb{background:#0e2e1e;border-radius:2px}
</style>
""", unsafe_allow_html=True)

PLOTLY = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Space Grotesk, sans-serif', color='#3a7a58', size=11),
    margin=dict(l=8,r=8,t=36,b=8),
    xaxis=dict(gridcolor='#0a1e14', linecolor='#0e2e1e', tickcolor='#0e2e1e'),
    yaxis=dict(gridcolor='#0a1e14', linecolor='#0e2e1e', tickcolor='#0e2e1e'),
)
AC = '#2aaa70'

@st.cache_data
def process(file):
    df = pd.read_csv(file, encoding='latin1')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ','_')
    sales_col = next((c for c in df.columns if 'sales' in c), None)
    if sales_col is None: st.error("No sales column found."); st.stop()
    df = df.rename(columns={sales_col:'sales'})
    df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
    df.dropna(subset=['sales'], inplace=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat = [c for c in num_cols if c != 'sales']
    if not feat: st.error("No numeric feature columns found."); st.stop()
    X = df[feat].fillna(df[feat].median())
    y = df['sales']
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, random_state=42); rf.fit(Xtr,ytr)
    gb = GradientBoostingRegressor(n_estimators=200, random_state=42); gb.fit(Xtr,ytr)
    lr = LinearRegression(); lr.fit(Xtr,ytr)
    def mets(m):
        p=m.predict(Xte)
        return {'MAE':mean_absolute_error(yte,p),'RMSE':np.sqrt(mean_squared_error(yte,p)),'R2':r2_score(yte,p)}
    results={'Random Forest':mets(rf),'Gradient Boosting':mets(gb),'Linear Regression':mets(lr)}
    fi = pd.Series(rf.feature_importances_, index=feat).sort_values()
    ypred = rf.predict(Xte)
    return df, rf, results, fi, feat, yte.values, ypred

st.markdown("""
<div style="background:#030d0a;border-bottom:1px solid #0e2e1e;padding:2.2rem 3rem 1.8rem;position:relative;overflow:hidden">
  <div style="position:absolute;top:-80px;right:-80px;width:320px;height:320px;background:radial-gradient(circle,#0a3a1e18 0%,transparent 70%);pointer-events:none"></div>
  <div style="font-size:.62rem;letter-spacing:.22em;text-transform:uppercase;color:#1a6a40;margin-bottom:.6rem">CodSoft &nbsp;/&nbsp; Data Science &nbsp;/&nbsp; Task 04</div>
  <div style="font-size:2.4rem;font-weight:700;line-height:1.05;color:#eaf5f0;letter-spacing:-.03em;margin-bottom:.4rem">Sales<br><span style="color:#2aaa70">Prediction</span></div>
  <div style="font-size:.82rem;color:#3a6a50;line-height:1.6">Forecasting product sales from advertising spend — Random Forest, Gradient Boosting &amp; Linear Regression.</div>
</div>
""", unsafe_allow_html=True)

_, mid, _ = st.columns([1,2,1])
with mid:
    st.markdown("""
    <div style="text-align:center;padding:1.8rem 0 .6rem">
      <div style="width:52px;height:52px;margin:0 auto 1rem;border:1px solid #0e2e1e;border-radius:12px;background:#030d0a;display:flex;align-items:center;justify-content:center">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2aaa70" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>
        </svg>
      </div>
      <div style="font-size:1rem;font-weight:600;color:#eaf5f0;margin-bottom:.3rem">Drop your dataset here</div>
      <div style="font-size:.78rem;color:#2a5a40;margin-bottom:1rem">Upload <span style="color:#2aaa70;font-family:monospace">advertising.csv</span> to begin</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("", type="csv", label_visibility="collapsed")

if uploaded is None:
    st.stop()

df, rf, results, fi, feat, y_true, y_pred = process(uploaded)
rf_r2  = results['Random Forest']['R2']
rf_mae = results['Random Forest']['MAE']

st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:#0e2e1e;border-bottom:1px solid #0e2e1e">
  <div style="background:#030d0a;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#1a4a30;margin-bottom:.35rem">Total records</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#eaf5f0">{len(df):,}</div></div>
  <div style="background:#030d0a;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#1a4a30;margin-bottom:.35rem">Avg sales</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#eaf5f0">{df['sales'].mean():.1f}</div></div>
  <div style="background:#030d0a;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#1a4a30;margin-bottom:.35rem">RF R² score</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#eaf5f0">{rf_r2:.3f}</div></div>
  <div style="background:#030d0a;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#1a4a30;margin-bottom:.35rem">RF MAE</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#eaf5f0">{rf_mae:.2f}</div></div>
  <div style="background:#030d0a;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#1a4a30;margin-bottom:.35rem">Features</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#eaf5f0">{len(feat)}</div></div>
</div>
""", unsafe_allow_html=True)

if 'tab' not in st.session_state: st.session_state.tab = 'explore'
n1,n2,n3,_ = st.columns([1,1,1,6])
with n1:
    if st.button("📊  Explore", use_container_width=True): st.session_state.tab='explore'
with n2:
    if st.button("🤖  Models",  use_container_width=True): st.session_state.tab='models'
with n3:
    if st.button("🔮  Predict", use_container_width=True): st.session_state.tab='predict'

tab = st.session_state.tab
st.markdown("<div style='padding:2rem 3rem;background:#030d0a'>", unsafe_allow_html=True)

if tab == 'explore':
    st.markdown("<div style='font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:#1a6a40;margin-bottom:1.4rem'>Sales analysis</div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Histogram(x=df['sales'], nbinsx=25, marker_color=AC, marker_line_width=0, opacity=0.85))
        fig.update_layout(**PLOTLY, title='Sales distribution', height=300)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if len(feat) >= 1:
            f = feat[0]
            fig = px.scatter(df, x=f, y='sales', title=f'{f.title()} vs Sales', opacity=0.6)
            fig.update_traces(marker=dict(color=AC, size=5, line_width=0))
            fig.update_layout(**PLOTLY, height=300)
            st.plotly_chart(fig, use_container_width=True)
    if len(feat) >= 2:
        c3,c4 = st.columns(2)
        for i,(col,f) in enumerate(zip([c3,c4], feat[1:3])):
            with col:
                fig = px.scatter(df, x=f, y='sales', title=f'{f.title()} vs Sales', opacity=0.6, trendline='ols')
                fig.update_traces(marker=dict(color=AC, size=5, line_width=0))
                fig.update_layout(**PLOTLY, height=300)
                st.plotly_chart(fig, use_container_width=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', marker=dict(color=AC, size=5, opacity=0.6, line_width=0)))
    mn,mx=min(y_true.min(),y_pred.min()),max(y_true.max(),y_pred.max())
    fig.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode='lines',line=dict(color='#1a4a30',dash='dash',width=1),showlegend=False))
    fig.update_layout(**PLOTLY, title='Actual vs predicted sales', height=320, xaxis_title='Actual', yaxis_title='Predicted', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

elif tab == 'models':
    st.markdown("<div style='font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:#1a6a40;margin-bottom:1.4rem'>Model performance</div>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col,met in zip([c1,c2,c3],['MAE','RMSE','R2']):
        with col:
            vals=[results[m][met] for m in results]
            fig=go.Figure(go.Bar(x=list(results.keys()),y=vals,marker_color=['#0a2818','#1a6a40',AC],marker_line_width=0,text=[f'{v:.3f}' for v in vals],textposition='outside',textfont=dict(family='Space Mono',size=11,color=AC)))
            fig.update_layout(**PLOTLY,title=met,height=280)
            st.plotly_chart(fig,use_container_width=True)
    fi_s=fi.sort_values()
    fig=go.Figure(go.Bar(x=fi_s.values*100,y=fi_s.index,orientation='h',marker_color=AC,marker_line_width=0,opacity=0.85,text=[f'{v*100:.1f}%' for v in fi_s.values],textposition='outside',textfont=dict(family='Space Mono',size=11,color=AC)))
    fig.update_layout(**PLOTLY,title='Feature importance',xaxis_title='Importance (%)',height=300)
    st.plotly_chart(fig,use_container_width=True)

elif tab == 'predict':
    st.markdown("<div style='font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:#1a6a40;margin-bottom:1.4rem'>Predict sales</div>", unsafe_allow_html=True)
    inp_vals={}
    cols=st.columns(len(feat)) if len(feat)<=4 else st.columns(4)
    for i,f in enumerate(feat):
        with cols[i%len(cols)]:
            label=f.replace('_',' ').title()
            default=float(df[f].median())
            mn_v=float(df[f].min()); mx_v=float(df[f].max())
            inp_vals[f]=st.slider(label, mn_v, mx_v, default)
    if st.button("Predict sales", use_container_width=True):
        inp=pd.DataFrame([inp_vals])
        pred=rf.predict(inp)[0]
        avg=df['sales'].mean()
        diff=((pred-avg)/avg)*100
        st.markdown(f"""
        <div style="background:#071a10;border:1px solid #1a6a4044;border-radius:12px;padding:1.6rem 2rem;margin-top:1rem;display:flex;align-items:center;justify-content:space-between">
          <div><div style="font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;color:#2a6a48">Predicted sales</div>
          <div style="font-family:'Space Mono',monospace;font-size:2.4rem;font-weight:700;color:{AC};margin-top:.2rem">{pred:.2f}</div>
          <div style="font-size:.8rem;color:#1a5a38;margin-top:.3rem">units / revenue</div></div>
          <div style="text-align:right"><div style="font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;color:#2a6a48">vs avg</div>
          <div style="font-family:'Space Mono',monospace;font-size:1.4rem;font-weight:700;color:{'#2aaa70' if diff>=0 else '#cc5555'};margin-top:.2rem">{'+' if diff>=0 else ''}{diff:.1f}%</div></div>
        </div>
        <div style="margin-top:.8rem">
          <div style="background:#0a1e14;border-radius:4px;height:6px;overflow:hidden"><div style="height:100%;width:{min(100,int(pred/df['sales'].max()*100))}%;background:{AC};border-radius:4px"></div></div>
          <div style="display:flex;justify-content:space-between;font-size:.68rem;color:#1a4a30;margin-top:.35rem;font-family:'Space Mono',monospace"><span>0</span><span>Max: {df['sales'].max():.1f}</span></div>
        </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)