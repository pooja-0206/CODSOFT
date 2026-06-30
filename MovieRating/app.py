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
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Movie Rating Prediction", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[class*="css"],.stApp{font-family:'Space Grotesk',sans-serif !important;background:#0d0800 !important;color:#e0d8cc}
.block-container{padding:0 !important;max-width:100% !important}
#MainMenu,footer,header,[data-testid="stToolbar"]{visibility:hidden;height:0}
section[data-testid="stSidebar"]{display:none}
.stButton>button{background:#1a1000 !important;border:1px solid #6a4a00 !important;color:#d4a830 !important;border-radius:8px !important;font-family:'Space Grotesk',sans-serif !important;font-size:.8rem !important;letter-spacing:.08em !important;padding:.6rem 1.4rem !important;transition:all .2s !important}
.stButton>button:hover{background:#2a1800 !important;border-color:#d4a830 !important;color:#f0c840 !important}
[data-testid="stSelectbox"]>div>div{background:#1a1000 !important;border:1px solid #3a2800 !important;border-radius:8px !important;color:#e0d8cc !important}
[data-testid="stFileUploadDropzone"]{background:#1a1000 !important;border:1px dashed #3a2800 !important;border-radius:16px !important;padding:1.5rem !important}
section[data-testid="stFileUploaderDropzone"] button{background:#1a1000 !important;border:1px solid #6a4a00 !important;color:#d4a830 !important;border-radius:8px !important}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:#0d0800}::-webkit-scrollbar-thumb{background:#3a2800;border-radius:2px}
</style>
""", unsafe_allow_html=True)

PLOTLY = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Space Grotesk, sans-serif', color='#8a7040', size=11),
    margin=dict(l=8,r=8,t=36,b=8),
    xaxis=dict(gridcolor='#2a1800', linecolor='#3a2800', tickcolor='#3a2800'),
    yaxis=dict(gridcolor='#2a1800', linecolor='#3a2800', tickcolor='#3a2800'),
)
AC = '#d4a830'

@st.cache_data
def process(file):
    df = pd.read_csv(file, encoding='latin1')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ','_')
    # find rating column
    rating_col = next((c for c in df.columns if 'rating' in c), None)
    if rating_col is None:
        st.error("No rating column found in dataset."); st.stop()
    df = df.rename(columns={rating_col: 'rating'})
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df.dropna(subset=['rating'], inplace=True)
    # encode categoricals
    for col in ['genre','director','actor_1','actor_2','actor_3']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            df[col+'_enc'] = LabelEncoder().fit_transform(df[col].astype(str))
    # numeric features
    num_feats = ['genre_enc','director_enc','actor_1_enc','actor_2_enc','actor_3_enc']
    num_feats = [f for f in num_feats if f in df.columns]
    for nc in ['year','votes','duration']:
        if nc in df.columns:
            df[nc] = pd.to_numeric(df[nc].astype(str).str.replace(',',''), errors='coerce')
            df[nc].fillna(df[nc].median(), inplace=True)
            num_feats.append(nc)
    if len(num_feats) < 2:
        st.error("Not enough features found. Check your dataset columns."); st.stop()
    X = df[num_feats].fillna(0)
    y = df['rating']
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, random_state=42);       rf.fit(Xtr,ytr)
    gb = GradientBoostingRegressor(n_estimators=200, random_state=42);   gb.fit(Xtr,ytr)
    lr = LinearRegression();                                              lr.fit(Xtr,ytr)
    def metrics(model):
        p = model.predict(Xte)
        return {'MAE':mean_absolute_error(yte,p), 'RMSE':np.sqrt(mean_squared_error(yte,p)), 'R2':r2_score(yte,p)}
    results = {'Random Forest':metrics(rf), 'Gradient Boosting':metrics(gb), 'Linear Regression':metrics(lr)}
    fi = pd.Series(rf.feature_importances_, index=num_feats).sort_values()
    ypred = rf.predict(Xte)
    return df, rf, results, fi, num_feats, yte.values, ypred

st.markdown("""
<div style="background:#0d0800;border-bottom:1px solid #3a2800;padding:2.2rem 3rem 1.8rem;position:relative;overflow:hidden">
  <div style="position:absolute;top:-80px;right:-80px;width:320px;height:320px;background:radial-gradient(circle,#6a380018 0%,transparent 70%);pointer-events:none"></div>
  <div style="font-size:.62rem;letter-spacing:.22em;text-transform:uppercase;color:#8a6020;margin-bottom:.6rem">CodSoft &nbsp;/&nbsp; Data Science &nbsp;/&nbsp; Task 02</div>
  <div style="font-size:2.4rem;font-weight:700;line-height:1.05;color:#f5eedc;letter-spacing:-.03em;margin-bottom:.4rem">Movie Rating<br><span style="color:#d4a830">Prediction</span></div>
  <div style="font-size:.82rem;color:#7a6040;line-height:1.6">Predicting movie ratings from genre, director &amp; cast — Random Forest, Gradient Boosting &amp; Linear Regression.</div>
</div>
""", unsafe_allow_html=True)

_, mid, _ = st.columns([1,2,1])
with mid:
    st.markdown("""
    <div style="text-align:center;padding:1.8rem 0 .6rem">
      <div style="width:52px;height:52px;margin:0 auto 1rem;border:1px solid #3a2800;border-radius:12px;background:#0d0800;display:flex;align-items:center;justify-content:center">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#d4a830" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="7" x2="7" y2="7"/><line x1="2" y1="17" x2="7" y2="17"/><line x1="17" y1="7" x2="22" y2="7"/><line x1="17" y1="17" x2="22" y2="17"/>
        </svg>
      </div>
      <div style="font-size:1rem;font-weight:600;color:#f5eedc;margin-bottom:.3rem">Drop your dataset here</div>
      <div style="font-size:.78rem;color:#6a4a20;margin-bottom:1rem">Upload the <span style="color:#d4a830;font-family:monospace">IMDb Movies India.csv</span> file to begin</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("", type="csv", label_visibility="collapsed")

if uploaded is None:
    st.stop()

df, rf, results, fi, feat, y_true, y_pred = process(uploaded)
rf_r2  = results['Random Forest']['R2']
rf_mae = results['Random Forest']['MAE']

st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:#3a2800;border-bottom:1px solid #3a2800">
  <div style="background:#0d0800;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#4a3010;margin-bottom:.35rem">Total movies</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#f5eedc">{len(df):,}</div></div>
  <div style="background:#0d0800;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#4a3010;margin-bottom:.35rem">Avg rating</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#f5eedc">{df['rating'].mean():.1f}</div><div style="font-size:.7rem;color:#d4a830;margin-top:.15rem">out of 10</div></div>
  <div style="background:#0d0800;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#4a3010;margin-bottom:.35rem">RF R² score</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#f5eedc">{rf_r2:.3f}</div></div>
  <div style="background:#0d0800;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#4a3010;margin-bottom:.35rem">RF MAE</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#f5eedc">{rf_mae:.2f}</div></div>
  <div style="background:#0d0800;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#4a3010;margin-bottom:.35rem">Features used</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#f5eedc">{len(feat)}</div></div>
</div>
""", unsafe_allow_html=True)

if 'tab' not in st.session_state: st.session_state.tab = 'explore'
n1,n2,n3,_ = st.columns([1,1,1,6])
with n1:
    if st.button("🎬  Explore", use_container_width=True): st.session_state.tab='explore'
with n2:
    if st.button("🤖  Models",  use_container_width=True): st.session_state.tab='models'
with n3:
    if st.button("🔮  Predict", use_container_width=True): st.session_state.tab='predict'

tab = st.session_state.tab
st.markdown("<div style='padding:2rem 3rem;background:#0d0800'>", unsafe_allow_html=True)

if tab == 'explore':
    st.markdown("<div style='font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:#8a6020;margin-bottom:1.4rem'>Rating analysis</div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Histogram(x=df['rating'], nbinsx=30, marker_color=AC, marker_line_width=0, opacity=0.85))
        fig.update_layout(**PLOTLY, title='Rating distribution', height=300)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if 'genre' in df.columns:
            top_genres = df.groupby('genre')['rating'].mean().sort_values(ascending=False).head(10)
            fig = go.Figure(go.Bar(x=top_genres.values, y=top_genres.index, orientation='h',
                                   marker_color=AC, marker_line_width=0, opacity=0.85))
            fig.update_layout(**PLOTLY, title='Top genres by avg rating', height=300)
            st.plotly_chart(fig, use_container_width=True)
    if 'year' in df.columns:
        yr = df.groupby('year')['rating'].mean().reset_index()
        fig = px.line(yr, x='year', y='rating', title='Average rating over years')
        fig.update_traces(line_color=AC, line_width=2)
        fig.update_layout(**PLOTLY, height=300)
        st.plotly_chart(fig, use_container_width=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers',
                             marker=dict(color=AC, size=4, opacity=0.5, line_width=0)))
    mn,mx = min(y_true.min(),y_pred.min()), max(y_true.max(),y_pred.max())
    fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
                             line=dict(color='#6a4a20', dash='dash', width=1), showlegend=False))
    fig.update_layout(**PLOTLY, title='Actual vs predicted ratings', height=340,
                      xaxis_title='Actual', yaxis_title='Predicted', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

elif tab == 'models':
    st.markdown("<div style='font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:#8a6020;margin-bottom:1.4rem'>Model comparison</div>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    metrics_list = ['MAE','RMSE','R2']
    colors_m = ['#3a2800','#6a4a10',AC]
    for i,(col,met) in enumerate(zip([c1,c2,c3], metrics_list)):
        with col:
            vals = [results[m][met] for m in results]
            fig = go.Figure(go.Bar(x=list(results.keys()), y=vals, marker_color=colors_m,
                                   marker_line_width=0, text=[f'{v:.3f}' for v in vals],
                                   textposition='outside', textfont=dict(family='Space Mono',size=11,color='#d4a830')))
            fig.update_layout(**PLOTLY, title=met, height=280)
            st.plotly_chart(fig, use_container_width=True)
    fi_s = fi.sort_values()
    fig = go.Figure(go.Bar(x=fi_s.values*100, y=fi_s.index, orientation='h',
                           marker_color=AC, marker_line_width=0, opacity=0.85,
                           text=[f'{v*100:.1f}%' for v in fi_s.values], textposition='outside',
                           textfont=dict(family='Space Mono',size=11,color='#d4a830')))
    fig.update_layout(**PLOTLY, title='Feature importance', xaxis_title='Importance (%)', height=320)
    st.plotly_chart(fig, use_container_width=True)

elif tab == 'predict':
    st.markdown("<div style='font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:#8a6020;margin-bottom:1.4rem'>Predict a movie rating</div>", unsafe_allow_html=True)
    inp_vals = {}
    cols = st.columns(3)
    for i, f in enumerate(feat):
        with cols[i % 3]:
            label = f.replace('_enc','').replace('_',' ').title()
            if f in ['year','votes','duration']:
                default = float(df[f].median()) if f in df.columns else 0.0
                inp_vals[f] = st.number_input(label, value=default)
            else:
                orig_col = f.replace('_enc','')
                if orig_col in df.columns:
                    options = sorted(df[orig_col].dropna().unique().tolist())
                    sel = st.selectbox(label, options)
                    from sklearn.preprocessing import LabelEncoder as LE
                    le = LE(); le.fit(df[orig_col].astype(str))
                    try: inp_vals[f] = le.transform([str(sel)])[0]
                    except: inp_vals[f] = 0
                else:
                    inp_vals[f] = st.number_input(label, value=0.0)
    if st.button("Predict rating", use_container_width=True):
        inp  = pd.DataFrame([inp_vals])
        pred = rf.predict(inp)[0]
        stars = int(round(pred / 2))
        star_str = '★' * stars + '☆' * (5 - stars)
        st.markdown(f"""
        <div style="background:#1a1000;border:1px solid #6a4a0044;border-radius:12px;padding:1.6rem 2rem;margin-top:1rem;display:flex;align-items:center;justify-content:space-between">
          <div><div style="font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;color:#7a5a20">Predicted rating</div>
          <div style="font-family:'Space Mono',monospace;font-size:2.4rem;font-weight:700;color:{AC};margin-top:.2rem">{pred:.1f} <span style="font-size:1rem;color:#6a4a20">/ 10</span></div>
          <div style="font-size:1.4rem;color:{AC};margin-top:.3rem;letter-spacing:.1em">{star_str}</div></div>
          <div style="text-align:right"><div style="font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;color:#7a5a20">Rating tier</div>
          <div style="font-family:'Space Mono',monospace;font-size:1.2rem;font-weight:700;margin-top:.2rem;color:{'#1aaa7a' if pred>=7 else '#d4a830' if pred>=5 else '#cc5555'}">{'EXCELLENT' if pred>=7 else 'AVERAGE' if pred>=5 else 'POOR'}</div></div>
        </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)