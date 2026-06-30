# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Titanic | Survival Intelligence", page_icon="🚢", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"], .stApp { font-family: 'Space Grotesk', sans-serif !important; background: #03070d !important; color: #cdd6e0; }
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; height: 0; }
section[data-testid="stSidebar"] { display: none; }
.stButton > button { background: #071828 !important; border: 1px solid #1a4a6a !important; color: #7ab8d8 !important; border-radius: 8px !important; font-family: 'Space Grotesk', sans-serif !important; font-size: 0.8rem !important; letter-spacing: 0.08em !important; padding: 0.6rem 1.4rem !important; transition: all 0.2s !important; }
.stButton > button:hover { background: #0a2438 !important; border-color: #2a7aaa !important; color: #aad4ee !important; }
[data-testid="stSelectbox"] > div > div { background: #060d14 !important; border: 1px solid #0e1e2e !important; border-radius: 8px !important; color: #cdd6e0 !important; }
[data-testid="stFileUploadDropzone"] { background: #060d14 !important; border: 1px dashed #1a3a54 !important; border-radius: 16px !important; padding: 1.5rem !important; }
[data-testid="stFileUploadDropzone"]:hover { background: #07111c !important; border-color: #2a6a9a !important; }
section[data-testid="stFileUploaderDropzone"] button { background: #071828 !important; border: 1px solid #1a4a6a !important; color: #7ab8d8 !important; border-radius: 8px !important; }
::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: #03070d; } ::-webkit-scrollbar-thumb { background: #0e1e2e; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

PLOTLY = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Space Grotesk, sans-serif', color='#3a6a88', size=11),
    margin=dict(l=8, r=8, t=36, b=8),
    xaxis=dict(gridcolor='#0a1a24', linecolor='#0e1e2e', tickcolor='#0e1e2e'),
    yaxis=dict(gridcolor='#0a1a24', linecolor='#0e1e2e', tickcolor='#0e1e2e'),
)
SC = '#1aaa7a'
PC = '#cc5555'

@st.cache_data
def process(file):
    df = pd.read_csv(file, encoding='latin1')
    df.drop(columns=[c for c in ['PassengerId','Name','Ticket','Cabin'] if c in df.columns], inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['SibSp'].fillna(0, inplace=True)
    df['Parch'].fillna(0, inplace=True)
    df['Sex_enc'] = LabelEncoder().fit_transform(df['Sex'])
    df['Emb_enc'] = LabelEncoder().fit_transform(df['Embarked'])
    feat = ['Pclass','Sex_enc','Age','SibSp','Parch','Fare','Emb_enc']
    X, y = df[feat], df['Survived']
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=300, random_state=42); rf.fit(Xtr, ytr)
    lr = LogisticRegression(max_iter=500);                          lr.fit(Xtr, ytr)
    gb = GradientBoostingClassifier(n_estimators=200, random_state=42); gb.fit(Xtr, ytr)
    accs = {
        'Random Forest':       accuracy_score(yte, rf.predict(Xte)),
        'Logistic Regression': accuracy_score(yte, lr.predict(Xte)),
        'Gradient Boosting':   accuracy_score(yte, gb.predict(Xte)),
    }
    cm  = confusion_matrix(yte, rf.predict(Xte))
    rpt = classification_report(yte, rf.predict(Xte), output_dict=True)
    fi  = pd.Series(rf.feature_importances_, index=['Class','Gender','Age','Siblings','Parents','Fare','Embarked']).sort_values()
    return df, rf, accs, cm, rpt, fi, feat

st.markdown("""
<div style="background:#03070d;border-bottom:1px solid #0e1e2e;padding:2.2rem 3rem 1.8rem;position:relative;overflow:hidden">
  <div style="position:absolute;top:-100px;right:-100px;width:380px;height:380px;background:radial-gradient(circle,#0a3a6e1a 0%,transparent 70%);pointer-events:none"></div>
  <div style="font-size:.62rem;letter-spacing:.22em;text-transform:uppercase;color:#2a6496;margin-bottom:.6rem">CodSoft &nbsp;/&nbsp; Data Science &nbsp;/&nbsp; Task 01</div>
  <div style="font-size:2.4rem;font-weight:700;line-height:1.05;color:#eaf0f6;letter-spacing:-.03em;margin-bottom:.4rem">Titanic Survival<br><span style="color:#1a8a6a">Intelligence</span></div>
  <div style="font-size:.82rem;color:#4a6a80;line-height:1.6">Machine learning analysis of 891 passengers — Random Forest, Logistic Regression &amp; Gradient Boosting.</div>
</div>
""", unsafe_allow_html=True)

_, mid, _ = st.columns([1, 2, 1])
with mid:
    st.markdown("""
    <div style="text-align:center;padding:1.8rem 0 .6rem">
      <div style="width:52px;height:52px;margin:0 auto 1rem;border:1px solid #0e2e48;border-radius:12px;background:#03070d;display:flex;align-items:center;justify-content:center">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1a6a9a" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
      </div>
      <div style="font-size:1rem;font-weight:600;color:#eaf0f6;margin-bottom:.3rem">Drop your dataset here</div>
      <div style="font-size:.78rem;color:#2a5a78;margin-bottom:1rem">Upload <span style="color:#1a7a5a;font-family:monospace">Titanic-Dataset.csv</span> to begin</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("", type="csv", label_visibility="collapsed")
    st.markdown("""
    <div style="display:flex;justify-content:center;gap:1.8rem;padding:1rem 0 .5rem">
      <div style="display:flex;align-items:center;gap:.45rem"><div style="width:20px;height:20px;border-radius:50%;background:#060d14;border:1px solid #0e2e48;font-size:.6rem;font-family:monospace;color:#2a5a78;display:flex;align-items:center;justify-content:center;flex-shrink:0">1</div><span style="font-size:.68rem;color:#2a4a60;letter-spacing:.04em">Upload CSV</span></div>
      <div style="display:flex;align-items:center;gap:.45rem"><div style="width:20px;height:20px;border-radius:50%;background:#060d14;border:1px solid #0e2e48;font-size:.6rem;font-family:monospace;color:#2a5a78;display:flex;align-items:center;justify-content:center;flex-shrink:0">2</div><span style="font-size:.68rem;color:#2a4a60;letter-spacing:.04em">Explore data</span></div>
      <div style="display:flex;align-items:center;gap:.45rem"><div style="width:20px;height:20px;border-radius:50%;background:#060d14;border:1px solid #0e2e48;font-size:.6rem;font-family:monospace;color:#2a5a78;display:flex;align-items:center;justify-content:center;flex-shrink:0">3</div><span style="font-size:.68rem;color:#2a4a60;letter-spacing:.04em">Train models</span></div>
      <div style="display:flex;align-items:center;gap:.45rem"><div style="width:20px;height:20px;border-radius:50%;background:#060d14;border:1px solid #0e2e48;font-size:.6rem;font-family:monospace;color:#2a5a78;display:flex;align-items:center;justify-content:center;flex-shrink:0">4</div><span style="font-size:.68rem;color:#2a4a60;letter-spacing:.04em">Predict</span></div>
    </div>
    """, unsafe_allow_html=True)

if uploaded is None:
    st.stop()

df, rf, accs, cm, rpt, fi, feat = process(uploaded)
survived = int(df['Survived'].sum())
perished = int((df['Survived']==0).sum())
total    = len(df)
rate     = survived / total * 100
rf_acc   = accs['Random Forest']

st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:#0e1e2e;border-bottom:1px solid #0e1e2e">
  <div style="background:#03070d;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#2a4a60;margin-bottom:.35rem">Total passengers</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#eaf0f6">{total}</div></div>
  <div style="background:#03070d;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#2a4a60;margin-bottom:.35rem">Survived</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#eaf0f6">{survived}</div><div style="font-size:.7rem;color:#1a8a6a;margin-top:.15rem">+{rate:.1f}% survival rate</div></div>
  <div style="background:#03070d;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#2a4a60;margin-bottom:.35rem">Perished</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#eaf0f6">{perished}</div><div style="font-size:.7rem;color:#b04040;margin-top:.15rem">{100-rate:.1f}% fatality rate</div></div>
  <div style="background:#03070d;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#2a4a60;margin-bottom:.35rem">RF accuracy</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#eaf0f6">{rf_acc*100:.1f}%</div></div>
  <div style="background:#03070d;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#2a4a60;margin-bottom:.35rem">Avg age</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#eaf0f6">{df['Age'].mean():.0f} yrs</div></div>
</div>
""", unsafe_allow_html=True)

if 'tab' not in st.session_state: st.session_state.tab = 'explore'
n1, n2, n3, _ = st.columns([1,1,1,6])
with n1:
    if st.button("📊  Explore", use_container_width=True): st.session_state.tab = 'explore'
with n2:
    if st.button("🤖  Models",  use_container_width=True): st.session_state.tab = 'models'
with n3:
    if st.button("🔮  Predict", use_container_width=True): st.session_state.tab = 'predict'

tab = st.session_state.tab
st.markdown("<div style='padding:2rem 3rem;background:#03070d'>", unsafe_allow_html=True)

if tab == 'explore':
    st.markdown("<div style='font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:#1a5a7a;margin-bottom:1.4rem'>Survival analysis</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Pie(labels=['Survived','Perished'], values=[survived, perished], hole=0.68, marker_colors=[SC,PC], textinfo='percent+label', textfont=dict(size=12,family='Space Mono')))
        fig.add_annotation(text=f"<b>{rate:.0f}%</b>", x=0.5, y=0.5, showarrow=False, font=dict(size=24,color='#eaf0f6',family='Space Mono'))
        fig.update_layout(**PLOTLY, title='Overall survival', height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        gdf = df.groupby(['Sex','Survived']).size().reset_index(name='n')
        gdf['outcome'] = gdf['Survived'].map({0:'Perished',1:'Survived'})
        fig = px.bar(gdf, x='Sex', y='n', color='outcome', barmode='group', color_discrete_map={'Survived':SC,'Perished':PC}, title='Gender vs survival')
        fig.update_traces(marker_line_width=0, width=0.35)
        fig.update_layout(**PLOTLY, height=300, legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#4a7a98')))
        st.plotly_chart(fig, use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        cls = df.groupby(['Pclass','Survived']).size().unstack(fill_value=0)
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Survived', x=[f'Class {i}' for i in cls.index], y=cls[1], marker_color=SC, marker_line_width=0, width=0.35))
        fig.add_trace(go.Bar(name='Perished',  x=[f'Class {i}' for i in cls.index], y=cls[0], marker_color=PC, marker_line_width=0, width=0.35))
        fig.update_layout(**PLOTLY, barmode='group', title='Class vs survival', height=300, legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#4a7a98')))
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = go.Figure()
        for s, col, lbl in [(1,SC,'Survived'),(0,PC,'Perished')]:
            fig.add_trace(go.Violin(y=df[df['Survived']==s]['Age'], name=lbl, line_color=col, opacity=0.7, box_visible=True, meanline_visible=True))
        fig.update_layout(**PLOTLY, title='Age distribution', height=300, violingap=0.25, legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#4a7a98')))
        st.plotly_chart(fig, use_container_width=True)
    fig = px.scatter(df, x='Age', y='Fare', color=df['Survived'].map({0:'Perished',1:'Survived'}), color_discrete_map={'Survived':SC,'Perished':PC}, opacity=0.6, title='Age vs Fare')
    fig.update_traces(marker=dict(size=5, line_width=0))
    fig.update_layout(**PLOTLY, height=320, legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#4a7a98')))
    st.plotly_chart(fig, use_container_width=True)

elif tab == 'models':
    st.markdown("<div style='font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:#1a5a7a;margin-bottom:1.4rem'>Model performance</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        vals = [v*100 for v in accs.values()]
        fig = go.Figure(go.Bar(x=list(accs.keys()), y=vals, marker_color=['#0d3a5a','#1a6a9a',SC], marker_line_width=0, text=[f'{v:.1f}%' for v in vals], textposition='outside', textfont=dict(family='Space Mono',size=13,color='#7ab8d8')))
        fig.update_layout(**PLOTLY, title='Accuracy comparison', yaxis_range=[0,108], height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure(go.Heatmap(z=cm, x=['Pred: Perished','Pred: Survived'], y=['Actual: Perished','Actual: Survived'], colorscale=[[0,'#03070d'],[0.4,'#071828'],[1,'#1a6a9a']], text=cm, texttemplate='<b>%{text}</b>', textfont=dict(size=24,family='Space Mono',color='#eaf0f6'), showscale=False))
        fig.update_layout(**PLOTLY, title='Confusion matrix', height=320)
        st.plotly_chart(fig, use_container_width=True)
    fi_s = fi.sort_values()
    fig = go.Figure(go.Bar(x=fi_s.values*100, y=fi_s.index, orientation='h', marker_color=['#0d3060','#0f3a70','#124480','#154e90','#1a5ca0','#1a65b0',SC], marker_line_width=0, text=[f'{v*100:.1f}%' for v in fi_s.values], textposition='outside', textfont=dict(family='Space Mono',size=11,color='#4a8aaa')))
    fig.update_layout(**PLOTLY, title='Feature importance', xaxis_title='Importance (%)', height=360)
    st.plotly_chart(fig, use_container_width=True)
    cr = pd.DataFrame(rpt).T.round(3)
    cr = cr[['precision','recall','f1-score']].drop('support', errors='ignore')
    st.dataframe(cr.style.background_gradient(cmap='Blues', axis=None).format('{:.3f}'), use_container_width=True)

elif tab == 'predict':
    st.markdown("<div style='font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:#1a5a7a;margin-bottom:1.4rem'>Passenger profile</div>", unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        pclass = st.selectbox("Passenger class", [1,2,3], index=2)
        sex    = st.selectbox("Gender", ["Male","Female"])
    with pc2:
        age   = st.slider("Age", 1, 80, 28)
        sibsp = st.slider("Siblings / spouses aboard", 0, 8, 0)
    with pc3:
        parch = st.slider("Parents / children aboard", 0, 6, 0)
        fare  = st.slider("Fare paid (GBP)", 0, 520, 32)
        emb   = st.selectbox("Embarked from", ["Southampton","Cherbourg","Queenstown"])
    emb_map = {"Southampton":2,"Cherbourg":0,"Queenstown":1}
    if st.button("Run prediction", use_container_width=True):
        inp   = pd.DataFrame([[pclass, 0 if sex=="Male" else 1, age, sibsp, parch, fare, emb_map[emb]]], columns=feat)
        pred  = rf.predict(inp)[0]
        proba = rf.predict_proba(inp)[0]
        conf  = proba[pred]*100
        col   = SC if pred==1 else PC
        label = "SURVIVED" if pred==1 else "PERISHED"
        bg    = "#061a12" if pred==1 else "#1a0606"
        bdr   = "#0f4a2e" if pred==1 else "#4a0f0f"
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {bdr};border-radius:12px;padding:1.6rem 2rem;display:flex;align-items:center;justify-content:space-between;margin-top:1rem">
          <div><div style="font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;color:#3a6a80">Prediction</div>
          <div style="font-family:'Space Mono',monospace;font-size:1.8rem;font-weight:700;color:{col};margin-top:.2rem">{label}</div></div>
          <div style="text-align:right"><div style="font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;color:#3a6a80">Confidence</div>
          <div style="font-family:'Space Mono',monospace;font-size:1.8rem;font-weight:700;color:{col};margin-top:.2rem">{conf:.1f}%</div></div>
        </div>
        <div style="margin-top:.8rem">
          <div style="background:#0a1a24;border-radius:4px;height:6px;overflow:hidden"><div style="height:100%;width:{int(proba[1]*100)}%;background:{SC};border-radius:4px"></div></div>
          <div style="display:flex;justify-content:space-between;font-size:.68rem;color:#2a5a78;margin-top:.35rem;font-family:'Space Mono',monospace"><span>Perished {proba[0]*100:.1f}%</span><span>Survived {proba[1]*100:.1f}%</span></div>
        </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)