# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Iris | Flower Classification", page_icon="🌸", layout="wide", initial_sidebar_state="collapsed")

COLORS = ['#7c3aed','#059669','#d97706']
C1,C2,C3 = COLORS

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[class*="css"],.stApp{font-family:'Space Grotesk',sans-serif !important;background:#07030d !important;color:#ddd6e0}
.block-container{padding:0 !important;max-width:100% !important}
#MainMenu,footer,header,[data-testid="stToolbar"]{visibility:hidden;height:0}
section[data-testid="stSidebar"]{display:none}
.stButton>button{background:#120720 !important;border:1px solid #4a1a6a !important;color:#c07ad8 !important;border-radius:8px !important;font-family:'Space Grotesk',sans-serif !important;font-size:.8rem !important;letter-spacing:.08em !important;padding:.6rem 1.4rem !important;transition:all .2s !important}
.stButton>button:hover{background:#1e0a38 !important;border-color:#8a3aaa !important;color:#e0aaee !important}
[data-testid="stSelectbox"]>div>div{background:#0e0718 !important;border:1px solid #2e1a3e !important;border-radius:8px !important;color:#ddd6e0 !important}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:#07030d}::-webkit-scrollbar-thumb{background:#1e0a38;border-radius:2px}
</style>
""", unsafe_allow_html=True)

PLOTLY = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Space Grotesk, sans-serif', color='#7a5a88', size=11),
    margin=dict(l=8,r=8,t=36,b=8),
    xaxis=dict(gridcolor='#1a0a24', linecolor='#2e1a3e', tickcolor='#2e1a3e'),
    yaxis=dict(gridcolor='#1a0a24', linecolor='#2e1a3e', tickcolor='#2e1a3e'),
)

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=['sepal_length','sepal_width','petal_length','petal_width'])
    df['species'] = [iris.target_names[i] for i in iris.target]
    df['target']  = iris.target
    X, y = df[['sepal_length','sepal_width','petal_length','petal_width']], df['target']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf  = RandomForestClassifier(n_estimators=200, random_state=42); rf.fit(Xtr, ytr)
    svm = SVC(probability=True, kernel='rbf', random_state=42);      svm.fit(Xtr, ytr)
    knn = KNeighborsClassifier(n_neighbors=5);                        knn.fit(Xtr, ytr)
    accs = {
        'Random Forest': accuracy_score(yte, rf.predict(Xte)),
        'SVM':           accuracy_score(yte, svm.predict(Xte)),
        'KNN':           accuracy_score(yte, knn.predict(Xte)),
    }
    cm  = confusion_matrix(yte, rf.predict(Xte))
    rpt = classification_report(yte, rf.predict(Xte), target_names=iris.target_names, output_dict=True)
    fi  = pd.Series(rf.feature_importances_, index=['Sepal Length','Sepal Width','Petal Length','Petal Width'])
    return df, rf, accs, cm, rpt, fi, iris.target_names

df, rf, accs, cm, rpt, fi, species_names = load_data()

# HERO
st.markdown("""
<div style="background:#07030d;border-bottom:1px solid #2e1a3e;padding:2.2rem 3rem 1.8rem;position:relative;overflow:hidden">
  <div style="position:absolute;top:-80px;right:-80px;width:320px;height:320px;background:radial-gradient(circle,#4a1a6e18 0%,transparent 70%);pointer-events:none"></div>
  <div style="font-size:.62rem;letter-spacing:.22em;text-transform:uppercase;color:#6a3a8a;margin-bottom:.6rem">CodSoft &nbsp;/&nbsp; Data Science &nbsp;/&nbsp; Task 03</div>
  <div style="font-size:2.4rem;font-weight:700;line-height:1.05;color:#f0eaf6;letter-spacing:-.03em;margin-bottom:.4rem">Iris Flower<br><span style="color:#8a3aaa">Classification</span></div>
  <div style="font-size:.82rem;color:#6a4a80;line-height:1.6">Classifying 3 Iris species from petal &amp; sepal measurements — Random Forest, SVM &amp; KNN.</div>
</div>
""", unsafe_allow_html=True)

# KPI
rf_acc = accs['Random Forest']
st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:#2e1a3e;border-bottom:1px solid #2e1a3e">
  <div style="background:#07030d;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#4a2a60;margin-bottom:.35rem">Total flowers</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#f0eaf6">150</div></div>
  <div style="background:#07030d;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#4a2a60;margin-bottom:.35rem">Species</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#f0eaf6">3</div><div style="font-size:.7rem;color:#7c3aed;margin-top:.15rem">50 samples each</div></div>
  <div style="background:#07030d;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#4a2a60;margin-bottom:.35rem">Features</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#f0eaf6">4</div><div style="font-size:.7rem;color:#059669;margin-top:.15rem">sepal &amp; petal</div></div>
  <div style="background:#07030d;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#4a2a60;margin-bottom:.35rem">RF accuracy</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#f0eaf6">{rf_acc*100:.1f}%</div></div>
  <div style="background:#07030d;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#4a2a60;margin-bottom:.35rem">Best model</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#f0eaf6">{max(accs,key=accs.get)[:2].upper()}</div></div>
</div>
""", unsafe_allow_html=True)

if 'tab' not in st.session_state: st.session_state.tab = 'explore'
n1,n2,n3,_ = st.columns([1,1,1,6])
with n1:
    if st.button("🌸  Explore",  use_container_width=True): st.session_state.tab='explore'
with n2:
    if st.button("🤖  Models",   use_container_width=True): st.session_state.tab='models'
with n3:
    if st.button("🔮  Classify", use_container_width=True): st.session_state.tab='classify'

tab = st.session_state.tab
st.markdown("<div style='padding:2rem 3rem;background:#07030d'>", unsafe_allow_html=True)

if tab == 'explore':
    st.markdown("<div style='font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:#5a2a7a;margin-bottom:1.4rem'>Species exploration</div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        fig = px.scatter(df, x='petal_length', y='petal_width', color='species',
                         color_discrete_sequence=COLORS, title='Petal length vs petal width',
                         labels={'petal_length':'Petal Length (cm)','petal_width':'Petal Width (cm)'})
        fig.update_traces(marker=dict(size=7, line_width=0), opacity=0.85)
        fig.update_layout(**PLOTLY, height=320, legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8a6a98')))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(df, x='sepal_length', y='sepal_width', color='species',
                         color_discrete_sequence=COLORS, title='Sepal length vs sepal width',
                         labels={'sepal_length':'Sepal Length (cm)','sepal_width':'Sepal Width (cm)'})
        fig.update_traces(marker=dict(size=7, line_width=0), opacity=0.85)
        fig.update_layout(**PLOTLY, height=320, legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8a6a98')))
        st.plotly_chart(fig, use_container_width=True)
    c3,c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        for sp, col in zip(species_names, COLORS):
            fig.add_trace(go.Violin(y=df[df['species']==sp]['petal_length'], name=sp,
                                    line_color=col, opacity=0.75, box_visible=True, meanline_visible=True))
        fig.update_layout(**PLOTLY, title='Petal length distribution', height=300,
                          legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8a6a98')))
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        means = df.groupby('species')[['sepal_length','sepal_width','petal_length','petal_width']].mean()
        fig = go.Figure()
        for sp, col in zip(species_names, COLORS):
            fig.add_trace(go.Bar(name=sp, x=['Sepal L','Sepal W','Petal L','Petal W'],
                                 y=means.loc[sp], marker_color=col, marker_line_width=0, opacity=0.85))
        fig.update_layout(**PLOTLY, barmode='group', title='Average measurements by species', height=300,
                          legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8a6a98')))
        st.plotly_chart(fig, use_container_width=True)
    fig = px.scatter_matrix(df, dimensions=['sepal_length','sepal_width','petal_length','petal_width'],
                            color='species', color_discrete_sequence=COLORS, title='Feature pair plot')
    fig.update_traces(marker=dict(size=3, line_width=0), diagonal_visible=False, opacity=0.7)
    fig.update_layout(**PLOTLY, height=500)
    st.plotly_chart(fig, use_container_width=True)

elif tab == 'models':
    st.markdown("<div style='font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:#5a2a7a;margin-bottom:1.4rem'>Model performance</div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        vals = [v*100 for v in accs.values()]
        fig = go.Figure(go.Bar(x=list(accs.keys()), y=vals, marker_color=['#4a1a6a','#7c3aed','#a06aed'],
                               marker_line_width=0, text=[f'{v:.1f}%' for v in vals], textposition='outside',
                               textfont=dict(family='Space Mono',size=13,color='#c07ad8')))
        fig.update_layout(**PLOTLY, title='Accuracy comparison', yaxis_range=[0,108], height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure(go.Heatmap(z=cm, x=list(species_names), y=list(species_names),
                                   colorscale=[[0,'#07030d'],[0.5,'#1e0a38'],[1,'#7c3aed']],
                                   text=cm, texttemplate='<b>%{text}</b>',
                                   textfont=dict(size=22,family='Space Mono',color='#f0eaf6'), showscale=False))
        fig.update_layout(**PLOTLY, title='Confusion matrix', height=320)
        st.plotly_chart(fig, use_container_width=True)
    fi_s = fi.sort_values()
    fig = go.Figure(go.Bar(x=fi_s.values*100, y=fi_s.index, orientation='h',
                           marker_color=['#2a0a4a','#3a1060','#5a2080','#7c3aed'],
                           marker_line_width=0, text=[f'{v*100:.1f}%' for v in fi_s.values],
                           textposition='outside', textfont=dict(family='Space Mono',size=11,color='#a07ab8')))
    fig.update_layout(**PLOTLY, title='Feature importance', xaxis_title='Importance (%)', height=300)
    st.plotly_chart(fig, use_container_width=True)
    cr = pd.DataFrame(rpt).T.round(3)
    cr = cr[['precision','recall','f1-score']].drop('support', errors='ignore').dropna()
    st.dataframe(cr.style.background_gradient(cmap='Purples', axis=None).format('{:.3f}'), use_container_width=True)

elif tab == 'classify':
    st.markdown("<div style='font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:#5a2a7a;margin-bottom:1.4rem'>Classify a flower</div>", unsafe_allow_html=True)
    cc1,cc2 = st.columns(2)
    with cc1:
        sl = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, step=0.1)
        sw = st.slider("Sepal width (cm)",  2.0, 4.5, 3.0, step=0.1)
    with cc2:
        pl = st.slider("Petal length (cm)", 1.0, 7.0, 4.0, step=0.1)
        pw = st.slider("Petal width (cm)",  0.1, 2.5, 1.2, step=0.1)
    if st.button("Classify flower", use_container_width=True):
        inp   = pd.DataFrame([[sl,sw,pl,pw]], columns=['sepal_length','sepal_width','petal_length','petal_width'])
        pred  = rf.predict(inp)[0]
        proba = rf.predict_proba(inp)[0]
        sp    = species_names[pred]
        col   = COLORS[pred]
        conf  = proba[pred]*100
        sp_info = {
            'setosa':     ('Iris Setosa',     'Small flowers with broad sepals. Found in Arctic regions.'),
            'versicolor': ('Iris Versicolor', 'Medium-sized flowers. Common in North America.'),
            'virginica':  ('Iris Virginica',  'Largest of the three species. Found in eastern North America.'),
        }
        name, desc = sp_info[sp]
        st.markdown(f"""
        <div style="background:#0e0718;border:1px solid {col}44;border-radius:12px;padding:1.6rem 2rem;margin-top:1rem">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem">
            <div><div style="font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;color:#6a4a80">Predicted species</div>
            <div style="font-family:'Space Mono',monospace;font-size:1.8rem;font-weight:700;color:{col};margin-top:.2rem">{name}</div>
            <div style="font-size:.8rem;color:#7a5a90;margin-top:.3rem">{desc}</div></div>
            <div style="text-align:right"><div style="font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;color:#6a4a80">Confidence</div>
            <div style="font-family:'Space Mono',monospace;font-size:1.8rem;font-weight:700;color:{col};margin-top:.2rem">{conf:.1f}%</div></div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem">
            {''.join([f'<div style="background:#07030d;border-radius:6px;padding:.6rem .8rem;border:1px solid #2e1a3e"><div style="font-size:.6rem;color:#5a3a70;letter-spacing:.1em;text-transform:uppercase">{species_names[i]}</div><div style="font-family:monospace;font-size:.9rem;color:{COLORS[i]};margin-top:.2rem">{proba[i]*100:.1f}%</div></div>' for i in range(3)])}
          </div>
        </div>
        """, unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=list(species_names), y=proba*100, marker_color=COLORS,
                               marker_line_width=0, text=[f'{p*100:.1f}%' for p in proba],
                               textposition='outside', textfont=dict(family='Space Mono',size=13,color='#c0a0d8')))
        fig.update_layout(**PLOTLY, title='Classification probabilities', yaxis_range=[0,115], height=280)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)