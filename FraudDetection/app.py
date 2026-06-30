# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[class*="css"],.stApp{font-family:'Space Grotesk',sans-serif !important;background:#0d0a03 !important;color:#e0dcc8}
.block-container{padding:0 !important;max-width:100% !important}
#MainMenu,footer,header,[data-testid="stToolbar"]{visibility:hidden;height:0}
section[data-testid="stSidebar"]{display:none}
.stButton>button{background:#1a1600 !important;border:1px solid #6a5a00 !important;color:#ccaa00 !important;border-radius:8px !important;font-family:'Space Grotesk',sans-serif !important;font-size:.8rem !important;letter-spacing:.08em !important;padding:.6rem 1.4rem !important;transition:all .2s !important}
.stButton>button:hover{background:#2a2400 !important;border-color:#ccaa00 !important;color:#eeca20 !important}
[data-testid="stSelectbox"]>div>div{background:#1a1600 !important;border:1px solid #2e2800 !important;border-radius:8px !important;color:#e0dcc8 !important}
[data-testid="stFileUploadDropzone"]{background:#1a1600 !important;border:1px dashed #2e2800 !important;border-radius:16px !important;padding:1.5rem !important}
section[data-testid="stFileUploaderDropzone"] button{background:#1a1600 !important;border:1px solid #6a5a00 !important;color:#ccaa00 !important;border-radius:8px !important}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:#0d0a03}::-webkit-scrollbar-thumb{background:#2e2800;border-radius:2px}
</style>
""", unsafe_allow_html=True)

PLOTLY = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Space Grotesk, sans-serif', color='#7a6a30', size=11),
    margin=dict(l=8,r=8,t=36,b=8),
    xaxis=dict(gridcolor='#1e1a08', linecolor='#2e2800', tickcolor='#2e2800'),
    yaxis=dict(gridcolor='#1e1a08', linecolor='#2e2800', tickcolor='#2e2800'),
)
AC  = '#ccaa00'
FC  = '#cc4444'
GC  = '#44aa66'

@st.cache_data
def process(file):
    df = pd.read_csv(file, encoding='latin1')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ','_')
    fraud_col = next((c for c in df.columns if 'fraud' in c or 'class' in c), None)
    if fraud_col is None: st.error("No fraud/class column found."); st.stop()
    df = df.rename(columns={fraud_col:'fraud'})
    df['fraud'] = pd.to_numeric(df['fraud'], errors='coerce')
    df.dropna(subset=['fraud'], inplace=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat = [c for c in num_cols if c != 'fraud'][:20]
    X = df[feat].fillna(0)
    y = df['fraud']
    # undersample majority for balance
    fraud_idx    = df[df['fraud']==1].index
    genuine_idx  = df[df['fraud']==0].sample(n=min(len(fraud_idx)*10, len(df[df['fraud']==0])), random_state=42).index
    bal_idx      = fraud_idx.tolist() + genuine_idx.tolist()
    Xb, yb       = X.loc[bal_idx], y.loc[bal_idx]
    sc           = StandardScaler(); Xb_sc = sc.fit_transform(Xb)
    Xtr,Xte,ytr,yte = train_test_split(Xb_sc, yb, test_size=0.2, random_state=42, stratify=yb)
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42); rf.fit(Xtr,ytr)
    lr = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42);         lr.fit(Xtr,ytr)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42);                      gb.fit(Xtr,ytr)
    def mets(m):
        p=m.predict(Xte); pr=m.predict_proba(Xte)[:,1]
        rpt=classification_report(yte,p,output_dict=True)
        return {'Accuracy':accuracy_score(yte,p),'AUC':roc_auc_score(yte,pr),
                'Precision':rpt['1']['precision'],'Recall':rpt['1']['recall'],'F1':rpt['1']['f1-score']}
    results={'Random Forest':mets(rf),'Logistic Regression':mets(lr),'Gradient Boosting':mets(gb)}
    cm   = confusion_matrix(yte, rf.predict(Xte))
    rpt  = classification_report(yte, rf.predict(Xte), output_dict=True)
    fi   = pd.Series(rf.feature_importances_, index=feat).sort_values()
    fpr,tpr,_ = roc_curve(yte, rf.predict_proba(Xte)[:,1])
    return df, rf, sc, results, cm, rpt, fi, feat, fpr, tpr

st.markdown("""
<div style="background:#0d0a03;border-bottom:1px solid #2e2800;padding:2.2rem 3rem 1.8rem;position:relative;overflow:hidden">
  <div style="position:absolute;top:-80px;right:-80px;width:320px;height:320px;background:radial-gradient(circle,#3a300018 0%,transparent 70%);pointer-events:none"></div>
  <div style="font-size:.62rem;letter-spacing:.22em;text-transform:uppercase;color:#8a7020;margin-bottom:.6rem">CodSoft &nbsp;/&nbsp; Data Science &nbsp;/&nbsp; Task 05</div>
  <div style="font-size:2.4rem;font-weight:700;line-height:1.05;color:#f5f0dc;letter-spacing:-.03em;margin-bottom:.4rem">Credit Card<br><span style="color:#ccaa00">Fraud Detection</span></div>
  <div style="font-size:.82rem;color:#6a6030;line-height:1.6">Identifying fraudulent transactions using Random Forest, Logistic Regression &amp; Gradient Boosting.</div>
</div>
""", unsafe_allow_html=True)

_, mid, _ = st.columns([1,2,1])
with mid:
    st.markdown("""
    <div style="text-align:center;padding:1.8rem 0 .6rem">
      <div style="width:52px;height:52px;margin:0 auto 1rem;border:1px solid #2e2800;border-radius:12px;background:#0d0a03;display:flex;align-items:center;justify-content:center">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ccaa00" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <rect x="1" y="4" width="22" height="16" rx="2" ry="2"/><line x1="1" y1="10" x2="23" y2="10"/>
        </svg>
      </div>
      <div style="font-size:1rem;font-weight:600;color:#f5f0dc;margin-bottom:.3rem">Drop your dataset here</div>
      <div style="font-size:.78rem;color:#5a5020;margin-bottom:1rem">Upload <span style="color:#ccaa00;font-family:monospace">creditcard.csv</span> to begin</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("", type="csv", label_visibility="collapsed")

if uploaded is None:
    st.stop()

df, rf, sc, results, cm, rpt, fi, feat, fpr, tpr = process(uploaded)
n_fraud   = int(df['fraud'].sum())
n_genuine = int((df['fraud']==0).sum())
fraud_pct = n_fraud / len(df) * 100
rf_auc    = results['Random Forest']['AUC']

st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:#2e2800;border-bottom:1px solid #2e2800">
  <div style="background:#0d0a03;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#4a4010;margin-bottom:.35rem">Transactions</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#f5f0dc">{len(df):,}</div></div>
  <div style="background:#0d0a03;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#4a4010;margin-bottom:.35rem">Fraudulent</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#f5f0dc">{n_fraud:,}</div><div style="font-size:.7rem;color:{FC};margin-top:.15rem">{fraud_pct:.2f}% of total</div></div>
  <div style="background:#0d0a03;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#4a4010;margin-bottom:.35rem">Genuine</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#f5f0dc">{n_genuine:,}</div><div style="font-size:.7rem;color:{GC};margin-top:.15rem">{100-fraud_pct:.2f}% of total</div></div>
  <div style="background:#0d0a03;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#4a4010;margin-bottom:.35rem">RF AUC score</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#f5f0dc">{rf_auc:.3f}</div></div>
  <div style="background:#0d0a03;padding:1.2rem 2rem"><div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#4a4010;margin-bottom:.35rem">RF F1 (fraud)</div><div style="font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#f5f0dc">{results['Random Forest']['F1']:.3f}</div></div>
</div>
""", unsafe_allow_html=True)

if 'tab' not in st.session_state: st.session_state.tab = 'explore'
n1,n2,n3,_ = st.columns([1,1,1,6])
with n1:
    if st.button("📊  Explore", use_container_width=True): st.session_state.tab='explore'
with n2:
    if st.button("🤖  Models",  use_container_width=True): st.session_state.tab='models'
with n3:
    if st.button("🔍  Detect",  use_container_width=True): st.session_state.tab='detect'

tab = st.session_state.tab
st.markdown("<div style='padding:2rem 3rem;background:#0d0a03'>", unsafe_allow_html=True)

if tab == 'explore':
    st.markdown("<div style='font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:#8a7020;margin-bottom:1.4rem'>Transaction analysis</div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Pie(labels=['Genuine','Fraudulent'], values=[n_genuine,n_fraud],
                               hole=0.65, marker_colors=[GC,FC],
                               textinfo='percent+label', textfont=dict(size=12,family='Space Mono')))
        fig.add_annotation(text=f"<b>{fraud_pct:.1f}%</b><br>fraud", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16,color='#f5f0dc',family='Space Mono'))
        fig.update_layout(**PLOTLY, title='Class distribution', height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if 'amount' in df.columns:
            fig = go.Figure()
            for cls,col,lbl in [(0,GC,'Genuine'),(1,FC,'Fraudulent')]:
                fig.add_trace(go.Violin(y=df[df['fraud']==cls]['amount'].clip(upper=df['amount'].quantile(0.99)),
                                        name=lbl, line_color=col, opacity=0.7, box_visible=True, meanline_visible=True))
            fig.update_layout(**PLOTLY, title='Transaction amount by class', height=300,
                              legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#9a8a40')))
            st.plotly_chart(fig, use_container_width=True)
        else:
            if len(feat)>0:
                fig=go.Figure(go.Histogram(x=df[feat[0]], nbinsx=30, marker_color=AC, marker_line_width=0, opacity=0.85))
                fig.update_layout(**PLOTLY, title=f'{feat[0]} distribution', height=300)
                st.plotly_chart(fig, use_container_width=True)

elif tab == 'models':
    st.markdown("<div style='font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:#8a7020;margin-bottom:1.4rem'>Model performance</div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Heatmap(z=cm, x=['Pred: Genuine','Pred: Fraud'], y=['Actual: Genuine','Actual: Fraud'],
                                   colorscale=[[0,'#0d0a03'],[0.5,'#2e2400'],[1,AC]],
                                   text=cm, texttemplate='<b>%{text}</b>',
                                   textfont=dict(size=22,family='Space Mono',color='#f5f0dc'), showscale=False))
        fig.update_layout(**PLOTLY, title='Confusion matrix', height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color=AC, width=2), name='ROC curve'))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(color='#3a3010', dash='dash', width=1), showlegend=False))
        fig.update_layout(**PLOTLY, title=f'ROC curve (AUC={rf_auc:.3f})', height=320,
                          xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                          legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#9a8a40')))
        st.plotly_chart(fig, use_container_width=True)
    metrics_show = ['Accuracy','AUC','Precision','Recall','F1']
    met_colors = ['#1a1600','#2e2a00','#4a4000','#6a5a00',AC]
    for met,col in zip(metrics_show, met_colors):
        vals=[results[m][met] for m in results]
        fig=go.Figure(go.Bar(x=list(results.keys()),y=vals,marker_color=['#1a1600','#3a3000',AC],marker_line_width=0,text=[f'{v:.3f}' for v in vals],textposition='outside',textfont=dict(family='Space Mono',size=11,color=AC)))
        fig.update_layout(**PLOTLY,title=met,height=220)
        st.plotly_chart(fig,use_container_width=True)
        break
    fi_s=fi.tail(10)
    fig=go.Figure(go.Bar(x=fi_s.values*100,y=fi_s.index,orientation='h',marker_color=AC,marker_line_width=0,opacity=0.85,text=[f'{v*100:.1f}%' for v in fi_s.values],textposition='outside',textfont=dict(family='Space Mono',size=11,color=AC)))
    fig.update_layout(**PLOTLY,title='Top 10 feature importance',xaxis_title='Importance (%)',height=360)
    st.plotly_chart(fig,use_container_width=True)

elif tab == 'detect':
    st.markdown("<div style='font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:#8a7020;margin-bottom:1.4rem'>Transaction screening</div>", unsafe_allow_html=True)
    inp_vals={}
    show_feats = feat[:6]
    cols=st.columns(3)
    for i,f in enumerate(show_feats):
        with cols[i%3]:
            label=f.replace('_',' ').title()
            mn_v=float(df[f].quantile(0.01)); mx_v=float(df[f].quantile(0.99))
            default=float(df[f].median())
            inp_vals[f]=st.slider(label, mn_v, mx_v, default)
    for f in feat:
        if f not in inp_vals:
            inp_vals[f]=float(df[f].median())
    if st.button("Screen transaction", use_container_width=True):
        inp_df=pd.DataFrame([inp_vals])[feat]
        inp_sc=sc.transform(inp_df)
        pred=rf.predict(inp_sc)[0]
        proba=rf.predict_proba(inp_sc)[0]
        conf=proba[pred]*100
        is_fraud=pred==1
        col=FC if is_fraud else GC
        label='FRAUDULENT' if is_fraud else 'GENUINE'
        bg='#1a0606' if is_fraud else '#061a0a'
        bdr='#4a0f0f' if is_fraud else '#0f4a1e'
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {bdr};border-radius:12px;padding:1.6rem 2rem;margin-top:1rem;display:flex;align-items:center;justify-content:space-between">
          <div><div style="font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;color:#7a6a30">Transaction verdict</div>
          <div style="font-family:'Space Mono',monospace;font-size:1.8rem;font-weight:700;color:{col};margin-top:.2rem">{label}</div>
          <div style="font-size:.8rem;color:#5a5020;margin-top:.3rem">{'High risk — flag for review' if is_fraud else 'Low risk — transaction appears normal'}</div></div>
          <div style="text-align:right"><div style="font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;color:#7a6a30">Confidence</div>
          <div style="font-family:'Space Mono',monospace;font-size:1.8rem;font-weight:700;color:{col};margin-top:.2rem">{conf:.1f}%</div></div>
        </div>
        <div style="margin-top:.8rem">
          <div style="background:#1a1600;border-radius:4px;height:6px;overflow:hidden"><div style="height:100%;width:{int(proba[1]*100)}%;background:{FC};border-radius:4px"></div></div>
          <div style="display:flex;justify-content:space-between;font-size:.68rem;color:#4a3a10;margin-top:.35rem;font-family:'Space Mono',monospace"><span>Genuine {proba[0]*100:.1f}%</span><span>Fraud {proba[1]*100:.1f}%</span></div>
        </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)