"""
성균관대 1조 - GNN 기반 조직적 어뷰징 네트워크 탐지 대시보드
실행: streamlit run 성균관대1조_분석코드.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import networkx as nx
import os, pathlib

# ──────────────────────────────────────────────
# 0. 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="GNN 사기 리뷰 탐지 | 성균관대 1조",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# 1. 글로벌 CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
  .kpi-card {
      background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
      border: 1px solid #3a3f5c;
      border-radius: 14px;
      padding: 20px 24px;
      text-align: center;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
  }
  .kpi-label { color: #8b9ab8; font-size: 0.82rem; font-weight: 600;
               text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
  .kpi-value { color: #e8eaf6; font-size: 2rem; font-weight: 800; line-height: 1.1; }
  .kpi-sub   { color: #6c7a9c; font-size: 0.76rem; margin-top: 4px; }
  .section-header {
      border-left: 4px solid #5c6bc0;
      padding-left: 12px;
      margin: 24px 0 12px 0;
      font-size: 1.2rem; font-weight: 700; color: #c5cae9;
  }
  section[data-testid="stSidebar"] {
      background: #13151f;
      border-right: 1px solid #2a2f47;
  }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 2. 데이터 로드 (캐시)
# ──────────────────────────────────────────────
BASE = pathlib.Path(__file__).parent

@st.cache_data
def load_graph_meta():
    p = BASE / "graph_meta.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {
        "n_nodes": 36157, "n_features": 110,
        "feature_blocks": {"basic": 22, "agg": 17, "text": 64, "graph": 7},
        "edges": {"review__rur__review": 23488, "review__burst__review": 84,
                  "review__sburst__review": 27190, "review__pday__review": 584,
                  "review__upur__review": 12000, "review__sim__review": 3324},
        "fraud_ratio": 0.1537,
        "sampling": {"start": "2013-01-01", "end": "2014-12-31", "top_n_prod": 30},
    }

@st.cache_data
def load_results():
    results = {}
    for fname in ["results_stage8.json", "results_stage8_ablation.json",
                  "results_care_gnn_test.json"]:
        p = BASE / fname
        if p.exists():
            with open(p) as f:
                results[fname] = json.load(f)
    csv_p = BASE / "results_final_table.csv"
    results["final_table"] = pd.read_csv(csv_p) if csv_p.exists() else None
    return results

@st.cache_data
def load_subgraph():
    p = BASE / "subgraph_sample.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None

@st.cache_data
def load_embeddings():
    p = BASE / "text_embeddings_bge_64d.npy"
    if p.exists():
        return np.load(str(p))
    return None

@st.cache_data
def load_agg_yearly():
    """연도별 × 라벨별 건수 집계 (agg_yearly_counts.csv)"""
    p = BASE / "agg_yearly_counts.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)   # columns: year, label, count

@st.cache_data
def load_agg_rating():
    """별점 × 라벨별 건수 집계 (agg_rating_dist.csv)"""
    p = BASE / "agg_rating_dist.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)   # columns: rating, label, count

@st.cache_data
def load_agg_prod():
    """식당별 통계 집계 (agg_prod_stats.csv)"""
    p = BASE / "agg_prod_stats.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)   # columns: prod_id, total, fraud, avg_rating, fraud_rate

meta      = load_graph_meta()
results   = load_results()
df_raw    = load_subgraph()
agg_yearly = load_agg_yearly()
agg_rating = load_agg_rating()
agg_prod   = load_agg_prod()
embeddings = load_embeddings()

# ──────────────────────────────────────────────
# 3. 사이드바
# ──────────────────────────────────────────────
PAGES = {
    "🏠  프로젝트 개요":      "home",
    "🕵️  EDA":               "fraud",
    "🔗  그래프 구조 분석":   "graph",
    "🤖  모델 성능 비교":     "model",
}

with st.sidebar:
    st.markdown("## 🕵️ GNN 사기 탐지")
    st.markdown("**성균관대학교 1조**")
    st.markdown("---")
    page_label = st.radio("페이지 선택", list(PAGES.keys()), label_visibility="collapsed")
    page = PAGES[page_label]
    st.markdown("---")
    st.caption("📊 YelpZip 데이터셋 기반\n\n🧠 RGCN × HeteroSAGE+ 앙상블")

# ──────────────────────────────────────────────
# 공통 색상
# ──────────────────────────────────────────────
C_FRAUD  = "#ff5252"
C_NORMAL = "#4caf50"
C_ACCENT = "#5c6bc0"
C_WARN   = "#ff9800"

EDGE_LABELS = [
    "R-U-R (동일 유저)",
    "Burst Star (집중 폭발)",
    "Short Burst (단기 폭발)",
    "P-Day (식당×일 집중)",
    "U-P-U (유저-식당 공유)",
    "Sim (텍스트 유사도)",
]
EDGE_COLORS = ["#5c6bc0", "#ff5252", "#ff9800", "#26a69a", "#ab47bc", "#42a5f5"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font_color="#c5cae9",
    margin=dict(l=20, r=20, t=40, b=20),
)

# ╔══════════════════════════════════════════════╗
# ║  PAGE 1: 홈                                  ║
# ╚══════════════════════════════════════════════╝
if page == "home":
    st.markdown("# 🕵️ GNN 기반 조직적 어뷰징 네트워크 탐지")
    st.markdown(
        "YelpZip 리뷰 데이터를 **그래프 신경망(GNN)** 으로 분석하여 "
        "조직적 사기 리뷰 네트워크를 탐지합니다."
    )
    st.markdown("---")

    total_n  = meta["n_nodes"]
    fraud_r  = meta["fraud_ratio"] * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "총 리뷰(노드)", f"{total_n:,}", "서브그래프 내"),
        (c2, "사기 리뷰 비율", f"{fraud_r:.1f}%", "5,557개"),
        (c3, "총 엣지(관계선)", f"{sum(meta['edges'].values()):,}", "6종 관계"),
        (c4, "노드 피처 수", str(meta["n_features"]), "4개 블록"),
        (c5, "최고 PR-AUC", "0.504", "HeteroSAGE+ 앙상블"),
    ]
    for col, label, val, sub in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1.1, 1])

    with col_l:
        st.markdown('<div class="section-header">📌 프로젝트 파이프라인</div>',
                    unsafe_allow_html=True)
        steps = [
            ("①", "데이터 수집 & 전처리",
             "YelpZip 60만 건 → 상위 30개 식당 × 2013-14년 서브그래프 추출"),
            ("②", "그래프 구조 설계",
             "6가지 엣지: RUR, Burst, ShortBurst, PDay, UPUR, TextSim"),
            ("③", "피처 엔지니어링",
             "110차원 피처: 기본(22) + 집계(17) + 텍스트(64) + 그래프(7)"),
            ("④", "GNN 모델링",
             "RGCN, HeteroSAGE+, GraphSAGE, GAT, GCN, HAN 비교 실험"),
            ("⑤", "앙상블 & 평가",
             "3-seed 앙상블 → PR-AUC 0.504, Macro-F1 0.680 달성"),
            ("⑥", "시각화 대시보드",
             "Streamlit 기반 인터랙티브 대시보드 구현"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex; gap:12px; align-items:flex-start;
                        background:#1a1d2e; border-radius:10px; padding:12px 16px;
                        margin-bottom:8px; border-left:3px solid {C_ACCENT};">
              <span style="color:{C_ACCENT}; font-size:1.3rem; font-weight:800;
                           min-width:28px;">{num}</span>
              <div>
                <div style="color:#c5cae9; font-weight:700; font-size:0.95rem;">{title}</div>
                <div style="color:#6c7a9c; font-size:0.82rem; margin-top:2px;">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-header">🎯 GNN이 필요한 이유</div>',
                    unsafe_allow_html=True)
        reasons = [
            ("🔗", "조직적 패턴 포착",    C_FRAUD,
             "사기꾼은 '텍스트'를 속여도\n'관계 구조'는 숨기기 어렵습니다."),
            ("📡", "정보 전파 학습",       C_ACCENT,
             "의심 리뷰 주변 노드에\n사기 확률이 전파됩니다."),
            ("🛡️", "변칙 수법 대응",      C_WARN,
             "새로운 패턴도 엣지 설계로\n유연하게 대응합니다."),
        ]
        for icon, title, color, desc in reasons:
            st.markdown(f"""
            <div style="background:#1a1d2e; border-radius:10px; padding:16px 18px;
                        margin-bottom:10px; border:1px solid #2a2f47;">
              <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                <span style="font-size:1.4rem;">{icon}</span>
                <span style="color:{color}; font-weight:700;">{title}</span>
              </div>
              <div style="color:#8b9ab8; font-size:0.85rem; white-space:pre-line;">{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">📅 샘플링 조건</div>',
                    unsafe_allow_html=True)
        samp = meta.get("sampling", {})
        st.markdown(f"""
        <div style="background:#1a1d2e; border-radius:10px; padding:14px 18px;
                    border:1px solid #2a2f47; font-size:0.85rem; color:#8b9ab8;">
          📆 &nbsp;기간: <b style="color:#c5cae9;">{samp.get('start','2013-01-01')} ~ {samp.get('end','2014-12-31')}</b><br>
          🏪 &nbsp;상위 식당: <b style="color:#c5cae9;">Top {samp.get('top_n_prod', 30)}개</b> 식당 중심 추출<br>
          📦 &nbsp;노드 수: <b style="color:#c5cae9;">{total_n:,}개</b> (1만~5만 기준 충족)<br>
          ⚖️ &nbsp;분할 비율: <b style="color:#c5cae9;">Train 80% / Test 20%</b>
        </div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════╗
# ║  PAGE 2: 그래프 구조 분석                    ║
# ╚══════════════════════════════════════════════╝
elif page == "graph":
    st.markdown("# 🔗 그래프 구조 분석")
    st.markdown("사기 리뷰 탐지를 위해 설계된 **6종 엣지**와 **110차원 피처** 구조를 분석합니다.")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 엣지 분포", "🧩 피처 블록", "👤 유저 분석", "🕸️ 그래프 구조도"])

    with tab1:
        edge_values = list(meta["edges"].values())
        col_chart, col_info = st.columns([1.4, 1])

        with col_chart:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=EDGE_LABELS, y=edge_values,
                marker_color=EDGE_COLORS,
                text=[f"{v:,}" for v in edge_values],
                textposition="outside",
                textfont=dict(color="#c5cae9", size=12),
                hovertemplate="<b>%{x}</b><br>엣지 수: %{y:,}<extra></extra>",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title="엣지 타입별 연결 수",
                              xaxis=dict(tickangle=-15, tickfont=dict(size=10)),
                              yaxis=dict(title="엣지 수", gridcolor="#2a2f47"),
                              showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col_info:
            st.markdown('<div class="section-header">🔍 엣지 설계 근거</div>',
                        unsafe_allow_html=True)
            edge_descs = [
                ("R-U-R (동일 유저)",       EDGE_COLORS[0],
                 "같은 유저가 작성한 리뷰끼리 연결\n→ 다계정 작업장 탐지"),
                ("Burst Star (집중 폭발)",         EDGE_COLORS[1],
                 "1일 내 15건+ 같은 별점 집중\n→ 급격한 평점 조작 포착"),
                ("Short Burst (단기 폭발)",   EDGE_COLORS[2],
                 "2일 내 5건+, 텍스트<200자\n→ 소규모 조직 탐지"),
                ("P-Day (식당×시간 집중)",      EDGE_COLORS[3],
                 "같은 식당·같은 날 20건+ (별점/텍스트 무관)\n→ 이상 집중 패턴 포착"),
                ("U-P-U (유저-식당 공유)",  EDGE_COLORS[4],
                 "사기 비율 높은 식당 공유 유저\n→ 조직적 협업 탐지"),
                ("Sim (텍스트 유사도)",        EDGE_COLORS[5],
                 "BGE 임베딩 코사인 유사도\n→ 복붙·템플릿 리뷰 탐지"),
            ]
            for name, color, desc in edge_descs:
                st.markdown(f"""
                <div style="display:flex; gap:10px; align-items:flex-start;
                            padding:8px 12px; border-radius:8px; margin-bottom:6px;
                            background:#1a1d2e; border-left:3px solid {color};">
                  <div>
                    <b style="color:{color}; font-size:0.82rem;">{name}</b><br>
                    <span style="color:#8b9ab8; font-size:0.78rem;
                                 white-space:pre-line;">{desc}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

    with tab2:
        col_pie, col_detail = st.columns([1, 1.2])

        with col_pie:
            fb = meta["feature_blocks"]
            fb_labels = ["기본 (Basic)", "집계 (Agg)", "텍스트 (Text)", "그래프 (Graph)"]
            fb_values = [fb["basic"], fb["agg"], fb["text"], fb["graph"]]
            fb_colors = [C_ACCENT, C_WARN, "#42a5f5", C_FRAUD]
            fig2 = go.Figure(go.Pie(
                labels=fb_labels, values=fb_values,
                marker_colors=fb_colors, hole=0.52,
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>%{value}차원 (%{percent})<extra></extra>",
            ))
            fig2.add_annotation(text=f"<b>110</b><br>차원", x=0.5, y=0.5,
                                font_size=18, font_color="#e8eaf6", showarrow=False)
            fig2.update_layout(**PLOTLY_LAYOUT, title="노드 피처 블록 구성",
                               height=360, showlegend=True,
                               legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig2, use_container_width=True)

        with col_detail:
            st.markdown('<div class="section-header">📋 피처 블록 상세</div>',
                        unsafe_allow_html=True)
            feat_info = [
                ("기본 (Basic)", "22차원", C_ACCENT,
                 ["rating", "txt_len", "date 파생 (연/월/일/요일)",
                  "유저 리뷰 수/평균별점", "식당 리뷰 수/평균별점", "유저×식당 동시 리뷰 여부"]),
                ("집계 (Agg)", "17차원", C_WARN,
                 ["유저별 사기 이웃 비율", "식당별 사기 비율",
                  "엣지 타입별 연결 수 (6종)", "이웃 평균 별점/텍스트 길이"]),
                ("텍스트 (Text)", "64차원", "#42a5f5",
                 ["BGE-m3 임베딩 → PCA 64차원 축소",
                  "리뷰 원본 텍스트 의미 벡터",
                  "코사인 유사도 기반 엣지 생성에도 활용"]),
                ("그래프 (Graph)", "7차원", C_FRAUD,
                 ["in_burst_strict", "in_short_burst", "in_prodday", "in_upur",
                  "user_min_interval_log_z", "user_std_interval_log_z",
                  "user_max_reviews_per_day_z"]),
            ]
            for name, dim, color, items in feat_info:
                with st.expander(f"**{name}** — {dim}", expanded=False):
                    for item in items:
                        st.markdown(f"- `{item}`")

        # Rating 분포
    with tab3:
        if df_raw is None:
            st.warning("subgraph_sample.parquet 파일이 없습니다.")
        else:
            df = df_raw.copy()
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year
            user_stats = df.groupby("user_id").agg(
                total=("label","count"), fraud=("label","sum"),
                avg_rating=("rating","mean")).reset_index()
            user_stats["fraud_rate"] = user_stats["fraud"] / user_stats["total"]
            user_stats = user_stats[user_stats["total"] >= 3].sort_values("fraud_rate", ascending=False)

            col_u1, col_u2 = st.columns(2)
            with col_u1:
                st.markdown('<div class="section-header">🚨 고위험 유저 TOP 20</div>',
                            unsafe_allow_html=True)
                top20 = user_stats.head(20)
                fig_u = go.Figure(go.Bar(
                    y=[f"유저 {uid}" for uid in top20["user_id"]],
                    x=top20["fraud_rate"] * 100, orientation="h",
                    marker=dict(color=top20["fraud_rate"],
                                colorscale=[[0,"#ff9800"],[1,"#ff1744"]],
                                showscale=True, colorbar=dict(title="%")),
                    text=[f"{r:.0f}%" for r in top20["fraud_rate"]*100],
                    textposition="outside",
                    hovertemplate="유저 %{y}<br>사기 비율: %{x:.1f}%<extra></extra>",
                ))
                fig_u.update_layout(**PLOTLY_LAYOUT,
                                    xaxis=dict(title="사기 리뷰 비율 (%)", gridcolor="#2a2f47"),
                                    yaxis=dict(autorange="reversed"), height=480)
                st.plotly_chart(fig_u, use_container_width=True)

            with col_u2:
                st.markdown('<div class="section-header">📊 리뷰 수 vs 사기 비율</div>',
                            unsafe_allow_html=True)
                samp_u = user_stats.sample(min(300, len(user_stats)), random_state=42)
                fig_u2 = px.scatter(samp_u, x="total", y="fraud_rate",
                                    color="fraud_rate",
                                    color_continuous_scale=[[0,C_NORMAL],[0.3,C_WARN],[1,C_FRAUD]],
                                    size="total", size_max=20,
                                    hover_data={"user_id":True,"total":True,"fraud":True},
                                    labels={"total":"총 리뷰 수","fraud_rate":"사기 비율"})
                fig_u2.update_layout(**PLOTLY_LAYOUT, height=480, coloraxis_showscale=False)
                st.plotly_chart(fig_u2, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            for col, lbl, val in [
                (c1, "분석 유저 수",   f"{len(user_stats):,}명"),
                (c2, "100% 사기 유저", f"{(user_stats.fraud_rate==1).sum():,}명"),
                (c3, "평균 리뷰 수",   f"{user_stats.total.mean():.1f}건"),
                (c4, "고위험 유저",    f"{(user_stats.fraud_rate >= 0.8).sum():,}명"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="kpi-card">
                      <div class="kpi-label">{lbl}</div>
                      <div class="kpi-value">{val}</div>
                    </div>""", unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="section-header">🕸️ 6종 엣지 기반 사기 네트워크 (인터랙티브)</div>',
                    unsafe_allow_html=True)

        ETYPE_META = {
            'rur':    ('R-U-R (동일 유저)',      '#5c6bc0', '같은 유저가 작성한 리뷰 연결'),
            'burst':  ('Burst Star (집중 폭발)',       '#ff5252', '1일 내 8건+ 집중 리뷰 연결 (시각화용 완화)'),
            'sburst': ('Short Burst (단기 폭발)', '#ff9800', '2일 내 4건+ 집중 리뷰 연결 (시각화용 완화)'),
            'pday':   ('P-Day (상품×일 집중)',    '#26a69a', '같은 상품·같은 날 10건+ 연결 (시각화용 완화)'),
            'upur':   ('U-P-U (유저-상품 공유)','#ab47bc', '사기 비율 높은 상품 공유 유저 연결'),
            'sim':    ('Sim (텍스트 유사도)',      '#42a5f5', 'BGE 임베딩 코사인 유사도 ≥0.80 연결'),
        }

        # 엣지 표시 선택
        col_ctrl1, col_ctrl2 = st.columns([1.5, 1])
        with col_ctrl1:
            show_etypes = st.multiselect(
                "표시할 엣지 타입 선택",
                options=list(ETYPE_META.keys()),
                default=list(ETYPE_META.keys()),
                format_func=lambda k: ETYPE_META[k][0],
            )
        with col_ctrl2:
            layout_algo = st.selectbox("레이아웃", ["spring", "kamada_kawai"], index=0)

        if df_raw is not None and embeddings is not None:
            # ── 샘플링 ──
            _df = df_raw.copy()
            _df['date']      = pd.to_datetime(_df['date'])
            _df['date_only'] = _df['date'].dt.date
            _df = _df.reset_index(drop=True)

            np.random.seed(42)
            core_prods = [3745, 4698, 3237]
            day_cnt    = _df.groupby(['prod_id','date_only']).size().reset_index(name='cnt')
            top_days   = day_cnt.nlargest(10, 'cnt')
            burst_idx  = []
            for _, row in top_days.iterrows():
                burst_idx.extend(
                    _df[(_df.prod_id==row.prod_id) & (_df.date_only==row.date_only)].index.tolist()
                )
            burst_idx = list(set(burst_idx))

            multi_users = (_df[_df.prod_id.isin(core_prods)]
                           .groupby('user_id').size()
                           .pipe(lambda s: s[s >= 2]).index.tolist())
            rur_idx = _df[(_df.user_id.isin(multi_users)) &
                          (_df.prod_id.isin(core_prods))].index.tolist()

            fraud_pool  = _df[(_df.label==1) & (_df.prod_id.isin(core_prods))].index.tolist()
            normal_pool = _df[(_df.label==0) & (_df.prod_id.isin(core_prods))].index.tolist()
            fraud_sel   = np.random.choice(fraud_pool,  min(80, len(fraud_pool)),  replace=False).tolist()
            normal_sel  = np.random.choice(normal_pool, min(80, len(normal_pool)), replace=False).tolist()

            sample_idx = sorted(set(burst_idx + rur_idx[:60] + fraud_sel + normal_sel))[:300]
            sdf = _df.loc[sample_idx].copy().reset_index(drop=True)
            sdf['orig_idx'] = sample_idx

            # ── 6개 엣지 구성 ──
            from itertools import combinations as _comb
            all_edges = {t: [] for t in ETYPE_META}

            for uid, grp in sdf.groupby('user_id'):
                idxs = grp.index.tolist()
                if len(idxs) > 1:
                    for a, b in _comb(idxs, 2): all_edges['rur'].append((a,b))

            for (pid, d), grp in sdf.groupby(['prod_id','date_only']):
                n = len(grp); idxs = grp.index.tolist()
                if n >= 8:
                    for a, b in _comb(idxs, 2): all_edges['burst'].append((a,b))
                if n >= 10:
                    for a, b in _comb(idxs, 2): all_edges['pday'].append((a,b))

            for pid, pgrp in sdf.groupby('prod_id'):
                pgrp = pgrp.sort_values('date').reset_index()
                for i in range(len(pgrp)):
                    win = pgrp[(pgrp.date >= pgrp.iloc[i].date) &
                               (pgrp.date <= pgrp.iloc[i].date + pd.Timedelta(days=2))]
                    if len(win) >= 4:
                        for a, b in list(_comb(win['index'].tolist(), 2))[:8]:
                            all_edges['sburst'].append((a,b))
            all_edges['sburst'] = list(set(all_edges['sburst']))

            prod_fr = sdf.groupby('prod_id')['label'].mean()
            for pid in prod_fr[prod_fr >= 0.2].index:
                users = sdf[sdf.prod_id==pid]['user_id'].unique()
                for u1, u2 in list(_comb(users, 2))[:20]:
                    r1 = sdf[(sdf.user_id==u1)&(sdf.prod_id==pid)].index.tolist()
                    r2 = sdf[(sdf.user_id==u2)&(sdf.prod_id==pid)].index.tolist()
                    if r1 and r2: all_edges['upur'].append((r1[0], r2[0]))
            all_edges['upur'] = list(set(all_edges['upur']))

            s_emb   = embeddings[sdf['orig_idx'].values]
            norms   = np.linalg.norm(s_emb, axis=1, keepdims=True)
            s_emb_n = s_emb / (norms + 1e-8)
            sim_mat = s_emb_n @ s_emb_n.T
            np.fill_diagonal(sim_mat, 0)
            for a, b in np.argwhere(sim_mat >= 0.80):
                if a < b: all_edges['sim'].append((int(a), int(b)))

            # ── networkx 그래프 구성 ──
            G = nx.Graph()
            for i, row in sdf.iterrows():
                G.add_node(i, label=int(row.label), rating=row.rating,
                           user=row.user_id, prod=row.prod_id,
                           txt=str(row.text)[:80]+"…")

            for etype in show_etypes:
                for a, b in all_edges[etype]:
                    if G.has_node(a) and G.has_node(b):
                        if G.has_edge(a, b):
                            G[a][b]['types'].append(etype)
                        else:
                            G.add_edge(a, b, types=[etype])

            # 레이아웃
            if layout_algo == "kamada_kawai" and len(G.nodes) < 200:
                pos = nx.kamada_kawai_layout(G)
            else:
                pos = nx.spring_layout(G, seed=42, k=0.55)

            # ── plotly 렌더링 ──
            fig_net = go.Figure()

            # 엣지 타입별로 별도 trace
            for etype in show_etypes:
                ename, ecolor, _ = ETYPE_META[etype]
                ex, ey = [], []
                for a, b, data in G.edges(data=True):
                    if etype in data.get('types', []):
                        x0,y0 = pos[a]; x1,y1 = pos[b]
                        ex += [x0, x1, None]; ey += [y0, y1, None]
                if ex:
                    fig_net.add_trace(go.Scatter(
                        x=ex, y=ey, mode='lines', name=ename,
                        line=dict(color=ecolor, width=1.2),
                        opacity=0.55, hoverinfo='none',
                    ))

            # 노드 (정상 / 사기 구분)
            for lbl, color, shape, size, node_name in [
                (0, C_NORMAL, 'circle',  9, '✅ 정상 리뷰'),
                (1, C_FRAUD,  'diamond', 13, '🚨 사기 리뷰'),
            ]:
                nodes = [n for n in G.nodes() if G.nodes[n]['label'] == lbl]
                if not nodes: continue
                nx_list = [pos[n][0] for n in nodes]
                ny_list = [pos[n][1] for n in nodes]
                htxt    = [
                    f"리뷰 #{n}<br>라벨: {'🚨사기' if lbl==1 else '✅정상'}<br>"
                    f"별점: {G.nodes[n]['rating']}⭐<br>"
                    f"유저: {G.nodes[n]['user']} | 상품: {G.nodes[n]['prod']}<br>"
                    f"텍스트: {G.nodes[n]['txt']}"
                    for n in nodes
                ]
                fig_net.add_trace(go.Scatter(
                    x=nx_list, y=ny_list, mode='markers', name=node_name,
                    marker=dict(color=color, size=size, symbol=shape,
                                line=dict(color='#fff', width=0.8)),
                    hovertext=htxt, hoverinfo='text',
                ))

            fig_net.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(
                    text=f"6종 엣지 기반 사기 네트워크 — {len(sdf)}개 노드 (사기 {sdf.label.sum()}, 정상 {(sdf.label==0).sum()})",
                    x=0.5, xanchor='center',
                ),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=580,
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(30,33,48,0.85)',
                            bordercolor='#3a3f5c', borderwidth=1,
                            font=dict(size=11)),
            )
            st.plotly_chart(fig_net, use_container_width=True)

            # ── 엣지 타입 통계 ──
            st.markdown('<div class="section-header">📊 엣지 타입별 사기-정상 연결 현황</div>',
                        unsafe_allow_html=True)
            stat_rows = []
            for etype, (ename, ecolor, edesc) in ETYPE_META.items():
                total = len(all_edges[etype])
                ff = fn = nn = 0
                for a, b in all_edges[etype]:
                    la = int(sdf.loc[a,'label']); lb = int(sdf.loc[b,'label'])
                    if la==1 and lb==1: ff += 1
                    elif la!=lb:        fn += 1
                    else:               nn += 1
                stat_rows.append({
                    '엣지 타입': ename, '총 엣지': total,
                    '사기-사기': ff, '사기-정상': fn, '정상-정상': nn,
                    '설명': edesc,
                })
            stat_df = pd.DataFrame(stat_rows)

            fig_stat = go.Figure()
            colors_bar = {'사기-사기': C_FRAUD, '사기-정상': C_WARN, '정상-정상': C_NORMAL}
            for col, color in colors_bar.items():
                fig_stat.add_trace(go.Bar(
                    name=col,
                    x=stat_df['엣지 타입'],
                    y=stat_df[col],
                    marker_color=color,
                    opacity=0.85,
                    hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y}}<extra></extra>",
                ))
            fig_stat.update_layout(
                **PLOTLY_LAYOUT, barmode='stack',
                xaxis=dict(tickangle=-10, gridcolor='#2a2f47'),
                yaxis=dict(title='엣지 수', gridcolor='#2a2f47'),
                legend=dict(orientation='h', y=1.1),
                height=320,
            )
            st.plotly_chart(fig_stat, use_container_width=True)

            st.caption(
                "※ 시각화 목적상 일부 엣지 임계값을 완화: "
                "Burst ≥8 (실제 ≥15) / Short Burst ≥4 (실제 ≥5) / P-Day ≥10 (실제 ≥20) "
                "| Sim: BGE 코사인 유사도 ≥0.80 | 노드: 300개 서브샘플"
            )


# ╔══════════════════════════════════════════════╗
# ║  PAGE 3: 모델 성능 비교                      ║
# ╚══════════════════════════════════════════════╝
elif page == "model":
    st.markdown("# 🤖 모델 성능 비교")
    st.markdown("다양한 GNN 계열 모델의 **PR-AUC**와 **Macro-F1** 성능을 비교합니다.")
    st.markdown("---")

    df_perf = results.get("final_table")
    if df_perf is None:
        st.warning("results_final_table.csv를 찾을 수 없습니다.")
        st.stop()

    df_perf = df_perf.sort_values("PR-AUC", ascending=False).reset_index(drop=True)

    def get_color(row):
        if row["type"] == "ensemble": return "#5c6bc0"
        if "SAGE" in str(row["model"]) or "RGCN" in str(row["model"]): return "#ff9800"
        return "#42a5f5"
    df_perf["color"] = df_perf.apply(get_color, axis=1)

    col_bar, col_scatter = st.columns(2)

    with col_bar:
        st.markdown('<div class="section-header">📊 PR-AUC 순위</div>',
                    unsafe_allow_html=True)
        fig_bar = go.Figure(go.Bar(
            y=df_perf["model"], x=df_perf["PR-AUC"],
            orientation="h",
            marker_color=df_perf["color"],
            text=[f"{v:.4f}" for v in df_perf["PR-AUC"]],
            textposition="outside",
            textfont=dict(color="#c5cae9", size=11),
            hovertemplate="<b>%{y}</b><br>PR-AUC: %{x:.4f}<extra></extra>",
        ))
        fig_bar.update_layout(**PLOTLY_LAYOUT,
                              xaxis=dict(range=[0.3, 0.55], gridcolor="#2a2f47"),
                              yaxis=dict(autorange="reversed"), height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_scatter:
        st.markdown('<div class="section-header">🎯 PR-AUC vs Macro-F1</div>',
                    unsafe_allow_html=True)
        fig_sc = go.Figure()
        for _, row in df_perf.iterrows():
            fig_sc.add_trace(go.Scatter(
                x=[row["PR-AUC"]], y=[row["Macro-F1"]],
                mode="markers+text",
                marker=dict(color=row["color"], size=14,
                            line=dict(color="#fff", width=1)),
                text=[str(row["model"]).split("(")[0].strip()],
                textposition="top center",
                textfont=dict(size=9, color="#c5cae9"),
                hovertemplate=(f"<b>{row['model']}</b><br>"
                               f"PR-AUC: {row['PR-AUC']:.4f}<br>"
                               f"Macro-F1: {row['Macro-F1']:.4f}<extra></extra>"),
                showlegend=False,
            ))
        fig_sc.update_layout(**PLOTLY_LAYOUT,
                             xaxis=dict(title="PR-AUC", gridcolor="#2a2f47", range=[0.33, 0.54]),
                             yaxis=dict(title="Macro-F1", gridcolor="#2a2f47", range=[0.59, 0.71]),
                             height=400)
        st.plotly_chart(fig_sc, use_container_width=True)

    # 최고 성능 KPI 카드
    best_prc_row  = df_perf.loc[df_perf["PR-AUC"].idxmax()]
    best_f1_row   = df_perf.loc[df_perf["Macro-F1"].idxmax()]
    best_single   = df_perf[df_perf["type"] == "single"].iloc[0]

    kpi_cols = st.columns(3)
    for col, title, val, sub, color in [
        (kpi_cols[0], "최고 PR-AUC",
         f"{best_prc_row['PR-AUC']:.4f}",
         str(best_prc_row["model"]), C_ACCENT),
        (kpi_cols[1], "최고 Macro-F1",
         f"{best_f1_row['Macro-F1']:.4f}",
         str(best_f1_row["model"]), C_WARN),
        (kpi_cols[2], "단일 모델 최고 PR-AUC",
         f"{best_single['PR-AUC']:.4f}",
         str(best_single["model"]), C_NORMAL),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-color:{color}30;">
              <div class="kpi-label">{title}</div>
              <div class="kpi-value" style="color:{color};">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # 전체 성능 표
    st.markdown('<div class="section-header">📋 전체 성능 테이블</div>',
                unsafe_allow_html=True)
    disp = df_perf[["model", "type", "PR-AUC", "Macro-F1"]].copy()
    disp.insert(0, "순위", range(1, len(disp)+1))
    disp.columns = ["순위", "모델", "유형", "PR-AUC", "Macro-F1"]
    disp["PR-AUC"]   = disp["PR-AUC"].map("{:.4f}".format)
    disp["Macro-F1"] = disp["Macro-F1"].map("{:.4f}".format)
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════╗
# ║  PAGE 4: EDA                  ║
# ╚══════════════════════════════════════════════╝
elif page == "fraud":
    st.markdown("# 🕵️ EDA")
    st.markdown("실제 YelpZip 데이터에서 탐지된 **사기 리뷰 패턴**을 다각도로 분석합니다.")
    st.markdown("---")

    if df_raw is None:
        st.error("subgraph_sample.parquet 파일이 없습니다.")
        st.stop()

    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    tab_time, tab_prod, tab_rating, tab_text = st.tabs(
        ["📅 시간 패턴", "🏪 식당 패턴", "⭐ 별점 패턴", "📝 리뷰 탐색"]
    )

    with tab_time:
        st.markdown('<div class="section-header">📅 연도별 사기 리뷰 발생 추이 (전체 2004~2015)</div>',
                    unsafe_allow_html=True)
        yearly   = agg_yearly if agg_yearly is not None else None
        if yearly is None:
            st.warning("agg_yearly_counts.csv 파일이 없습니다.")
            st.stop()
        y_fraud  = yearly[yearly.label==1]
        y_normal = yearly[yearly.label==0]

        fig_t = go.Figure()
        fig_t.add_trace(go.Bar(x=y_normal["year"].astype(str), y=y_normal["count"],
                               name="정상", marker_color=C_NORMAL, opacity=0.85,
                               hovertemplate="%{x}년<br>정상: %{y:,}건<extra></extra>"))
        fig_t.add_trace(go.Bar(x=y_fraud["year"].astype(str),  y=y_fraud["count"],
                               name="사기",  marker_color=C_FRAUD,  opacity=0.85,
                               hovertemplate="%{x}년<br>사기: %{y:,}건<extra></extra>"))
        fig_t.update_layout(**PLOTLY_LAYOUT, barmode="group",
                            xaxis=dict(title="연도", gridcolor="#2a2f47", dtick=1),
                            yaxis=dict(gridcolor="#2a2f47", title="리뷰 수"),
                            height=340, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_t, use_container_width=True)

        # 사기 비율 추이
        st.markdown('<div class="section-header">📈 연도별 사기 비율 (%)</div>',
                    unsafe_allow_html=True)
        pivot = yearly.pivot(index="year", columns="label", values="count").fillna(0)
        pivot.columns = ["normal", "fraud"]
        pivot["fraud_rate"] = pivot["fraud"] / (pivot["normal"] + pivot["fraud"]) * 100
        total_fraud  = yearly[yearly.label==1]["count"].sum()
        total_all    = yearly["count"].sum()
        avg_rate     = total_fraud / total_all * 100

        fig_rate = go.Figure(go.Bar(
            x=pivot.index.astype(str), y=pivot["fraud_rate"],
            marker=dict(color=pivot["fraud_rate"],
                        colorscale=[[0, C_WARN], [1, C_FRAUD]],
                        showscale=False),
            text=[f"{v:.1f}%" for v in pivot["fraud_rate"]],
            textposition="outside",
            textfont=dict(color="#c5cae9"),
            hovertemplate="%{x}년<br>사기 비율: %{y:.1f}%<extra></extra>",
        ))
        fig_rate.add_hline(y=avg_rate, line_dash="dash", line_color=C_WARN,
                           annotation_text=f"전체 평균 {avg_rate:.1f}%",
                           annotation_font_color=C_WARN)
        fig_rate.update_layout(**PLOTLY_LAYOUT,
                               xaxis=dict(title="연도", gridcolor="#2a2f47", dtick=1),
                               yaxis=dict(gridcolor="#2a2f47", title="사기 비율 (%)",
                                          range=[0, pivot["fraud_rate"].max() * 1.3]),
                               height=280)
        st.plotly_chart(fig_rate, use_container_width=True)

    with tab_prod:

        if agg_prod is not None:
            prod_stats_full = agg_prod.copy()
            top_prods_full = prod_stats_full.sort_values("total", ascending=False).head(20)

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown('<div class="section-header">🏪 리뷰 수 상위 20개 식당 — 사기 비율</div>',
                            unsafe_allow_html=True)
                fig_p = go.Figure()
                xlabels = [f"식당 {pid}" for pid in top_prods_full["prod_id"]]
                fig_p.add_trace(go.Bar(x=xlabels,
                                       y=top_prods_full["total"] - top_prods_full["fraud"],
                                       name="정상", marker_color=C_NORMAL, opacity=0.85))
                fig_p.add_trace(go.Bar(x=xlabels, y=top_prods_full["fraud"],
                                       name="사기", marker_color=C_FRAUD, opacity=0.85))
                fig_p.add_trace(go.Scatter(x=xlabels, y=top_prods_full["fraud_rate"]*100,
                                           name="사기 비율(%)", mode="lines+markers",
                                           line=dict(color=C_WARN, width=2),
                                           yaxis="y2"))
                fig_p.update_layout(**PLOTLY_LAYOUT, barmode="stack",
                                    xaxis=dict(tickangle=-45),
                                    yaxis=dict(title="리뷰 수", gridcolor="#2a2f47"),
                                    yaxis2=dict(title="사기 비율(%)", overlaying="y",
                                                side="right", range=[0,100]),
                                    legend=dict(orientation="h", y=1.1), height=420)
                st.plotly_chart(fig_p, use_container_width=True)

            with col_p2:
                st.markdown('<div class="section-header">📊 식당별 평균 별점 분포 (정상 vs 사기)</div>',
                            unsafe_allow_html=True)
                # agg_prod는 prod_id별 avg_rating 포함
                # fraud/normal 분리: fraud>0 이면 사기 상품, 아니면 정상 판단 (fraud_rate 기준)
                fig_p2 = go.Figure()
                for lbl, color, name in [(0, C_NORMAL, "정상"), (1, C_FRAUD, "사기")]:
                    sub = (prod_stats_full[prod_stats_full["fraud_rate"] >= 0.5]
                           if lbl == 1
                           else prod_stats_full[prod_stats_full["fraud_rate"] < 0.5])
                    fig_p2.add_trace(go.Histogram(
                        x=sub["avg_rating"], name=name, marker_color=color,
                        opacity=0.75, nbinsx=20,
                        hovertemplate=f"{name}<br>평균 별점: %{{x:.2f}}<br>식당 수: %{{y}}<extra></extra>",
                    ))
                fig_p2.update_layout(**PLOTLY_LAYOUT, barmode="overlay",
                                     xaxis=dict(title="식당 평균 별점", gridcolor="#2a2f47"),
                                     yaxis=dict(title="식당 수", gridcolor="#2a2f47"),
                                     legend=dict(orientation="h", y=1.1), height=420)
                st.plotly_chart(fig_p2, use_container_width=True)

            # KPI 카드
            c1, c2, c3, c4 = st.columns(4)
            total_prods = len(prod_stats_full)
            high_fraud = (prod_stats_full["fraud_rate"] >= 0.3).sum()
            avg_fr = prod_stats_full["fraud_rate"].mean()
            max_fr_row = prod_stats_full.loc[prod_stats_full["fraud_rate"].idxmax()]
            for col, lbl, val in [
                (c1, "전체 식당 수",       f"{total_prods:,}개"),
                (c2, "고위험 식당 (≥30%)", f"{high_fraud:,}개"),
                (c3, "평균 사기 비율",      f"{avg_fr*100:.1f}%"),
                (c4, "최고 사기율 식당",    f"식당 {int(max_fr_row.prod_id)} ({max_fr_row.fraud_rate*100:.0f}%)"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="kpi-card">
                      <div class="kpi-label">{lbl}</div>
                      <div class="kpi-value">{val}</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.warning("agg_prod_stats.csv 파일이 없습니다.")

    with tab_rating:

        if agg_rating is not None:
            _rat = agg_rating.copy()
            r_fraud  = _rat[_rat.label==1].set_index("rating")["count"].sort_index()
            r_normal = _rat[_rat.label==0].set_index("rating")["count"].sort_index()

            col_r1, col_r2 = st.columns(2)

            with col_r1:
                st.markdown('<div class="section-header">📊 별점별 리뷰 수 (절대값)</div>',
                            unsafe_allow_html=True)
                fig_ra = go.Figure()
                fig_ra.add_trace(go.Bar(name="정상", x=r_normal.index.astype(str),
                                        y=r_normal.values, marker_color=C_NORMAL, opacity=0.85,
                                        hovertemplate="별점 %{x}★<br>정상: %{y:,}건<extra></extra>"))
                fig_ra.add_trace(go.Bar(name="사기", x=r_fraud.index.astype(str),
                                        y=r_fraud.values,  marker_color=C_FRAUD,  opacity=0.85,
                                        hovertemplate="별점 %{x}★<br>사기: %{y:,}건<extra></extra>"))
                fig_ra.update_layout(**PLOTLY_LAYOUT, barmode="group",
                                     xaxis=dict(title="별점", gridcolor="#2a2f47"),
                                     yaxis=dict(title="리뷰 수", gridcolor="#2a2f47"),
                                     legend=dict(orientation="h", y=1.1), height=350)
                st.plotly_chart(fig_ra, use_container_width=True)

            with col_r2:
                st.markdown('<div class="section-header">📊 별점별 비율 (%) — 양극단 패턴 확인</div>',
                            unsafe_allow_html=True)
                r_fraud_pct  = (r_fraud  / r_fraud.sum()  * 100).round(1)
                r_normal_pct = (r_normal / r_normal.sum() * 100).round(1)
                fig_rb = go.Figure()
                fig_rb.add_trace(go.Bar(name="정상", x=r_normal_pct.index.astype(str),
                                        y=r_normal_pct.values.round(1),
                                        marker_color=C_NORMAL, opacity=0.85,
                                        text=[f"{v:.1f}%" for v in r_normal_pct.values],
                                        textposition="outside",
                                        hovertemplate="별점 %{x}★<br>정상 비율: %{y:.1f}%<extra></extra>"))
                fig_rb.add_trace(go.Bar(name="사기", x=r_fraud_pct.index.astype(str),
                                        y=r_fraud_pct.values.round(1),
                                        marker_color=C_FRAUD,  opacity=0.85,
                                        text=[f"{v:.1f}%" for v in r_fraud_pct.values],
                                        textposition="outside",
                                        hovertemplate="별점 %{x}★<br>사기 비율: %{y:.1f}%<extra></extra>"))
                fig_rb.update_layout(**PLOTLY_LAYOUT, barmode="group",
                                     xaxis=dict(title="별점", gridcolor="#2a2f47"),
                                     yaxis=dict(title="비율 (%)", gridcolor="#2a2f47",
                                                range=[0, 55]),
                                     legend=dict(orientation="h", y=1.1), height=350)
                st.plotly_chart(fig_rb, use_container_width=True)

            # 인사이트 카드
            st.markdown('<div class="section-header">🔍 핵심 인사이트</div>',
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            fraud_1 = round(float(r_fraud_pct.get(1.0, 0)), 1)
            fraud_5 = round(float(r_fraud_pct.get(5.0, 0)), 1)
            norm_1  = round(float(r_normal_pct.get(1.0, 0)), 1)
            norm_5  = round(float(r_normal_pct.get(5.0, 0)), 1)
            for col, lbl, val, sub, color in [
                (c1, "사기 리뷰 1★ 비율", f"{fraud_1}%",
                 f"정상 {norm_1}%의 {fraud_1/norm_1:.1f}배", C_FRAUD),
                (c2, "사기 리뷰 5★ 비율", f"{fraud_5}%",
                 f"정상 {norm_5}%와 비교", C_FRAUD),
                (c3, "1★+5★ 합산 (사기)", f"{fraud_1+fraud_5:.1f}%",
                 "양극단 집중 패턴", C_WARN),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-color:{color}30;">
                      <div class="kpi-label">{lbl}</div>
                      <div class="kpi-value" style="color:{color};">{val}</div>
                      <div class="kpi-sub">{sub}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#1a1d2e; border-radius:10px; padding:16px 18px;
                        border-left:4px solid {C_WARN}; color:#c5cae9; font-size:0.88rem;
                        line-height:1.7;">
              💡 <b>왜 사기 리뷰에 1★과 5★이 몰릴까요?</b><br>
              • <b style="color:{C_FRAUD};">5★ 도배</b>: 특정 상점에 유리한 방향으로 평점을 올리기 위한 작업장 리뷰<br>
              • <b style="color:{C_FRAUD};">1★ 공격</b>: 경쟁 상점을 무너뜨리기 위한 악의적 별점 테러<br>
              • 반면 정상 유저는 중간 점수(2~4★)를 고루 분포하여 사용하는 경향이 있어,
                <b>양극단 별점은 사기 리뷰의 주요 시그널</b>이 됩니다.
            </div>""", unsafe_allow_html=True)
        else:
            st.warning("agg_rating_dist.csv 파일이 없습니다.")

    with tab_text:
        st.markdown('<div class="section-header">🔍 사기 리뷰 샘플 탐색</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            label_filter = st.selectbox("라벨", ["전체", "사기 (1)", "정상 (0)"])
        with c2:
            rating_filter = st.multiselect("별점", [1,2,3,4,5], default=[1,2,3,4,5])
        with c3:
            n_show = st.slider("표시 개수", 5, 30, 10)

        df_show = df.copy()
        if label_filter == "사기 (1)":   df_show = df_show[df_show.label==1]
        elif label_filter == "정상 (0)": df_show = df_show[df_show.label==0]
        df_show = df_show[df_show["rating"].isin(rating_filter)]
        df_show = df_show.sample(min(n_show, len(df_show)), random_state=1)

        for _, row in df_show.iterrows():
            color = C_FRAUD if row.label==1 else C_NORMAL
            badge = "🚨 사기" if row.label==1 else "✅ 정상"
            st.markdown(f"""
            <div style="background:#1a1d2e; border-radius:10px; padding:14px 18px;
                        margin-bottom:10px; border-left:4px solid {color};">
              <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:{color}; font-weight:700;">{badge}</span>
                <span style="color:#6c7a9c; font-size:0.8rem;">
                  ⭐ {row.rating} | 유저 {row.user_id} | 식당 {row.prod_id} | {str(row.date)[:10]}
                </span>
              </div>
              <div style="color:#c5cae9; font-size:0.88rem; line-height:1.5;">
                {str(row.text)[:300]}{'…' if len(str(row.text))>300 else ''}
              </div>
            </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 푸터
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#4a5272; font-size:0.78rem;'>"
    "성균관대학교 1조 | GNN 기반 조직적 어뷰징 네트워크 탐지 | "
    "ITDA 연합학술제 2026 | YelpZip Dataset"
    "</div>",
    unsafe_allow_html=True,
)
