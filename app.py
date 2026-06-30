import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f5f7fa; }
    #MainMenu, footer, header { visibility: hidden; }

    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 40px 50px;
        border-radius: 16px;
        margin-bottom: 30px;
        color: white;
    }
    .hero h1 { font-size: 2.8rem; font-weight: 800; margin: 0; letter-spacing: -1px; }
    .hero p  { font-size: 1.1rem; color: #a0aec0; margin: 8px 0 0; }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        color: #e2e8f0;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-top: 14px;
    }

    .stat-row { display: flex; gap: 16px; margin-bottom: 28px; }
    .stat-card {
        flex: 1;
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        border-left: 4px solid;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .stat-card.blue  { border-color: #4299e1; }
    .stat-card.green { border-color: #48bb78; }
    .stat-card.purple{ border-color: #9f7aea; }
    .stat-card.orange{ border-color: #ed8936; }
    .stat-num  { font-size: 1.8rem; font-weight: 800; color: #1a202c; }
    .stat-label{ font-size: 0.82rem; color: #718096; margin-top: 2px; }

    .section-card {
        background: white;
        border-radius: 16px;
        padding: 32px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        height: 100%;
    }
    .section-title { font-size: 1.3rem; font-weight: 700; color: #1a202c; margin-bottom: 6px; }
    .section-sub { font-size: 0.88rem; color: #718096; margin-bottom: 24px; }

    .stTextInput input, .stNumberInput input {
        border-radius: 8px !important;
        border: 1.5px solid #e2e8f0 !important;
        padding: 10px 14px !important;
        font-size: 0.95rem !important;
        background: #f8fafc !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #4299e1 !important;
        box-shadow: 0 0 0 3px rgba(66,153,225,0.15) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 12px rgba(102,126,234,0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(102,126,234,0.45) !important;
    }

    .rec-card {
        display: flex;
        align-items: center;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        transition: all 0.2s;
    }
    .rec-card:hover { background: #edf2f7; border-color: #cbd5e0; }
    .rec-num {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: 700;
        font-size: 0.85rem;
        width: 28px; height: 28px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        margin-right: 14px;
        flex-shrink: 0;
    }
    .rec-name { font-size: 0.92rem; font-weight: 500; color: #2d3748; flex: 1; }
    .rec-score {
        font-size: 0.8rem;
        color: #a0aec0;
        background: #edf2f7;
        padding: 3px 10px;
        border-radius: 20px;
    }

    .segment-box {
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        margin: 16px 0;
    }
    .segment-name { font-size: 2rem; font-weight: 800; margin: 0; }
    .segment-desc { font-size: 0.95rem; margin: 8px 0 0; opacity: 0.85; }

    .metric-row { display: flex; gap: 12px; margin-top: 16px; }
    .metric-chip {
        flex: 1;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }
    .metric-chip-val { font-size: 1.2rem; font-weight: 700; color: #1a202c; }
    .metric-chip-lbl { font-size: 0.75rem; color: #718096; margin-top: 2px; }

    .divider { height: 1px; background: #e2e8f0; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('cluster_labels.pkl', 'rb') as f:
        cluster_labels = pickle.load(f)
    item_sim_df = pd.read_pickle('item_similarity.pkl')
    with open('product_list.pkl', 'rb') as f:
        product_list = pickle.load(f)
    return kmeans, scaler, cluster_labels, item_sim_df, product_list

kmeans, scaler, cluster_labels, item_sim_df, product_list = load_models()

# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛒 Shopper Spectrum</h1>
    <p>AI-powered Customer Segmentation & Product Recommendation System</p>
    <span class="hero-badge">⚡ E-Commerce Analytics · RFM Analysis · Collaborative Filtering</span>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-row">
    <div class="stat-card blue">
        <div class="stat-num">541K+</div>
        <div class="stat-label">📦 Total Transactions</div>
    </div>
    <div class="stat-card green">
        <div class="stat-num">4,372</div>
        <div class="stat-label">👥 Unique Customers</div>
    </div>
    <div class="stat-card purple">
        <div class="stat-num">4,223</div>
        <div class="stat-label">🏷️ Unique Products</div>
    </div>
    <div class="stat-card orange">
        <div class="stat-num">38</div>
        <div class="stat-label">🌍 Countries</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Two Columns ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — Product Recommendation
# ═══════════════════════════════════════════════════════════════════════════════
with col1:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">🎯 Product Recommendation</div>
        <div class="section-sub">Enter a product name to discover 5 similar products using collaborative filtering</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    product_input = st.text_input(
        "🔎 Product Name",
        placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER",
        key="product_input"
    )

    with st.expander("💡 Browse available products"):
        search_term = st.text_input("Search", placeholder="Type to filter...", key="search")
        if search_term:
            matches = [p for p in product_list if search_term.upper() in p][:15]
            for m in matches:
                st.code(m, language=None)
        else:
            st.caption("Type above to search products")

    if st.button("🔍 Get Recommendations", key="rec_btn"):
        if not product_input.strip():
            st.warning("⚠️ Please enter a product name.")
        else:
            query = product_input.strip().upper()
            if query not in item_sim_df.index:
                st.error(f"❌ Product **'{query}'** not found. Try browsing products above.")
            else:
                recs = item_sim_df[query].drop(query).sort_values(ascending=False).head(5)
                st.markdown(f"<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown(f"**Similar to:** `{query}`")
                for i, (prod, score) in enumerate(recs.items(), 1):
                    st.markdown(f"""
                    <div class="rec-card">
                        <div class="rec-num">{i}</div>
                        <div class="rec-name">{prod}</div>
                        <div class="rec-score">{score:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — Customer Segmentation
# ═══════════════════════════════════════════════════════════════════════════════
with col2:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📊 Customer Segmentation</div>
        <div class="section-sub">Enter RFM values to predict which customer segment this customer belongs to</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    recency   = st.number_input("📅 Recency — days since last purchase",  min_value=0,   max_value=1000,    value=30,    step=1)
    frequency = st.number_input("🔁 Frequency — number of orders",         min_value=1,   max_value=500,     value=5,     step=1)
    monetary  = st.number_input("💰 Monetary — total spend (£)",           min_value=0.0, max_value=100000.0, value=500.0, step=10.0)

    segment_styles = {
        'High-Value': {
            'bg': 'linear-gradient(135deg, #1a472a, #2d6a4f)',
            'color': '#d8f3dc',
            'icon': '🌟',
            'desc': 'Recent, frequent & high-spending. Reward with exclusive loyalty perks!'
        },
        'Regular': {
            'bg': 'linear-gradient(135deg, #1a3a5c, #2b5797)',
            'color': '#dbeafe',
            'icon': '✅',
            'desc': 'Steady buyer with consistent spend. Nurture with personalized offers.'
        },
        'Occasional': {
            'bg': 'linear-gradient(135deg, #7c4a00, #b7791f)',
            'color': '#fef3c7',
            'icon': '🔔',
            'desc': 'Infrequent buyer. Re-engage with targeted campaigns & discounts.'
        },
        'At-Risk': {
            'bg': 'linear-gradient(135deg, #7f1d1d, #c53030)',
            'color': '#fee2e2',
            'icon': '⚠️',
            'desc': "Hasn't purchased recently. Launch urgent win-back campaign!"
        },
    }

    if st.button("🎯 Predict Customer Segment", key="seg_btn"):
        rfm_input  = np.array([[recency, frequency, monetary]])
        rfm_scaled = scaler.transform(rfm_input)
        cluster_id = kmeans.predict(rfm_scaled)[0]
        segment    = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
        style      = segment_styles.get(segment, {'bg':'#e2e8f0','color':'#2d3748','icon':'📌','desc':''})

        st.markdown(f"""
        <div class="segment-box" style="background:{style['bg']};">
            <div style="font-size:2.5rem">{style['icon']}</div>
            <div class="segment-name" style="color:{style['color']}">{segment}</div>
            <div class="segment-desc" style="color:{style['color']}">{style['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-chip">
                <div class="metric-chip-val">{recency}d</div>
                <div class="metric-chip-lbl">Recency</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-val">{frequency}</div>
                <div class="metric-chip-lbl">Frequency</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-val">£{monetary:,.0f}</div>
                <div class="metric-chip-lbl">Monetary</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#a0aec0; font-size:0.82rem; padding:16px;
            background:white; border-radius:12px; margin-top:10px;'>
    🛒 <b>Shopper Spectrum</b> &nbsp;|&nbsp; E-Commerce Analytics Dashboard &nbsp;|&nbsp;
    KMeans Clustering · Collaborative Filtering · RFM Analysis
</div>
""", unsafe_allow_html=True)
