import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    #MainMenu, footer, header { visibility: hidden; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stRadio label { font-size: 1rem !important; }

    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
        padding: 30px 40px; border-radius: 16px; margin-bottom: 24px; color: white;
    }
    .hero h1 { font-size: 2.2rem; font-weight: 800; margin: 0; }
    .hero p { font-size: 1rem; color: #a0aec0; margin: 6px 0 0; }

    .stat-row { display: flex; gap: 14px; margin-bottom: 24px; flex-wrap: wrap; }
    .stat-card {
        flex: 1; min-width: 140px; background: white; border-radius: 12px;
        padding: 18px 20px; border-left: 4px solid; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .stat-card.blue  { border-color: #4299e1; }
    .stat-card.green { border-color: #48bb78; }
    .stat-card.purple{ border-color: #9f7aea; }
    .stat-card.orange{ border-color: #ed8936; }
    .stat-num   { font-size: 1.6rem; font-weight: 800; color: #1a202c; }
    .stat-label { font-size: 0.78rem; color: #718096; margin-top: 2px; }

    .card {
        background: white; border-radius: 14px; padding: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin-bottom: 20px;
    }
    .card-title { font-size: 1.1rem; font-weight: 700; color: #1a202c; margin-bottom: 14px; }

    .rec-card {
        display: flex; align-items: center; background: #f8fafc;
        border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 10px 14px; margin: 7px 0;
    }
    .rec-num {
        background: linear-gradient(135deg, #667eea, #764ba2); color: white;
        font-weight: 700; font-size: 0.82rem; width: 26px; height: 26px;
        border-radius: 50%; display: flex; align-items: center; justify-content: center;
        margin-right: 12px; flex-shrink: 0;
    }
    .rec-name { font-size: 0.88rem; font-weight: 500; color: #2d3748; flex: 1; }
    .rec-score { font-size: 0.78rem; color: #a0aec0; background: #edf2f7; padding: 2px 8px; border-radius: 20px; }

    .segment-box { border-radius: 14px; padding: 24px; text-align: center; margin: 14px 0; }
    .segment-name { font-size: 1.8rem; font-weight: 800; margin: 0; }
    .segment-desc { font-size: 0.9rem; margin: 6px 0 0; opacity: 0.9; }

    .metric-row { display: flex; gap: 10px; margin-top: 14px; }
    .metric-chip { flex: 1; background: #f8fafc; border: 1px solid #e2e8f0;
                   border-radius: 10px; padding: 10px; text-align: center; }
    .metric-chip-val { font-size: 1.1rem; font-weight: 700; color: #1a202c; }
    .metric-chip-lbl { font-size: 0.72rem; color: #718096; margin-top: 2px; }

    .seg-overview-card {
        border-radius: 12px; padding: 16px 20px; margin: 8px 0;
        display: flex; align-items: center; gap: 14px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important; border: none !important; border-radius: 10px !important;
        padding: 10px 20px !important; font-weight: 600 !important; width: 100% !important;
        box-shadow: 0 4px 12px rgba(102,126,234,0.35) !important;
    }
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
    item_sim_df = pd.read_pickle('item_similarity.pkl', compression='gzip')
    with open('product_list.pkl', 'rb') as f:
        product_list = pickle.load(f)
    return kmeans, scaler, cluster_labels, item_sim_df, product_list

kmeans, scaler, cluster_labels, item_sim_df, product_list = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:3rem'>🛒</div>
        <div style='font-size:1.2rem; font-weight:800; color:white;'>Shopper Spectrum</div>
        <div style='font-size:0.75rem; color:#a0aec0; margin-top:4px;'>E-Commerce Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio("", [
        "📊 Dashboard Overview",
        "🎯 Product Recommendations",
        "👥 Customer Segmentation"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Dataset Summary**")
    st.metric("Customers", "4,372")
    st.metric("Segments", "4")
    st.metric("Products", "3,866")
    st.metric("Countries", "38")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard Overview":
    st.markdown("""
    <div class="hero">
        <h1>🛒 Shopper Spectrum</h1>
        <p>AI-powered Customer Segmentation & Product Recommendation Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-row">
        <div class="stat-card blue"><div class="stat-num">541K+</div><div class="stat-label">📦 Total Transactions</div></div>
        <div class="stat-card green"><div class="stat-num">4,372</div><div class="stat-label">👥 Unique Customers</div></div>
        <div class="stat-card purple"><div class="stat-num">3,866</div><div class="stat-label">🏷️ Unique Products</div></div>
        <div class="stat-card orange"><div class="stat-num">38</div><div class="stat-label">🌍 Countries</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Customer Segments Overview
    st.markdown('<div class="card"><div class="card-title">👥 Customer Segments Overview</div>', unsafe_allow_html=True)
    segments = {
        'High-Value': {'count': 217, 'color': '#2d6a4f', 'bg': '#d8f3dc', 'icon': '🌟', 'desc': 'Recent, frequent, high-spend'},
        'Regular':    {'count': 34,  'color': '#2b5797', 'bg': '#dbeafe', 'icon': '✅', 'desc': 'Steady buyers'},
        'Occasional': {'count': 4121,'color': '#b7791f', 'bg': '#fef3c7', 'icon': '🔔', 'desc': 'Infrequent buyers'},
        'At-Risk':    {'count': 0,   'color': '#c53030', 'bg': '#fee2e2', 'icon': '⚠️', 'desc': 'Need win-back campaign'},
    }
    cols = st.columns(4)
    for col, (seg, info) in zip(cols, segments.items()):
        with col:
            st.markdown(f"""
            <div style='background:{info["bg"]}; border-left: 4px solid {info["color"]};
                        border-radius:10px; padding:14px; text-align:center;'>
                <div style='font-size:1.8rem'>{info["icon"]}</div>
                <div style='font-size:1.3rem; font-weight:800; color:{info["color"]}'>{info["count"]}</div>
                <div style='font-size:0.85rem; font-weight:700; color:{info["color"]}'>{seg}</div>
                <div style='font-size:0.72rem; color:#718096; margin-top:4px'>{info["desc"]}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Charts
    st.markdown('<div class="card"><div class="card-title">📈 RFM Profile by Segment</div>', unsafe_allow_html=True)
    seg_data = {
        'Segment':   ['High-Value', 'Regular', 'Occasional'],
        'Recency':   [21.3, 31.5, 60.0],
        'Frequency': [25.7, 21.0, 15.8],
        'Monetary':  [7164.8, 5332.5, 3783.4]
    }
    df_seg = pd.DataFrame(seg_data)
    colors = ['#2d6a4f', '#2b5797', '#b7791f']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('white')
    for ax, metric, title in zip(axes, ['Recency', 'Frequency', 'Monetary'],
                                  ['Avg Recency (days)', 'Avg Frequency (orders)', 'Avg Monetary (£)']):
        bars = ax.barh(df_seg['Segment'], df_seg[metric], color=colors, edgecolor='none', height=0.5)
        for bar, val in zip(bars, df_seg[metric]):
            ax.text(bar.get_width() + max(df_seg[metric]) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:,.1f}', va='center', fontsize=9, color='#333')
        ax.set_title(title, fontsize=10, fontweight='bold', color='#1a202c')
        ax.set_facecolor('#f8fafc')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # Pie chart + bar chart
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><div class="card-title">🥧 Customer Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('white')
        sizes  = [217, 34, 4121]
        labels = ['High-Value', 'Regular', 'Occasional']
        clrs   = ['#2d6a4f', '#2b5797', '#b7791f']
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=clrs,
                                           autopct='%1.1f%%', startangle=90,
                                           wedgeprops=dict(edgecolor='white', linewidth=2))
        for t in autotexts:
            t.set_fontsize(9)
            t.set_color('white')
            t.set_fontweight('bold')
        ax.set_facecolor('white')
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="card-title">💰 Avg Monetary by Segment</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('white')
        segs = ['High-Value', 'Regular', 'Occasional']
        vals = [7164.8, 5332.5, 3783.4]
        bars = ax.bar(segs, vals, color=['#2d6a4f', '#2b5797', '#b7791f'],
                      edgecolor='none', width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                    f'£{val:,.0f}', ha='center', fontsize=9, fontweight='bold', color='#333')
        ax.set_facecolor('#f8fafc')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('Average Spend (£)', fontsize=9)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRODUCT RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Product Recommendations":
    st.markdown("""
    <div class="hero">
        <h1>🎯 Product Recommendations</h1>
        <p>Discover similar products using Item-Based Collaborative Filtering</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔍 Find Similar Products</div>', unsafe_allow_html=True)

    product_input = st.text_input("Enter Product Name", placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER")

    with st.expander("💡 Browse available products"):
        search_term = st.text_input("Search products", key="search")
        if search_term:
            matches = [p for p in product_list if search_term.upper() in p][:15]
            for m in matches:
                st.code(m, language=None)
        else:
            st.caption("Type above to filter products")

    if st.button("🔍 Get Recommendations"):
        if not product_input.strip():
            st.warning("⚠️ Please enter a product name.")
        else:
            query = product_input.strip().upper()
            if query not in item_sim_df.index:
                st.error(f"❌ Product **'{query}'** not found. Try browsing above.")
            else:
                recs = item_sim_df[query].drop(query).sort_values(ascending=False).head(5)
                st.success(f"✅ Top 5 products similar to **{query}**")
                for i, (prod, score) in enumerate(recs.items(), 1):
                    st.markdown(f"""
                    <div class="rec-card">
                        <div class="rec-num">{i}</div>
                        <div class="rec-name">{prod}</div>
                        <div class="rec-score">{score:.3f}</div>
                    </div>""", unsafe_allow_html=True)

                # Similarity bar chart
                st.markdown("<br>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8, 3))
                fig.patch.set_facecolor('white')
                names = [p[:30] + '...' if len(p) > 30 else p for p in recs.index]
                bars = ax.barh(names[::-1], recs.values[::-1],
                               color=['#667eea','#764ba2','#9f7aea','#b794f4','#d6bcfa'],
                               edgecolor='none', height=0.5)
                for bar, val in zip(bars, recs.values[::-1]):
                    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                            f'{val:.3f}', va='center', fontsize=9)
                ax.set_xlabel('Cosine Similarity', fontsize=9)
                ax.set_title('Similarity Scores', fontsize=11, fontweight='bold')
                ax.set_facecolor('#f8fafc')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_xlim(0, 1)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segmentation":
    st.markdown("""
    <div class="hero">
        <h1>👥 Customer Segmentation</h1>
        <p>Predict customer segment using RFM Analysis & KMeans Clustering</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="card"><div class="card-title">📋 Enter RFM Values</div>', unsafe_allow_html=True)
        recency   = st.number_input("📅 Recency — days since last purchase",  min_value=0,   max_value=1000,     value=30,    step=1)
        frequency = st.number_input("🔁 Frequency — number of orders",         min_value=1,   max_value=500,      value=5,     step=1)
        monetary  = st.number_input("💰 Monetary — total spend (£)",           min_value=0.0, max_value=100000.0, value=500.0, step=10.0)

        st.markdown("<br>", unsafe_allow_html=True)

        segment_styles = {
            'High-Value': {'bg': 'linear-gradient(135deg,#1a472a,#2d6a4f)', 'color': '#d8f3dc', 'icon': '🌟', 'desc': 'Recent, frequent & high-spending. Reward with exclusive loyalty perks!'},
            'Regular':    {'bg': 'linear-gradient(135deg,#1a3a5c,#2b5797)', 'color': '#dbeafe', 'icon': '✅', 'desc': 'Steady buyer. Nurture with personalized offers.'},
            'Occasional': {'bg': 'linear-gradient(135deg,#7c4a00,#b7791f)', 'color': '#fef3c7', 'icon': '🔔', 'desc': 'Infrequent buyer. Re-engage with targeted campaigns.'},
            'At-Risk':    {'bg': 'linear-gradient(135deg,#7f1d1d,#c53030)', 'color': '#fee2e2', 'icon': '⚠️', 'desc': "Hasn't purchased recently. Launch urgent win-back campaign!"},
        }

        if st.button("🎯 Predict Customer Segment"):
            rfm_input  = np.array([[recency, frequency, monetary]])
            rfm_scaled = scaler.transform(rfm_input)
            cluster_id = kmeans.predict(rfm_scaled)[0]
            segment    = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
            style      = segment_styles.get(segment, {'bg': '#e2e8f0', 'color': '#2d3748', 'icon': '📌', 'desc': ''})

            st.markdown(f"""
            <div class="segment-box" style="background:{style['bg']};">
                <div style="font-size:2.5rem">{style['icon']}</div>
                <div class="segment-name" style="color:{style['color']}">{segment}</div>
                <div class="segment-desc" style="color:{style['color']}">{style['desc']}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-chip"><div class="metric-chip-val">{recency}d</div><div class="metric-chip-lbl">Recency</div></div>
                <div class="metric-chip"><div class="metric-chip-val">{frequency}</div><div class="metric-chip-lbl">Frequency</div></div>
                <div class="metric-chip"><div class="metric-chip-val">£{monetary:,.0f}</div><div class="metric-chip-lbl">Monetary</div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="card-title">📊 Segment Reference Guide</div>', unsafe_allow_html=True)
        guide = [
            ('🌟', 'High-Value', '#d8f3dc', '#2d6a4f', 'Low Recency, High Frequency, High Monetary'),
            ('✅', 'Regular',    '#dbeafe', '#2b5797', 'Medium Recency, Medium Frequency'),
            ('🔔', 'Occasional', '#fef3c7', '#b7791f', 'High Recency, Low Frequency, Low Monetary'),
            ('⚠️', 'At-Risk',    '#fee2e2', '#c53030', 'Very High Recency, Dropped off'),
        ]
        for icon, name, bg, color, desc in guide:
            st.markdown(f"""
            <div style='background:{bg}; border-left:4px solid {color}; border-radius:10px;
                        padding:12px 16px; margin:8px 0;'>
                <div style='font-size:1.1rem; font-weight:700; color:{color}'>{icon} {name}</div>
                <div style='font-size:0.8rem; color:#4a5568; margin-top:3px'>{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-title">💡 RFM Quick Guide</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.85rem; color:#4a5568; line-height:1.8'>
        📅 <b>Recency</b> — Days since last purchase<br>
        &nbsp;&nbsp;&nbsp;Low (1-30) = Recently active ✅<br>
        &nbsp;&nbsp;&nbsp;High (90+) = Inactive ⚠️<br><br>
        🔁 <b>Frequency</b> — Number of orders<br>
        &nbsp;&nbsp;&nbsp;High (20+) = Loyal customer ✅<br>
        &nbsp;&nbsp;&nbsp;Low (1-3) = One-time buyer ⚠️<br><br>
        💰 <b>Monetary</b> — Total spend in £<br>
        &nbsp;&nbsp;&nbsp;High (5000+) = High-value ✅<br>
        &nbsp;&nbsp;&nbsp;Low (<1000) = Low engagement ⚠️
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align:center; color:#a0aec0; font-size:0.78rem; padding:14px;
            background:white; border-radius:12px; margin-top:10px;'>
    🛒 <b>Shopper Spectrum</b> | KMeans Clustering · Collaborative Filtering · RFM Analysis
</div>""", unsafe_allow_html=True)
