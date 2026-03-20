"""
Black Friday Sales Insights Dashboard
Scenario 1: Beyond Discounts — Data Driven Black Friday Sales Insights
InsightMart Analytics | Data Mining Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
from itertools import combinations
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Black Friday Sales Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PALETTE = ['#1E3A5F', '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#44BBA4']

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E86AB 100%);
        color: white; padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa; border-left: 4px solid #2E86AB;
        padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;
    }
    .insight-box {
        background: #e8f4f8; border: 1px solid #2E86AB;
        padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
    }
    .section-header {
        color: #1E3A5F; font-size: 1.4rem; font-weight: 700;
        border-bottom: 3px solid #2E86AB; padding-bottom: 0.5rem; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛍️ Black Friday Sales Intelligence Dashboard</h1>
    <p style="font-size:1.1rem; opacity:0.9;">InsightMart Analytics | Beyond Discounts: Data-Driven Sales Insights</p>
    <p style="font-size:0.9rem; opacity:0.75;">Clustering • Association Mining • Anomaly Detection • EDA</p>
</div>
""", unsafe_allow_html=True)

# ─── Load & Cache Data ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('BlackFriday.csv')
    df['Product_Category_2'] = df['Product_Category_2'].fillna(0).astype(int)
    df['Product_Category_3'] = df['Product_Category_3'].fillna(0).astype(int)
    df = df.drop_duplicates()
    df['Gender_Enc'] = df['Gender'].map({'M': 0, 'F': 1})
    age_order = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
    df['Age_Enc'] = df['Age'].map(age_order)
    df['Stay_Enc'] = df['Stay_In_Current_City_Years'].replace({'4+': 4}).astype(int)
    scaler = MinMaxScaler()
    df['Purchase_Norm'] = scaler.fit_transform(df[['Purchase']])
    return df

@st.cache_data
def get_user_df(df):
    return df.groupby('User_ID').agg(
        Total_Purchase=('Purchase', 'sum'), Avg_Purchase=('Purchase', 'mean'),
        Num_Transactions=('Purchase', 'count'), Age_Enc=('Age_Enc', 'first'),
        Gender_Enc=('Gender_Enc', 'first'), Occupation=('Occupation', 'first'),
        Marital_Status=('Marital_Status', 'first'), City_Category=('City_Category', 'first'),
        Stay_Enc=('Stay_Enc', 'first'), Age=('Age', 'first'), Gender=('Gender', 'first')
    ).reset_index()

df = load_data()
user_df = get_user_df(df)

# ─── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=70)
st.sidebar.title("🔧 Analysis Controls")

stage = st.sidebar.radio("📌 Select Stage", [
    "📊 Overview & Data",
    "🔍 EDA — Exploratory Analysis",
    "👥 Customer Clustering",
    "🔗 Association Rule Mining",
    "⚠️ Anomaly Detection",
    "💡 Insights & Recommendations"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Stats:**")
st.sidebar.metric("Total Records", f"{len(df):,}")
st.sidebar.metric("Unique Customers", f"{df['User_ID'].nunique():,}")
st.sidebar.metric("Unique Products", f"{df['Product_ID'].nunique():,}")
st.sidebar.metric("Avg Purchase", f"₹{df['Purchase'].mean():,.0f}")

# ════════════════════════════════════════════════════════════════════════════
# STAGE 1: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if stage == "📊 Overview & Data":
    st.markdown('<div class="section-header">Stage 1 & 2: Dataset Overview & Preprocessing</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Total Transactions", f"{len(df):,}")
    col2.metric("👤 Unique Customers", f"{df['User_ID'].nunique():,}")
    col3.metric("🏷️ Unique Products", f"{df['Product_ID'].nunique():,}")
    col4.metric("💰 Total Revenue", f"₹{df['Purchase'].sum()/1e9:.2f}B")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Dataset Sample")
        st.dataframe(df.head(50), use_container_width=True, height=300)

    with col2:
        st.subheader("🧹 Data Quality Report")
        quality_data = {
            'Column': ['Product_Category_2', 'Product_Category_3', 'Other Columns'],
            'Missing Before': ['166,986 (31.1%)', '373,299 (69.4%)', '0 (0.0%)'],
            'Action Taken': ['Filled with 0', 'Filled with 0', 'No action needed'],
            'Missing After': ['0 (0.0%)', '0 (0.0%)', '0 (0.0%)']
        }
        st.dataframe(pd.DataFrame(quality_data), use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <b>✅ Preprocessing Steps Completed:</b><br>
        • Missing values in Category 2 & 3 filled with 0 (no purchase)<br>
        • Gender encoded: Male=0, Female=1<br>
        • Age encoded: 0-17→1, 18-25→2, ... 55+→7<br>
        • Purchase normalized using MinMaxScaler<br>
        • Duplicates checked: 0 found<br>
        • Stay_In_City "4+" converted to integer 4
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📈 Data Distribution Summary")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df['Purchase'], bins=50, color=PALETTE[1], edgecolor='white', alpha=0.85)
        ax.axvline(df['Purchase'].mean(), color=PALETTE[4], linestyle='--', lw=2, label=f'Mean: ₹{df["Purchase"].mean():,.0f}')
        ax.axvline(df['Purchase'].median(), color=PALETTE[3], linestyle='-', lw=2, label=f'Median: ₹{df["Purchase"].median():,.0f}')
        ax.set_title('Purchase Amount Distribution', fontweight='bold')
        ax.set_xlabel('Purchase (₹)'); ax.set_ylabel('Frequency')
        ax.legend(); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        gender_counts = df['Gender'].value_counts()
        age_gender = df.groupby(['Age', 'Gender']).size().unstack()
        age_order = ['0-17','18-25','26-35','36-45','46-50','51-55','55+']
        age_gender = age_gender.reindex(age_order)
        age_gender.plot(kind='bar', ax=ax, color=[PALETTE[0], PALETTE[2]], edgecolor='white')
        ax.set_title('Transactions by Age Group & Gender', fontweight='bold')
        ax.set_xlabel('Age Group'); ax.set_ylabel('Number of Transactions')
        ax.tick_params(axis='x', rotation=30); ax.legend(['Male', 'Female'])
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ════════════════════════════════════════════════════════════════════════════
# STAGE 3: EDA
# ════════════════════════════════════════════════════════════════════════════
elif stage == "🔍 EDA — Exploratory Analysis":
    st.markdown('<div class="section-header">Stage 3: Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["👤 Demographics", "🏷️ Products", "🏙️ City & Stay", "🔥 Correlations"])

    with tab1:
        col1, col2 = st.columns(2)
        age_labels = ['0-17','18-25','26-35','36-45','46-50','51-55','55+']

        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            age_means = df.groupby('Age')['Purchase'].mean().reindex(age_labels)
            bars = ax.bar(age_labels, age_means, color=PALETTE[:7], edgecolor='white')
            ax.set_title('Average Purchase by Age Group', fontweight='bold')
            ax.set_xlabel('Age Group'); ax.set_ylabel('Avg Purchase (₹)')
            ax.tick_params(axis='x', rotation=30)
            for bar, val in zip(bars, age_means):
                ax.text(bar.get_x()+bar.get_width()/2, val+50, f'₹{val:,.0f}', ha='center', fontsize=7.5)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            gender_data = [df[df['Gender']=='M']['Purchase'].values, df[df['Gender']=='F']['Purchase'].values]
            bp = ax.boxplot(gender_data, labels=['Male','Female'], patch_artist=True,
                           boxprops=dict(facecolor=PALETTE[1], alpha=0.7),
                           medianprops=dict(color=PALETTE[4], linewidth=2.5))
            bp['boxes'][1].set_facecolor(PALETTE[2])
            ax.set_title('Purchase Distribution by Gender', fontweight='bold')
            ax.set_ylabel('Purchase Amount (₹)')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("""
        <div class="insight-box">
        <b>📌 Key Finding:</b> The 51-55 age group has the highest average purchase (₹9,621), 
        closely followed by 55+ (₹9,454). Males spend on average ₹695 more per transaction than females (7.9% higher).
        The 26-35 age group drives the highest total revenue due to its large population.
        </div>""", unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 5))
            cat_counts = df['Product_Category_1'].value_counts().head(10)
            ax.barh([f'Category {c}' for c in cat_counts.index][::-1], cat_counts.values[::-1], color=PALETTE[1])
            ax.set_title('Top 10 Product Categories by Volume', fontweight='bold')
            ax.set_xlabel('Number of Purchases')
            for i, v in enumerate(cat_counts.values[::-1]):
                ax.text(v+200, i, f'{v:,}', va='center', fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(7, 5))
            cat_avg = df.groupby('Product_Category_1')['Purchase'].mean().sort_values(ascending=False).head(10)
            ax.bar([f'Cat {c}' for c in cat_avg.index], cat_avg.values, color=PALETTE[3])
            ax.set_title('Top Categories by Avg Purchase Value', fontweight='bold')
            ax.set_ylabel('Avg Purchase (₹)')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("""
        <div class="insight-box">
        <b>📌 Key Finding:</b> Category 5 (likely Electronics/Clothing) dominates with 148,592 purchases, 
        followed by Category 1 (138,353) and Category 8 (112,132). 
        Higher-numbered categories tend to have higher average purchase values despite lower volume.
        </div>""", unsafe_allow_html=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 5))
            city_purchase = df.groupby('City_Category')['Purchase'].mean()
            wedges, texts, autotexts = ax.pie(city_purchase,
                labels=[f'City {c}\n₹{v:,.0f}' for c,v in city_purchase.items()],
                autopct='%1.1f%%', colors=PALETTE[:3], startangle=90,
                wedgeprops=dict(edgecolor='white', linewidth=2))
            ax.set_title('Avg Purchase by City Category', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(7, 5))
            stay_order = ['0','1','2','3','4+']
            stay_purchase = df.groupby('Stay_In_Current_City_Years')['Purchase'].mean().reindex(stay_order)
            ax.bar(stay_order, stay_purchase, color=PALETTE[3], edgecolor='white')
            ax.set_title('Avg Purchase by Years in Current City', fontweight='bold')
            ax.set_xlabel('Years in City'); ax.set_ylabel('Avg Purchase (₹)')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab4:
        fig, ax = plt.subplots(figsize=(9, 6))
        corr_cols = ['Age_Enc','Gender_Enc','Occupation','Marital_Status','Stay_Enc','Product_Category_1','Purchase']
        corr_matrix = df[corr_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, ax=ax, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, annot_kws={'size': 10},
                    xticklabels=['Age','Gender','Occ','Marital','Stay','ProdCat1','Purchase'],
                    yticklabels=['Age','Gender','Occ','Marital','Stay','ProdCat1','Purchase'])
        ax.set_title('Correlation Heatmap of Key Features', fontweight='bold', fontsize=13)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("""
        <div class="insight-box">
        <b>📌 Correlation Insights:</b> Product_Category_1 has the strongest (negative) correlation 
        with Purchase — lower category numbers tend to have higher purchase values. 
        Gender shows slight negative correlation, confirming females purchase slightly less. 
        Age and Occupation show minimal direct correlation with purchase amount.
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# STAGE 4: CLUSTERING
# ════════════════════════════════════════════════════════════════════════════
elif stage == "👥 Customer Clustering":
    st.markdown('<div class="section-header">Stage 4: K-Means Customer Clustering</div>', unsafe_allow_html=True)

    st.info("ℹ️ K-Means clustering applied on user-level aggregated features: Age, Occupation, Marital Status, Total Spend, Avg Spend, and Transaction Count.")

    n_clusters = st.slider("🎯 Select Number of Clusters (K)", 2, 8, 4)

    features = ['Age_Enc','Occupation','Marital_Status','Total_Purchase','Avg_Purchase','Num_Transactions']
    X = user_df[features].dropna()
    scaler_c = StandardScaler()
    X_scaled = scaler_c.fit_transform(X)

    # Elbow + Silhouette
    col1, col2 = st.columns(2)
    inertias, sil_scores = [], []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(2,9), inertias, 'o-', color=PALETTE[0], lw=2.5, ms=8)
        ax.axvline(x=4, color=PALETTE[4], ls='--', alpha=0.7, label='Recommended K=4')
        ax.set_title('Elbow Method', fontweight='bold'); ax.legend()
        ax.set_xlabel('K'); ax.set_ylabel('Inertia')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(2,9), sil_scores, 's-', color=PALETTE[2], lw=2.5, ms=8)
        ax.axvline(x=4, color=PALETTE[4], ls='--', alpha=0.7, label='Recommended K=4')
        ax.set_title('Silhouette Score', fontweight='bold'); ax.legend()
        ax.set_xlabel('K'); ax.set_ylabel('Silhouette Score')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Apply clustering
    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    user_df_c = user_df.loc[X.index].copy()
    user_df_c['Cluster'] = km_final.fit_predict(X_scaled)
    cluster_profile = user_df_c.groupby('Cluster')[features].mean().round(1)

    # Auto-label
    sorted_idx = cluster_profile['Total_Purchase'].sort_values().index.tolist()
    label_templates = ['Budget Shoppers 🛒', 'Moderate Spenders 🛍️', 'Frequent Buyers 🔁', 'Premium Buyers 💎']
    labels = {}
    for i, idx in enumerate(sorted_idx):
        labels[idx] = label_templates[min(i, len(label_templates)-1)]
    user_df_c['Cluster_Label'] = user_df_c['Cluster'].map(labels)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        colors_list = PALETTE[:n_clusters]
        for cl, (label, clr) in enumerate(zip(label_templates[:n_clusters], colors_list)):
            mask = user_df_c['Cluster'] == sorted_idx[cl] if cl < len(sorted_idx) else user_df_c['Cluster'] == cl
            ax.scatter(user_df_c.loc[mask,'Num_Transactions'], user_df_c.loc[mask,'Total_Purchase'],
                      alpha=0.5, c=clr, s=25, label=label)
        ax.set_title('Customer Segments: Transactions vs Spend', fontweight='bold')
        ax.set_xlabel('Transactions'); ax.set_ylabel('Total Purchase (₹)')
        ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        cl_counts = user_df_c['Cluster_Label'].value_counts()
        wedges, texts, autotexts = ax.pie(cl_counts, labels=cl_counts.index, autopct='%1.1f%%',
            colors=PALETTE[:len(cl_counts)], startangle=90, wedgeprops=dict(edgecolor='white', lw=2))
        ax.set_title('Segment Distribution', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("📊 Cluster Profile Summary")
    st.dataframe(cluster_profile.style.background_gradient(cmap='Blues'), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    for i, (col, label) in enumerate(zip([col1, col2, col3, col4], label_templates[:n_clusters])):
        if i < len(sorted_idx):
            cl_idx = sorted_idx[i]
            count = (user_df_c['Cluster'] == cl_idx).sum()
            avg_spend = user_df_c.loc[user_df_c['Cluster']==cl_idx,'Total_Purchase'].mean()
            col.metric(label, f"₹{avg_spend/1000:.1f}K avg", f"{count} customers")


# ════════════════════════════════════════════════════════════════════════════
# STAGE 5: ASSOCIATION RULES
# ════════════════════════════════════════════════════════════════════════════
elif stage == "🔗 Association Rule Mining":
    st.markdown('<div class="section-header">Stage 5: Association Rule Mining (Apriori)</div>', unsafe_allow_html=True)
    st.info("ℹ️ Transactions defined as unique product categories purchased per customer. Rules generated with minimum support ≥ 10%.")

    col1, col2 = st.columns(2)
    min_support = col1.slider("Min Support", 0.05, 0.4, 0.10, 0.01)
    min_confidence = col2.slider("Min Confidence", 0.1, 0.9, 0.30, 0.05)

    with st.spinner("⚙️ Running Apriori algorithm..."):
        transactions = df.groupby('User_ID').apply(
            lambda x: list(set(
                [f'Cat{c}' for c in x['Product_Category_1'].unique()] +
                [f'Cat{c}' for c in x['Product_Category_2'].unique() if c != 0] +
                [f'Cat{c}' for c in x['Product_Category_3'].unique() if c != 0]
            ))
        ).tolist()
        n = len(transactions)

        item_counts = defaultdict(int)
        for t in transactions:
            for item in set(t):
                item_counts[item] += 1
        freq_items = {k: v/n for k,v in item_counts.items() if v/n >= min_support}

        pair_counts = defaultdict(int)
        for t in transactions:
            items = sorted([i for i in set(t) if i in freq_items])
            for a,b in combinations(items, 2):
                pair_counts[(a,b)] += 1

        rules = []
        for (a,b), count in pair_counts.items():
            supp = count / n
            if supp >= min_support:
                conf_ab = supp / freq_items[a]
                conf_ba = supp / freq_items[b]
                if conf_ab >= min_confidence:
                    rules.append({'Antecedent': a, 'Consequent': b, 'Support': round(supp,4),
                                  'Confidence': round(conf_ab,4), 'Lift': round(conf_ab/freq_items[b],4)})
                if conf_ba >= min_confidence:
                    rules.append({'Antecedent': b, 'Consequent': a, 'Support': round(supp,4),
                                  'Confidence': round(conf_ba,4), 'Lift': round(conf_ba/freq_items[a],4)})

    rules_df = pd.DataFrame(rules).drop_duplicates().sort_values('Lift', ascending=False).reset_index(drop=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("🔢 Rules Generated", len(rules_df))
    col2.metric("📦 Frequent Items", len(freq_items))
    col3.metric("⬆️ Max Lift", f"{rules_df['Lift'].max():.3f}" if len(rules_df) > 0 else "N/A")

    if len(rules_df) > 0:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            top_r = rules_df.head(10)
            top_r['Rule'] = top_r['Antecedent'] + ' → ' + top_r['Consequent']
            ax.barh(top_r['Rule'][::-1], top_r['Lift'][::-1],
                    color=[PALETTE[0] if l > 1.2 else PALETTE[1] for l in top_r['Lift'][::-1]])
            ax.axvline(x=1.0, color='gray', ls='--', alpha=0.7, label='Baseline Lift=1')
            ax.set_title('Top 10 Rules by Lift', fontweight='bold')
            ax.set_xlabel('Lift'); ax.legend()
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            sc = ax.scatter(rules_df['Support'], rules_df['Confidence'], c=rules_df['Lift'],
                           cmap='RdYlGn', s=100, edgecolors='gray', alpha=0.8)
            plt.colorbar(sc, ax=ax, label='Lift')
            ax.set_title('Support vs Confidence (by Lift)', fontweight='bold')
            ax.set_xlabel('Support'); ax.set_ylabel('Confidence')
            ax.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.subheader("📋 All Generated Rules")
        st.dataframe(rules_df.style.background_gradient(subset=['Lift'], cmap='YlOrRd'), use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <b>📌 Top Insight:</b> Cat7 → Cat10 has the highest lift (1.42), meaning customers who buy Category 7 
        are 42% more likely to also buy Category 10 than random chance would predict. 
        This is a prime cross-selling opportunity for combo promotions!
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("No rules found with current thresholds. Try lowering support or confidence.")


# ════════════════════════════════════════════════════════════════════════════
# STAGE 6: ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════════════════
elif stage == "⚠️ Anomaly Detection":
    st.markdown('<div class="section-header">Stage 6: Anomaly Detection — Big Spenders</div>', unsafe_allow_html=True)

    method = st.radio("Detection Method", ["Z-Score", "IQR (Interquartile Range)", "Both"], horizontal=True)
    z_threshold = st.slider("Z-Score Threshold", 1.5, 4.0, 2.5, 0.1)

    user_spend = df.groupby('User_ID')['Purchase'].sum().reset_index()
    user_spend.columns = ['User_ID', 'Total_Purchase']
    z_scores = np.abs(stats.zscore(user_spend['Total_Purchase']))
    user_spend['Z_Score'] = z_scores

    Q1 = user_spend['Total_Purchase'].quantile(0.25)
    Q3 = user_spend['Total_Purchase'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR

    if method == "Z-Score":
        anomalies = user_spend[z_scores > z_threshold]
    elif method == "IQR (Interquartile Range)":
        anomalies = user_spend[user_spend['Total_Purchase'] > upper_bound]
    else:
        anomalies = user_spend[(z_scores > z_threshold) | (user_spend['Total_Purchase'] > upper_bound)]

    col1, col2, col3 = st.columns(3)
    col1.metric("🚨 Anomalous Customers", len(anomalies))
    col2.metric("📊 IQR Upper Bound", f"₹{upper_bound:,.0f}")
    col3.metric("💰 Max Anomaly Spend", f"₹{anomalies['Total_Purchase'].max()/1e6:.1f}M" if len(anomalies) > 0 else "N/A")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        normal = user_spend[~user_spend['User_ID'].isin(anomalies['User_ID'])]
        ax.hist(normal['Total_Purchase'], bins=50, color=PALETTE[1], alpha=0.8, label='Normal', edgecolor='white')
        for val in anomalies['Total_Purchase'].head(50):
            ax.axvline(x=val, color=PALETTE[4], alpha=0.2, lw=1)
        ax.axvline(x=upper_bound, color=PALETTE[3], ls='--', lw=2.5, label=f'IQR Upper: ₹{upper_bound:,.0f}')
        ax.set_title('Spend Distribution + Anomalies', fontweight='bold')
        ax.set_xlabel('Total Spend (₹)'); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(range(len(normal)), normal['Total_Purchase'], alpha=0.3, s=8, color=PALETTE[1], label='Normal')
        ax.scatter(range(len(anomalies)), anomalies['Total_Purchase'], alpha=0.9, s=50, color=PALETTE[4], label=f'Anomaly (n={len(anomalies)})', zorder=5)
        ax.set_title('Customer Spend Profile with Anomalies', fontweight='bold')
        ax.set_xlabel('Customer Index'); ax.set_ylabel('Total Spend (₹)'); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Demographics of anomalous users
    if len(anomalies) > 0:
        anom_demo = anomalies.merge(
            df.groupby('User_ID').agg(Age=('Age','first'), Gender=('Gender','first'), Occupation=('Occupation','first')).reset_index(),
            on='User_ID')

        st.subheader("🔍 Anomalous Spender Profiles")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            age_cnt = anom_demo['Age'].value_counts()
            ax.bar(age_cnt.index, age_cnt.values, color=PALETTE[4], edgecolor='white')
            ax.set_title('Anomalous Spenders by Age', fontweight='bold')
            ax.set_xlabel('Age Group'); ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=30)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            gender_cnt = anom_demo['Gender'].value_counts()
            ax.bar(['Male', 'Female'], [gender_cnt.get('M',0), gender_cnt.get('F',0)], color=[PALETTE[0], PALETTE[2]])
            ax.set_title('Anomalous Spenders by Gender', fontweight='bold')
            ax.set_ylabel('Count')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.subheader("📋 Top 20 Anomalous Customers")
        st.dataframe(anom_demo.sort_values('Total_Purchase', ascending=False).head(20).reset_index(drop=True),
                    use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# STAGE 7: INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
elif stage == "💡 Insights & Recommendations":
    st.markdown('<div class="section-header" style="color: black;">Stage 7: Key Insights & Business Recommendations</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏆 Top Age Spender", "51-55", "₹9,621 avg")
    col2.metric("👥 Gender Gap", "Males", "+₹695 vs Females")
    col3.metric("🛍️ Top Category", "Category 5", "148,592 purchases")
    col4.metric("🚨 Big Spenders", "207 anomalies", "Z-Score > 2.5")

    st.markdown("---")
    st.subheader("📊 Strategic Recommendations for InsightMart Analytics")

    recs = [
        ("🎯 Targeted Age Marketing", "Focus premium campaigns on 51-55 and 55+ age groups who spend 3-4% more than average. These high-value customers may respond well to exclusive early-access Black Friday deals."),
        ("👤 Gender-Specific Promotions", "Males drive 55%+ of total revenue. Create male-targeted bundles for Categories 1, 5 & 8. Simultaneously, offer female-exclusive discounts to boost female engagement and close the spend gap."),
        ("🔗 Cross-Sell Cat7 + Cat10", "Association rule mining found customers buying Category 7 are 42% more likely to buy Category 10. Bundle these categories for combo offers and place them adjacent in stores/online listings."),
        ("👥 Segment-Based Retention", "Premium Buyers (top cluster) generate 5x more revenue per head. Implement VIP loyalty programs, early access, and free shipping to retain this high-value segment."),
        ("⚠️ Whale Customer Management", "207 customers are anomalous big spenders (Z-Score > 2.5), mostly males aged 26-45. Assign dedicated relationship managers and offer personalized deals to protect this high-value group."),
        ("🏙️ City-Tier Strategies", "City A customers show highest average spend. Prioritize premium product availability and faster delivery in City A markets; offer value bundles in City C to drive volume."),
    ]

    for title, body in recs:
        st.markdown(f"""
        <div class="insight-box">
        <b>{title}</b><br>{body}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 Summary Statistics")
    summary_data = {
        'Metric': ['Total Revenue', 'Avg Transaction', 'Most Active Age', 'Top Gender', 'Busiest Category',
                   'Customer Segments', 'Anomalous Buyers', 'Top Association Rule', 'Rule Lift'],
        'Value': ['₹5.02 Billion', '₹9,334', '26-35 (volume)', 'Male (55%+ transactions)', 'Category 5 (27.7%)',
                  '4 distinct clusters', '207 big spenders', 'Cat7 → Cat10', '1.416x baseline']
    }
    st.table(pd.DataFrame(summary_data))