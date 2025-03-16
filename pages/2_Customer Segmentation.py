import streamlit as st
import pandas as pd
st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide",
    initial_sidebar_state="collapsed"
)
def line_break():
    st.markdown("</br>", unsafe_allow_html=True)

# Hide the Streamlit header and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def line_break():
    st.markdown("</br>", unsafe_allow_html=True)

st.title("Customer Segmentation")
st.subheader("K-means Clustering")
line_break()
line_break()
st.write("Customer Classification:")
st.write("""
         1. Use clustering (e.g., k-means) to group customers into segments based on their purchase behavior (e.g., frequency, total spending, product preferences).
         2. Label the clusters with intuitive names (e.g., "High Spenders," "Occasional Buyers").""") 

line_break()
line_break()
st.write("#### After running the K-Means clustering algorithm with cluster values ranging from 1 to 20, the optimal number of clusters was determined to be 5.")

st.markdown("""
- **Cluster 0**: Loyal High-Spenders
- **Cluster 1**: Service-Focused Occasional Shoppers  
- **Cluster 2**: Price-Sensitive or Discount Shoppers
- **Cluster 3**: Active and Engaged Value Shoppers
- **Cluster 4**: Balanced Regulars
""")

line_break()
line_break()
# Display elbow curve image
st.image("elbow_curve.png", caption="Elbow Curve for K-Means Clustering", use_container_width=True)

# Adding detailed cluster descriptions
line_break()
st.subheader("Customer Segment Profiles")
st.write("Below are the identified customer segments based on our clustering analysis:")

# Using expanders for detailed cluster descriptions
with st.expander("🌟 Cluster 0: Loyal High-Spenders", expanded=True):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("### Key Metrics:")
        st.markdown("- 💰 **Monetary**: Highest (~$1055)")
        st.markdown("- 🔄 **Frequency**: Highest (≈13.7)")
        st.markdown("- ⏱️ **Recency**: Good (≈58 days)")
        st.markdown("- ⏳ **Between Purchases**: ≈48 days")
    with col2:
        st.markdown("### Segment Profile")
        st.info("""
        These customers represent the highest value segment with frequent purchases and high total spending. Their regular engagement pattern indicates strong loyalty to your salon.
        
        **Recommended Approach**: Implement a VIP program with exclusive services, early appointment access, and loyalty rewards. Focus on retention through personalized communications and premium service offerings.
        """)

with st.expander("💆 Cluster 1: Service-Focused Occasional Shoppers"):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("### Key Metrics:")
        st.markdown("- 💰 **Monetary**: Moderate (~$560)")
        st.markdown("- 🔄 **Frequency**: Moderate (≈7.8)")
        st.markdown("- ⏱️ **Recency**: Fair (≈71 days)")
        st.markdown("- 🧴 **Service Ratio**: High (≈0.77)")
        st.markdown("- ⏳ **Between Purchases**: ≈98 days")
    with col2:
        st.markdown("### Segment Profile")
        st.info("""
        This segment prefers services over products and visits less frequently. They might schedule appointments for specific services or special occasions rather than regular maintenance.
        
        **Recommended Approach**: Create service bundles and packages to encourage more frequent visits. Implement reminder campaigns about the benefits of regular service appointments. Consider seasonal promotions aligned with their typical visit cadence.
        """)

with st.expander("💲 Cluster 2: Price-Sensitive or Discount Shoppers"):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("### Key Metrics:")
        st.markdown("- 💰 **Monetary**: Lowest (~$463)")
        st.markdown("- 🔄 **Frequency**: Lowest (≈7.35)")
        st.markdown("- ⏱️ **Recency**: Poor (≈103 days)")
        st.markdown("- 🛍️ **Product Ratio**: Higher (≈0.47)")
        st.markdown("- 💵 **AOV**: Lower (≈$63.7)")
    with col2:
        st.markdown("### Segment Profile")
        st.info("""
        These customers appear to be most price-conscious, with a preference for products over services compared to other segments. Their lower engagement and spending indicate they may be responsive to promotions or only shop during sales.
        
        **Recommended Approach**: Target with limited-time offers, discount bundles on products, and entry-level service promotions. Develop a re-engagement strategy to bring these customers back more frequently.
        """)

with st.expander("✨ Cluster 3: Active and Engaged Value Shoppers"):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("### Key Metrics:")
        st.markdown("- 💰 **Monetary**: High (~$976)")
        st.markdown("- 🔄 **Frequency**: High (≈12.36)")
        st.markdown("- ⏱️ **Recency**: Best (≈50.7 days)")
        st.markdown("- 💵 **AOV**: Highest")
        st.markdown("- ⏳ **Between Purchases**: ≈56.6 days")
    with col2:
        st.markdown("### Segment Profile")
        st.info("""
        This segment represents recently active customers with high spending per visit. They value quality and are willing to pay for premium services or products, making them extremely valuable.
        
        **Recommended Approach**: Introduce them to higher-tier services and premium product lines. Create exclusive experiences and focus on enhancing each visit. Develop personalized recommendations based on their preferences.
        """)

with st.expander("⚖️ Cluster 4: Balanced Regulars"):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("### Key Metrics:")
        st.markdown("- 💰 **Monetary**: High (~$934)")
        st.markdown("- 🔄 **Frequency**: Good (≈11.77)")
        st.markdown("- ⏱️ **Recency**: Good (≈59.7 days)")
        st.markdown("- 🧴 **Service Ratio**: High (≈0.75)")
        st.markdown("- ⏳ **Between Purchases**: ≈61.5 days")
    with col2:
        st.markdown("### Segment Profile")
        st.info("""
        These customers demonstrate consistent behavior with a good balance between services and products. They're regular visitors who represent reliable revenue for the salon.
        
        **Recommended Approach**: Focus on cross-selling complementary services and products. Implement a consistent communication strategy to maintain their regular visit pattern. Consider loyalty rewards that recognize their consistent patronage.
        """)


line_break()
line_break()
line_break()
st.write("## Clustering Methodology")

st.markdown("""
### 1. Introduction

The objective of this methodology is to classify customers into distinct clusters based on their purchasing behavior and demographic characteristics. This is achieved using the K-Means clustering algorithm, which groups similar customers together to identify meaningful patterns.

### 2. Data Preprocessing

#### 2.1 Feature Extraction
Key customer attributes were extracted to serve as input features for clustering. These include:

- **Recency, Frequency, and Monetary Value (RFM)**: Measures how recently a customer made a purchase, how often they buy, and their total spending.
- **Demographics**: Age and gender, with gender being one-hot encoded.
- **Average Order Value (AOV)**: Computed as the total spending divided by the number of transactions.
- **Time Between Purchases**: The average duration between successive purchases.
- **Purchase Streak**: The longest consecutive sequence of daily purchases.
- **Product vs. Service Ratio**: The proportion of service-related purchases compared to product purchases.

#### 2.2 Feature Scaling
To ensure that all attributes contribute equally to clustering, StandardScaler from sklearn.preprocessing was applied to standardize numerical features.

### 3. Finding the Optimal Number of Clusters

To determine the ideal number of clusters:

- **Elbow Method** was applied by plotting the inertia (sum of squared distances) against different cluster values.
- The optimal cluster count was identified at the "elbow point" of the plot.

### 4. Clustering Process

- **K-Means Algorithm**: The dataset was clustered using the optimal K value determined from the elbow method.
- **Cluster Assignment**: Each customer was assigned a cluster label.
- **Cluster Interpretation**: Summary statistics were generated to understand customer characteristics in each cluster.

### 5. Results and Export

The classified customers, along with their respective cluster labels, were merged back with the original dataset and saved as cluster_samples.csv for further analysis.
""")

# Display sample data from clustering results
st.write("### Sample Clustering Results")
st.write("Below is a sample of the clustered customer data:")

# Read and display the cluster samples CSV
cluster_samples = pd.read_csv('cluster_samples.csv')
st.dataframe(cluster_samples)


# Add footer with attribution
line_break()
line_break()
line_break()
st.markdown("""
<div style='text-align: center; padding: 20px; position: fixed; bottom: 0; left: 0; right: 0; background-color: #e6e6fa; color: #000080;'>
    <p>Submitted by Ashar</p>
</div>
""", unsafe_allow_html=True) 