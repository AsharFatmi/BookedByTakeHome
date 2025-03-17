import streamlit as st
import pandas as pd
from customer_classification import CustomerClassifier
from data_analysis import DataAnalyzer
import matplotlib.pyplot as plt

# Configure the Streamlit page layout and title
st.set_page_config(
    page_title="Run K-means Clustering",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide the Streamlit header and footer for a cleaner UI
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Utility function to add vertical spacing
def line_break():
    st.markdown("</br>", unsafe_allow_html=True)

st.title("Run K-means Clustering")
line_break()

# Initialize session state for storing results
if 'clustering_complete' not in st.session_state:
    st.session_state.clustering_complete = False
if 'elbow_curve_generated' not in st.session_state:
    st.session_state.elbow_curve_generated = False
if 'optimal_clusters' not in st.session_state:
    st.session_state.optimal_clusters = None

# Main function to execute K-means clustering analysis
def run_kmeans(max_clusters, n_clusters=None):
    try:
        with st.spinner('Running K-means clustering analysis...'):
            # Initialize data analyzer and load data from CSV files
            analyzer = DataAnalyzer('customers.csv', 'items.csv', 'purchases.csv')
            
            if analyzer.load_data():
                # Initialize classifier for customer segmentation
                classifier = CustomerClassifier()
                
                if not n_clusters:
                    # Run elbow method to find optimal number of clusters
                    df_scaled = classifier.run_elbow_method(analyzer.df_merged, max_clusters)
                    st.session_state.elbow_curve_generated = True
                    st.success("Elbow curve generated successfully!")
                else:
                    # Run full clustering with user-specified number of clusters
                    classifier.set_n_clusters(n_clusters)
                    df_merged = classifier.fit_predict(analyzer.df_merged)
                    
                    if df_merged is not None:
                        # Save cluster results and get summary statistics
                        cluster_means = classifier.save_cluster(df_merged, "cluster_samples.csv")
                        st.session_state.clustering_complete = True
                        st.success("K-means clustering completed successfully!")
                        
                        # Display cluster means for analysis
                        st.subheader("Cluster Analysis Results")
                        st.dataframe(cluster_means)
                        
                        # Display cluster visualizations for interpretation
                        st.subheader("Cluster Visualizations")
                        try:
                            st.image("cluster_visualizations.png", caption="2D Cluster Visualizations", use_container_width=True)
                            st.image("cluster_visualizations_3d.png", caption="3D RFM Cluster Visualization", use_container_width=True)
                        except:
                            st.warning("Some visualizations could not be loaded.")
            else:
                st.error("Failed to load data. Please check your data files.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# STEP 1: Generate Elbow Curve to determine optimal number of clusters
st.subheader("Step 1: Generate Elbow Curve")
with st.form("kmeans_step1"):
    max_clusters = st.number_input(
        "Maximum number of clusters to consider:",
        min_value=2,
        max_value=50,
        value=20
    )
    generate_elbow = st.form_submit_button("Generate Elbow Curve")
    if generate_elbow:
        run_kmeans(max_clusters)

# Display Elbow Curve and enable Step 2 if Step 1 is complete
if st.session_state.elbow_curve_generated:
    try:
        st.image("elbow_curve.png", caption="Elbow Curve for K-Means Clustering", use_container_width=True)
        
        # STEP 2: Run Clustering with user-selected number of clusters
        st.subheader("Step 2: Run Clustering")
        with st.form("kmeans_step2"):
            st.write("Based on the elbow curve above, select the optimal number of clusters:")
            n_clusters = st.number_input(
                "Number of clusters:",
                min_value=2,
                max_value=max_clusters,
                value=5
            )
            run_clustering = st.form_submit_button("Run Clustering")
            if run_clustering:
                run_kmeans(max_clusters, n_clusters)
    except:
        st.warning("Elbow curve image not found. Please generate the elbow curve first.")

# Add footer with attribution
line_break()
line_break()
line_break()
st.markdown("""
<div style='text-align: center; padding: 20px; position: fixed; bottom: 0; left: 0; right: 0; background-color: #e6e6fa; color: #000080;'>
    <p>Submitted by Ashar</p>
</div>
""", unsafe_allow_html=True) 