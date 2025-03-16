import streamlit as st
import sys
import os
import pandas as pd

# Add the parent directory to sys.path to import inference module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import RecommenderInference, main

st.set_page_config(
    page_title="Product Recommendation",
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

st.title("Product Recommendation")
line_break()
st.write("This AI-based recommendation system uses collaborative filtering to suggest products based on customer purchase patterns and similarities between customers.")

# Create a container for the recommendation tool
rec_container = st.container()
with rec_container:
    st.subheader("Get Product Recommendations")
    
    # Load sample customer IDs for the selectbox
    try:
        customers_df = pd.read_csv("customers.csv")
        customer_ids = customers_df['CustomerID'].tolist()
        default_id = "Cust0003" if "Cust0003" in customer_ids else customer_ids[0] if customer_ids else "Cust0003"
    except Exception as e:
        st.warning(f"Could not load customer data: {str(e)}")
        customer_ids = [f"Cust{i:04d}" for i in range(1, 11)]  # Fallback
        default_id = "Cust0003"
    
    # Create two columns for the form
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Customer ID input (using selectbox for easier selection)
        customer_id = st.selectbox(
            "Select Customer ID:",
            options=customer_ids,
            index=customer_ids.index(default_id) if default_id in customer_ids else 0,
            help="Choose a customer ID to get personalized product recommendations"
        )
        
        # # Alternatively, allow manual entry
        # use_manual = st.checkbox("Or enter Customer ID manually")
        # if use_manual:
        #     manual_id = st.text_input("Enter Customer ID:", value=default_id)
        #     if manual_id:
        #         customer_id = manual_id
    
    with col2:
        st.write("")  # Add some spacing
        st.write("")  # Add some spacing
        get_recommendations = st.button("Get Recommendations", type="primary", use_container_width=True)

# Display recommendations when button is clicked
if get_recommendations:
    with st.spinner("Generating recommendations..."):
        try:
            # Get recommendations using the inference module
            model_base_dir = "saved_models"
            if not os.path.exists(model_base_dir):
                st.error(f"Model directory '{model_base_dir}' not found. Please make sure the model has been trained.")
            else:
                model_dirs = [d for d in os.listdir(model_base_dir) if os.path.isdir(os.path.join(model_base_dir, d))]
                if not model_dirs:
                    st.error(f"No model directories found in '{model_base_dir}'.")
                else:
                    latest_model_dir = os.path.join(model_base_dir, sorted(model_dirs)[-1])
                    
                    # Call the main function from inference.py
                    results = main([customer_id])
                    
                    if results and len(results) > 0:
                        result = results[0]  # Get the first (only) result
                        
                        # Display the results in a nice format
                        st.success(f"Recommendations generated for customer {result['customer_id']}")
                        
                        # Create tabs for recommendations and purchase history
                        rec_tab, history_tab = st.tabs(["📊 Recommendations", "📋 Purchase History"])
                        
                        with rec_tab:
                            if result['recommendations']:
                                st.subheader("Recommended Products")
                                rec_df = pd.DataFrame({
                                    "Product": result['recommendations'],
                                    "Rank": range(1, len(result['recommendations']) + 1)
                                })
                                st.table(rec_df.set_index("Rank"))
                            else:
                                st.info("No recommendations available for this customer.")
                        
                        with history_tab:
                            if result['purchase_history']:
                                st.subheader("Previous Purchases")
                                history_df = pd.DataFrame({
                                    "Product": result['purchase_history']
                                })
                                st.table(history_df)
                            else:
                                st.info("No purchase history available for this customer.")
                        
                        # Add line break for visual separation
                        line_break()
                        line_break()
                        line_break()
                        line_break()
                        line_break()
                        
                        # Technical explanation section
                        st.markdown("## 🔍 How does the Recommendation System Work?")
                        st.markdown("### Technical Deep Dive")
                        
                        # Overview Section
                        st.markdown("""
                        <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                        <h3 style='color: #1e3d59;'>System Overview</h3>
                        <p style='color: #1e3d59;'>Our recommendation system uses advanced machine learning techniques, specifically an autoencoder neural network combined with collaborative filtering, to generate personalized product recommendations.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Data Representation Section
                        st.markdown("### 1️⃣ Data Representation")
                        col1, col2 = st.columns([2,1])
                        with col1:
                            st.markdown("""
                            The system uses a **User-Item Matrix** where:
                            - Rows → Individual customers (e.g., Cust0003)
                            - Columns → Available items (e.g., "Haircut (Men's)", "Highlights")
                            - Cells → Purchase quantities (0 if no purchase)
                            """)
                        with col2:
                            st.info("📊 This matrix is constructed from the purchase history data in purchases.csv")

                        # Data Processing Section
                        st.markdown("### 2️⃣ Data Processing")
                        st.markdown("""
                        #### Data Scaling
                        - Uses MinMaxScaler to transform purchase quantities to [0,1] range
                        - Essential for effective neural network training
                        
                        #### Encoding
                        - Customer IDs and item names are encoded using LabelEncoder
                        - Converts text data to numerical format for model processing
                        """)

                        # Autoencoder Architecture Section
                        st.markdown("### 3️⃣ Autoencoder Architecture")
                        arch_col1, arch_col2 = st.columns([3,2])
                        with arch_col1:
                            st.markdown("""
                            #### 🧠 Neural Network Components
                            1. **Encoder**
                               - Input: Customer's scaled purchase history
                               - Output: Compressed preference embedding
                            
                            2. **Decoder**
                               - Input: Compressed embedding
                               - Output: Reconstructed purchase preferences
                            """)
                        with arch_col2:
                            st.success("""
                            **Training Objective**
                            
                            Minimize the difference between:
                            - Original purchase history
                            - Reconstructed preferences
                            """)

                        # Recommendation Generation Section
                        st.markdown("### 4️⃣ Generating Recommendations")
                        st.markdown("""
                        #### Step-by-Step Process
                        
                        1. **Retrieve & Process User Data**
                           - Get customer's purchase history
                           - Scale the data using trained scaler
                        
                        2. **Generate Predictions**
                           - Encode purchase history to embedding
                           - Decode embedding to preference predictions
                           - Inverse transform predictions to original scale
                        
                        3. **Filter & Rank**
                           - Mask already purchased items
                           - Rank remaining items by predicted preference
                           - Select top N recommendations
                        """)

                        # Why It Works Section
                        st.markdown("""
                        <div style='background-color: #e6f3ff; color: #1e3d59; padding: 20px; border-radius: 10px; margin-top: 20px;'>
                        <h3 style='color: #1e3d59;'>💡 Why These Recommendations?</h3>
                        <p style='color: #1e3d59;'>The autoencoder learns complex patterns in customer behavior by:</p>
                        <ul style='color: #1e3d59;'>
                        <li>Identifying hidden relationships between products</li>
                        <li>Learning customer purchase patterns</li>
                        <li>Understanding product complementarity</li>
                        <li>Capturing seasonal and temporal trends</li>
                        </ul>
                        <p style='color: #1e3d59;'>When a customer like Cust0003 purchases certain items, the system recognizes patterns similar to other customers 
                        and recommends products that have been successful with similar customer profiles.</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Technical Benefits
                        st.markdown("### 🎯 Technical Benefits")
                        ben_col1, ben_col2 = st.columns(2)
                        with ben_col1:
                            st.info("""
                            **Advantages**
                            - Handles sparse data effectively
                            - Learns non-linear relationships
                            - Automatically extracts features
                            - Scales well with data size
                            """)
                        with ben_col2:
                            st.warning("""
                            **Considerations**
                            - Requires sufficient training data
                            - Needs periodic retraining
                            - Computationally intensive
                            - Cold-start challenges
                            """)
                        
                        # Add line break for visual separation
                        line_break()
                        line_break()
                        
                        # Technical Report Section
                        st.markdown("## 📑 Detailed Technical Report")
                        
                        # Create tabs for different sections of the report
                        report_tabs = st.tabs([
                            "📌 Introduction & Architecture",
                            "🔄 Algorithm",
                            "🔧 Implementation",
                            "📊 Results & Future"
                        ])
                        
                        with report_tabs[0]:
                            line_break()
                            line_break()
                            line_break()
                            line_break()
                            line_break()
                            # Introduction
                            st.markdown("""
                            <div style='background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                            <h3 style='color: #1e3d59;'>1. Introduction</h3>
                            <p style='color: #1e3d59;'>This report details the design, implementation, and evaluation of a product recommendation system for a retail company. 
                            The system aims to enhance customer experience by providing personalized product recommendations based on individual purchase histories. 
                            The system leverages an autoencoder, a type of neural network, to learn latent patterns in customer purchasing behavior and predict 
                            items that customers are likely to be interested in.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # System Architecture
                            st.markdown("### 2. System Architecture")
                            st.markdown("The system is designed with a modular, object-oriented architecture:")
                            
                            arch_cols = st.columns(2)
                            with arch_cols[0]:
                                st.info("""
                                **Core Components:**
                                
                                🔹 **DataLoader**
                                - Loads data from CSV files
                                - Handles missing files
                                - Basic error handling
                                
                                🔹 **DataPreprocessor**
                                - Data cleaning & transformation
                                - Feature engineering
                                - Handles missing values
                                - One-hot encoding
                                - Creates user-item matrix
                                
                                🔹 **RecommenderModel**
                                - TensorFlow autoencoder model
                                - Model architecture definition
                                - Training management
                                """)
                            
                            with arch_cols[1]:
                                st.info("""
                                **Additional Components:**
                                
                                🔹 **RecommenderSystem**
                                - Coordinates data and model
                                - Generates recommendations
                                - Handles cold-start cases
                                - Manages new customers
                                
                                🔹 **Evaluator**
                                - Performance metrics
                                - Precision@k calculation
                                - Recall@k calculation
                                - System evaluation
                                """)
                        
                        with report_tabs[1]:
                            # Algorithm Section
                            st.markdown("### 3. Algorithm: Autoencoder-Based Collaborative Filtering")
                            
                            # Core Algorithm
                            st.markdown("""
                            <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                            <p style='color: #1e3d59;'>The core recommendation algorithm uses an autoencoder to learn compressed representations of user preferences:</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            algo_cols = st.columns(2)
                            with algo_cols[0]:
                                st.success("""
                                **Encoder Process**
                                - Compresses user-item interaction vector
                                - Creates lower-dimensional embedding
                                - Captures latent factors
                                - Represents user preferences
                                """)
                            
                            with algo_cols[1]:
                                st.success("""
                                **Decoder Process**
                                - Reconstructs original vector
                                - Predicts user preferences
                                - Generates item scores
                                - Enables recommendations
                                """)
                            
                            st.markdown("#### Training Process")
                            st.markdown("""
                            - Uses Mean Squared Error (MSE) loss
                            - Optimizes network weights
                            - Learns optimal embeddings
                            - Minimizes reconstruction error
                            """)
                            
                            st.markdown("#### Recommendation Generation")
                            rec_steps = st.columns(4)
                            with rec_steps[0]:
                                st.info("1. Get user vector")
                            with rec_steps[1]:
                                st.info("2. Generate embedding")
                            with rec_steps[2]:
                                st.info("3. Reconstruct preferences")
                            with rec_steps[3]:
                                st.info("4. Rank & recommend")
                                
                            # Add comparison table
                            st.markdown("### 📊 Comparison: Hybrid vs Autoencoder Approach")
                            st.markdown("""
                            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                            <table style='width: 100%; border-collapse: collapse;'>
                                <thead>
                                    <tr style='background-color: #1e3d59; color: white;'>
                                        <th style='padding: 12px; text-align: left; border: 1px solid #ddd;'>Feature</th>
                                        <th style='padding: 12px; text-align: left; border: 1px solid #ddd;'>Hybrid</th>
                                        <th style='padding: 12px; text-align: left; border: 1px solid #ddd;'>Autoencoder</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr style='background-color: white; color: #1e3d59;'>
                                        <td style='padding: 12px; border: 1px solid #ddd;'><strong>Algorithm</strong></td>
                                        <td style='padding: 12px; border: 1px solid #ddd;'>Collaborative Filtering + Content-Based Filtering</td>
                                        <td style='padding: 12px; border: 1px solid #ddd;'>Autoencoder (Neural Network)</td>
                                    </tr>
                                    <tr style='background-color: #f8f9fa; color: #1e3d59;'>
                                        <td style='padding: 12px; border: 1px solid #ddd;'><strong>Interpretability</strong></td>
                                        <td style='padding: 12px; border: 1px solid #ddd;'>Higher (easier to understand why items are recommended)</td>
                                        <td style='padding: 12px; border: 1px solid #ddd;'>Lower ("black box" nature of neural networks)</td>
                                    </tr>
                                    <tr style='background-color: white; color: #1e3d59;'>
                                        <td style='padding: 12px; border: 1px solid #ddd;'><strong>Cold Start</strong></td>
                                        <td style='padding: 12px; border: 1px solid #ddd;'>Top-selling items</td>
                                        <td style='padding: 12px; border: 1px solid #ddd;'>Top-selling items</td>
                                    </tr>
                                    <tr style='background-color: #f8f9fa; color: #1e3d59;'>
                                        <td style='padding: 12px; border: 1px solid #ddd;'><strong>Scalability</strong></td>
                                        <td style='padding: 12px; border: 1px solid #ddd;'>Can become slow with a huge number of users (collaborative part) or items (content-based part)</td>
                                        <td style='padding: 12px; border: 1px solid #ddd;'>Generally scales better, especially for very large datasets. Recommendation generation is fast after training</td>
                                    </tr>
                                    <tr style='background-color: white; color: #1e3d59;'>
                                        <td style='padding: 12px; border: 1px solid #ddd;'><strong>Data Used</strong></td>
                                        <td style='padding: 12px; border: 1px solid #ddd;'>Purchases (collaborative), Purchases + Item Features (content-based)</td>
                                        <td style='padding: 12px; border: 1px solid #ddd;'>Primarily Purchases (user-item interactions)</td>
                                    </tr>
                                    <tr style='background-color: #f8f9fa; color: #1e3d59;'>
                                        <td style='padding: 12px; border: 1px solid #ddd;'><strong>Complexity</strong></td>
                                        <td style='padding: 12px; border: 1px solid #ddd;'>Moderate (calculating similarities can be expensive)</td>
                                        <td style='padding: 12px; border: 1px solid #ddd;'>Higher (training the neural network can be time-consuming), but prediction is fast</td>
                                    </tr>
                                    <tr style='background-color: white; color: #1e3d59;'>
                                        <td style='padding: 12px; border: 1px solid #ddd;'><strong>Flexibility</strong></td>
                                        <td style='padding: 12px; border: 1px solid #ddd;'>Easier to incorporate additional features. Weights between collaborative and content-based can be tuned</td>
                                        <td style='padding: 12px; border: 1px solid #ddd;'>Less flexible for incorporating side information directly. Requires more careful feature engineering</td>
                                    </tr>
                                    <tr style='background-color: #f8f9fa; color: #1e3d59;'>
                                        <td style='padding: 12px; border: 1px solid #ddd;'><strong>Implicit Feedback</strong></td>
                                        <td style='padding: 12px; border: 1px solid #ddd;'>Handles implicit feedback (like purchase quantity) well</td>
                                        <td style='padding: 12px; border: 1px solid #ddd;'>Handles implicit feedback well (designed for this type of data)</td>
                                    </tr>
                                    <tr style='background-color: white; color: #1e3d59;'>
                                        <td style='padding: 12px; border: 1px solid #ddd;'><strong>"Serendipity"</strong></td>
                                        <td style='padding: 12px; border: 1px solid #ddd;'>Can potentially lead to less "surprising" recommendations (echo chamber effect)</td>
                                        <td style='padding: 12px; border: 1px solid #ddd;'>Can potentially discover more unexpected, relevant recommendations due to learned latent representations</td>
                                    </tr>
                                </tbody>
                            </table>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("#### Training Process")
                            st.markdown("""
                            - Uses Mean Squared Error (MSE) loss
                            - Optimizes network weights
                            - Learns optimal embeddings
                            - Minimizes reconstruction error
                            """)
                        
                        with report_tabs[2]:
                            # Implementation Details
                            st.markdown("### 4. Data Preprocessing")
                            preproc_cols = st.columns(2)
                            with preproc_cols[0]:
                                st.warning("""
                                **Data Cleaning**
                                - Handle missing values
                                - Convert data types
                                - One-hot encoding
                                - Create interaction matrix
                                """)
                            
                            with preproc_cols[1]:
                                st.warning("""
                                **Feature Engineering**
                                - Scale features
                                - Label encoding
                                - Date processing
                                - Matrix normalization
                                """)
                            
                            st.markdown("### 5. Implementation Stack")
                            tech_cols = st.columns(3)
                            with tech_cols[0]:
                                st.success("""
                                **Core Stack**
                                - Python
                                - TensorFlow
                                - Pandas
                                - NumPy
                                """)
                            
                            with tech_cols[1]:
                                st.success("""
                                **ML Tools**
                                - Scikit-learn
                                - Keras
                                - Datetime
                                - Logging
                                """)
                            
                            with tech_cols[2]:
                                st.success("""
                                **Code Structure**
                                - Modular design
                                - Type hinting
                                - Error handling
                                - Documentation
                                """)
                        
                        with report_tabs[3]:
                            # Results and Future Improvements
                            st.markdown("### 6. Results")
                            st.markdown("""
                            <div style='background-color: #f5f5f5; color: #1e3d59; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                            <ul style='color: #1e3d59;'>
                            <li>Precision and recall metrics calculated for recommendations</li>
                            <li>Fallback to top-selling items for new users</li>
                            <li>Cold-start handling implemented</li>
                            <li>Continuous evaluation system in place</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("### 7. Future Improvements")
                            future_cols = st.columns(2)
                            with future_cols[0]:
                                st.info("""
                                **Technical Improvements**
                                - Hyperparameter tuning
                                - Item metadata integration
                                - Denoising autoencoder
                                - Sequence-aware models
                                """)
                            
                            with future_cols[1]:
                                st.info("""
                                **Business Improvements**
                                - A/B testing
                                - Advanced metrics
                                - Scalability solutions
                                - Business KPI integration
                                """)
                            
                            st.markdown("### 8. Conclusion")
                            st.markdown("""
                            <div style='background-color: #e6f3ff; padding: 20px; border-radius: 10px; margin-top: 20px;'>
                            <p style='color: #1e3d59;'>The implemented product recommendation system provides a solid foundation for personalized recommendations 
                            based on customer purchase history. The autoencoder-based approach effectively captures latent patterns in 
                            purchasing behavior, and the modular design allows for future extensions and improvements.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No recommendations could be generated. The customer ID may not exist.")
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")


# Add footer with attribution
line_break()
line_break()
line_break()
st.markdown("""
<div style='text-align: center; padding: 20px; position: fixed; bottom: 0; left: 0; right: 0; background-color: #e6e6fa; color: #000080;'>
    <p>Submitted by Ashar</p>
</div>
""", unsafe_allow_html=True)


