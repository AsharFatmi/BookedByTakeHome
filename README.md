# BookedBy Take-Home Assignment

A comprehensive data science project built with Python, featuring three main components:
1. Data Analytics and Visualization
2. Customer Segmentation Analysis
3. Product Recommendation System

The project demonstrates capabilities in data processing, machine learning, and web application development using Streamlit, TensorFlow, and scikit-learn.

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/AsharFatmi/BookedByTakeHome.git
cd BookedByTakeHome
```

2. Create Conda environment from environment.yml:
```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate the environment
conda activate bookedby
```

3. Verify Installation:
```bash
# Check if all packages are installed correctly
conda list

# Test TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

Note: Using Conda with environment.yml ensures consistent environment setup across different systems and better dependency management.

## 🚀 Usage

1. Start the Streamlit application:
```bash
streamlit run Home.py
```

2. Navigate through the different pages:
   - **Home**: Overview and introduction
   - **Basic Analysis**: View sales trends and product performance
   - **Customer Segmentation**: Explore customer segments and behaviors
   - **Product Recommendation**: Generate personalized recommendations

3. Key Features per Page:
   - **Data Analytics**:
     - View sales trends and patterns
     - Analyze revenue by category
     - Identify top/bottom performing products
   
   - **Customer Segmentation**:
     - Explore 5 distinct customer segments
     - View detailed segment profiles
     - Access marketing recommendations
   
   - **Product Recommendation**:
     - Select customer ID from dropdown
     - View personalized recommendations
     - Explore purchase history

## 🌟 Features

### 1. Data Analytics
- **Sales Analysis**: Comprehensive analysis of sales trends and patterns
- **Revenue Insights**: Detailed breakdown of revenue by various dimensions
- **Product Performance**: Analysis of top and bottom performing products
- **Interactive Visualizations**: Dynamic charts and graphs for better insights

### 2. Customer Segmentation
- **K-means Clustering**: Advanced customer segmentation using behavioral data
- **Segment Profiling**: Detailed analysis of each customer segment
- **Visual Analysis**: Interactive visualizations of customer clusters
- **Actionable Insights**: Marketing and strategy recommendations per segment

### 3. Product Recommendation System
- **Personalized Recommendations**: AI-powered product suggestions using autoencoder
- **Purchase History Analysis**: Detailed view of customer purchase patterns
- **Real-time Processing**: Fast recommendation generation
- **Interactive Interface**: User-friendly recommendation exploration

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Data Requirements](#data-requirements)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)

## 📁 Project Structure

```
BookedByTakeHome/
├── data/
│   ├── customers.csv        # Customer information
│   ├── items.csv           # Product catalog
│   └── purchases.csv       # Purchase history
├── pages/
│   ├── 1_Option_1.py      # Data Analytics Dashboard
│   ├── 2_Customer_Segmentation.py  # Customer Segmentation Analysis
│   └── 3_Product_Recommendation.py # Product Recommendation System
├── saved_models/          # Directory for trained models
├── analysis_results.json  # Cached analysis results
├── cluster_analysis.py    # Customer segmentation logic
├── data_analysis.py      # Data analysis functions
├── inference.py         # Recommendation inference module
├── Home.py             # Main application entry point
└── requirements.txt    # Project dependencies
```

## 🔬 Technical Details

### Data Analytics Pipeline (`data_analysis.py`)

1. **Data Analysis** (`DataAnalyzer class`)
   - Sales trend analysis
   - Revenue calculations
   - Product performance metrics
   - Functions: `analyze_sales()`, `calculate_revenue()`, `get_product_metrics()`

2. **Visualization** (`DataAnalyzer class`)
   - Interactive charts
   - Performance dashboards
   - Trend visualizations
   - Functions: `plot_sales_trend()`, `create_revenue_charts()`

### Customer Segmentation (`cluster_analysis.py`)

1. **Preprocessing** (`ClusterAnalyzer class`)
   - Feature engineering
   - Data normalization
   - Customer metric calculations
   - Functions: `prepare_features()`, `calculate_customer_metrics()`

2. **Clustering** (`ClusterAnalyzer class`)
   - K-means implementation
   - Cluster optimization
   - Segment profiling
   - Functions: `perform_clustering()`, `analyze_segments()`

### Recommendation System (`inference.py`)

1. **Model Architecture** (`RecommenderInference class`)
   - Autoencoder implementation
   - Neural network layers
   - Training configuration
   - Functions: `build_model()`, `train_model()`

2. **Inference Pipeline** (`RecommenderInference class`)
   - Input processing
   - Recommendation generation
   - Results filtering
   - Functions: `generate_recommendations()`, `process_results()`

## 📊 Data Requirements

Required CSV files:
- **customers.csv**:
  - CustomerID (string)
  - Demographics
  - Subscription info

- **items.csv**:
  - ItemID (string)
  - ItemName (string)
  - Category
  - Price

- **purchases.csv**:
  - CustomerID (string)
  - ItemID (string)
  - Quantity (numeric)
  - PurchaseDate (datetime)
  - Amount (numeric)

## 🧠 Model Architecture

The recommendation system uses an autoencoder neural network:

```python
Input Layer (n_items)
    ↓
Encoder Layers
    ↓
Latent Space (embedding)
    ↓
Decoder Layers
    ↓
Output Layer (n_items)
```

Key components:
- Dense layers with ReLU activation
- Dropout for regularization
- MSE loss function
- Adam optimizer

## 👥 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👤 Author

- Ashar Fatmi

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Streamlit team for the web interface framework
- scikit-learn team for machine learning utilities
