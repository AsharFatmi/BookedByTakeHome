# BookedBy Take-Home Assignment

A comprehensive data science project built with Python, featuring three main components:
1. Data Analytics and Visualization
2. Customer Segmentation Analysis
3. Product Recommendation System

The project demonstrates capabilities in data processing, machine learning, and web application development using Streamlit, TensorFlow, and scikit-learn.

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BookedByTakeHome.git
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

### Data Analytics Pipeline (`