import os
import pandas as pd
import tensorflow as tf
from datetime import datetime
import logging
import yaml
from config_manager import ConfigManager
from recommendation_engine import DataLoader, DataPreprocessor, RecommenderModel, Evaluator
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_training_artifacts(model: RecommenderModel, preprocessor: DataPreprocessor, 
                          config: ConfigManager, model_dir: str = "saved_models"):
    """Save the trained model and necessary artifacts"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(model_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the TensorFlow models with proper extensions
    model.autoencoder.save(os.path.join(save_dir, "autoencoder.keras"))
    model.encoder.save(os.path.join(save_dir, "encoder.keras"))
    
    # Save preprocessor artifacts
    joblib.dump(preprocessor.scaler, os.path.join(save_dir, "scaler.joblib"))
    joblib.dump(preprocessor.user_encoder, os.path.join(save_dir, "user_encoder.joblib"))
    joblib.dump(preprocessor.item_encoder, os.path.join(save_dir, "item_encoder.joblib"))
    joblib.dump(preprocessor.feature_range, os.path.join(save_dir, "feature_range.joblib"))
    
    # Save the configuration
    config_path = os.path.join(save_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config._config, f)
    
    # Save model metadata
    metadata = {
        'timestamp': timestamp,
        'num_items': model.num_items,
        'model_path': save_dir
    }
    joblib.dump(metadata, os.path.join(save_dir, "metadata.joblib"))
    
    logger.info(f"Model and artifacts saved to {save_dir}")
    return save_dir

def train_model():
    # Initialize configuration
    config = ConfigManager("config.yaml")
    
    # Initialize components with config
    data_loader = DataLoader("customers.csv", "items.csv", "purchases.csv", config)
    preprocessor = DataPreprocessor(config)
    
    # Load and preprocess data
    customers_df, items_df, purchases_df = data_loader.load_data()
    customers_df = preprocessor.preprocess_customers(customers_df)
    items_df = preprocessor.preprocess_items(items_df)
    purchases_df = preprocessor.preprocess_purchases(purchases_df)
    
    # Create and process matrices
    customer_item_matrix = preprocessor.create_user_item_matrix(purchases_df)
    customer_item_matrix_scaled = preprocessor.scale_matrix(customer_item_matrix)
    customer_item_matrix_scaled = preprocessor.encode_ids(customers_df, items_df, customer_item_matrix_scaled)
    
    # Initialize model with config
    model = RecommenderModel(num_items=len(items_df), config=config)
    
    # Create train/test split using configured test period
    latest_purchase_date = purchases_df['PurchaseDate'].max()
    test_months = config.evaluation_config['test_period_months']
    test_start_date = latest_purchase_date - pd.DateOffset(months=test_months)
    
    test_customers = purchases_df[purchases_df['PurchaseDate'] >= test_start_date]['CustomerID'].unique()
    train_customers = purchases_df[purchases_df['PurchaseDate'] < test_start_date]['CustomerID'].unique()
    
    X_train = customer_item_matrix_scaled.loc[preprocessor.user_encoder.transform(train_customers)]
    X_test = customer_item_matrix_scaled.loc[preprocessor.user_encoder.transform(test_customers)]
    
    # Handle missing customers
    X_train = X_train.reindex(range(len(preprocessor.user_encoder.classes_)), 
                             fill_value=config.preprocessing_config['fill_value'])
    X_test = X_test.reindex(range(len(preprocessor.user_encoder.classes_)), 
                           fill_value=config.preprocessing_config['fill_value'])
    
    # Train the model
    logger.info("Starting model training...")
    history = model.train(X_train.values, X_test.values)
    logger.info("Model training completed")
    
    # Evaluate the model
    evaluator = Evaluator(config)
    avg_precision, avg_recall = evaluator.evaluate_model(
        model, preprocessor, items_df, customer_item_matrix, 
        customer_item_matrix_scaled, purchases_df, test_start_date
    )
    logger.info(f"Average Precision@{evaluator.top_k}: {avg_precision:.4f}")
    logger.info(f"Average Recall@{evaluator.top_k}: {avg_recall:.4f}")
    
    # Save the model and artifacts
    save_dir = save_training_artifacts(model, preprocessor, config)
    logger.info(f"Training artifacts saved to: {save_dir}")
    
    return save_dir, history

if __name__ == "__main__":
    save_dir, history = train_model() 