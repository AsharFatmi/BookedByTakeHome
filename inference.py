import os
import tensorflow as tf
import joblib
import logging
import pandas as pd
import numpy as np
from typing import List, Optional
from config_manager import ConfigManager
from recommendation_engine import DataLoader, DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommenderInference:
    def __init__(self, model_dir: str):
        """Initialize the recommender inference system
        
        Args:
            model_dir: Directory containing saved model artifacts
        """
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory {model_dir} does not exist")
        self.model_dir = model_dir
        self._load_artifacts()
        
    def _load_artifacts(self):
        """Load all saved model artifacts"""
        try:
            # Load metadata
            metadata_path = os.path.join(self.model_dir, "metadata.joblib")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
            self.metadata = joblib.load(metadata_path)
            
            # Load configuration
            config_path = os.path.join(self.model_dir, "config.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            self.config = ConfigManager(config_path)
            
            # Load preprocessor artifacts
            self.preprocessor = DataPreprocessor(self.config)
            artifact_files = {
                "scaler": "scaler.joblib",
                "user_encoder": "user_encoder.joblib",
                "item_encoder": "item_encoder.joblib",
                "feature_range": "feature_range.joblib"
            }
            
            for attr, filename in artifact_files.items():
                filepath = os.path.join(self.model_dir, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"{attr.capitalize()} file not found at {filepath}")
                setattr(self.preprocessor, attr, joblib.load(filepath))
            
            # Load TensorFlow models with .keras extension
            autoencoder_path = os.path.join(self.model_dir, "autoencoder.keras")
            encoder_path = os.path.join(self.model_dir, "encoder.keras")
            
            if not os.path.exists(autoencoder_path):
                raise FileNotFoundError(f"Autoencoder model not found at {autoencoder_path}")
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Encoder model not found at {encoder_path}")
                
            self.autoencoder = tf.keras.models.load_model(autoencoder_path)
            self.encoder = tf.keras.models.load_model(encoder_path)
            
            # Load data
            data_loader = DataLoader("customers.csv", "items.csv", "purchases.csv", self.config)
            self.customers_df, self.items_df, self.purchases_df = data_loader.load_data()
            
            # Create user-item matrix for looking up previous purchases
            self.customer_item_matrix = self.preprocessor.create_user_item_matrix(self.purchases_df)
            
            logger.info("Successfully loaded all model artifacts")
            
        except Exception as e:
            logger.error(f"Error loading model artifacts: {str(e)}")
            raise
        
    def get_user_purchase_history(self, customer_id: str) -> List[str]:
        """Get the purchase history for a user
        
        Args:
            customer_id: ID of the customer
            
        Returns:
            List of item names previously purchased by the customer
        """
        if customer_id not in self.customer_item_matrix.index:
            return []
            
        purchases = self.customer_item_matrix.loc[customer_id]
        purchased_items = purchases[purchases > 0].index.tolist()
        return self.items_df[self.items_df['ItemID'].isin(purchased_items)]['ItemName'].tolist()
        
    def recommend_for_user(self, customer_id: str, num_recommendations: Optional[int] = None) -> List[str]:
        """Generate recommendations for a user
        
        Args:
            customer_id: ID of the customer
            num_recommendations: Number of recommendations to generate
            
        Returns:
            List of recommended item names
        """
        if num_recommendations is None:
            num_recommendations = self.config.evaluation_config['top_k_recommendations']
            
        if customer_id not in self.preprocessor.user_encoder.classes_:
            logger.info(f"Customer {customer_id} not found. Recommending top selling items.")
            return self._get_top_selling_items(num_recommendations)
            
        # Create user vector
        user_vector = np.zeros((1, len(self.preprocessor.item_encoder.classes_)))
        if customer_id in self.customer_item_matrix.index:
            purchases = self.customer_item_matrix.loc[customer_id]
            for item_id, quantity in purchases[purchases > 0].items():
                if item_id in self.preprocessor.item_encoder.classes_:
                    idx = self.preprocessor.item_encoder.transform([item_id])[0]
                    user_vector[0, idx] = quantity
                    
        # Scale the user vector
        user_vector_scaled = self.preprocessor.scaler.transform(user_vector)
        
        # Get predictions
        reconstructed_user = self.autoencoder.predict(user_vector_scaled)
        reconstructed_user = self.preprocessor.inverse_transform_predictions(reconstructed_user)
        
        # Mask already purchased items
        already_purchased = self._get_user_purchased_items(customer_id)
        mask = np.ones(reconstructed_user.shape[1], dtype=bool)
        if already_purchased:
            purchased_indices = [self.preprocessor.item_encoder.transform([item_id])[0] 
                               for item_id in already_purchased]
            mask[purchased_indices] = False
        reconstructed_user[0, ~mask] = -np.inf
        
        # Get recommendations
        available_indices = np.where(mask)[0]
        sorted_indices = available_indices[np.argsort(reconstructed_user[0, available_indices])[::-1]]
        recommended_indices = sorted_indices[:num_recommendations]
        recommended_item_ids = self.preprocessor.item_encoder.inverse_transform(recommended_indices)
        
        return self.items_df[self.items_df['ItemID'].isin(recommended_item_ids)]['ItemName'].tolist()
        
    def _get_user_purchased_items(self, customer_id: str) -> List[str]:
        """Get items already purchased by user"""
        if customer_id in self.customer_item_matrix.index:
            purchases = self.customer_item_matrix.loc[customer_id]
            return purchases[purchases > 0].index.tolist()
        return []
        
    def _get_top_selling_items(self, num_recommendations: int) -> List[str]:
        """Get top selling items"""
        top_selling = self.customer_item_matrix.sum().nlargest(num_recommendations).index
        return self.items_df[self.items_df['ItemID'].isin(top_selling)]['ItemName'].tolist()

def main(customer_ids):
    # Get the latest model directory
    model_base_dir = "saved_models"
    model_dirs = [d for d in os.listdir(model_base_dir) if os.path.isdir(os.path.join(model_base_dir, d))]
    latest_model_dir = os.path.join(model_base_dir, sorted(model_dirs)[-1])
    
    # Initialize recommender
    recommender = RecommenderInference(latest_model_dir)
    
    
    
    results = []
    for customer_id in customer_ids:
        # Get purchase history and recommendations
        purchase_history = recommender.get_user_purchase_history(customer_id)
        recommendations = recommender.recommend_for_user(customer_id)
        
        # Build result object for this customer
        customer_result = {
            "customer_id": customer_id,
            "purchase_history": purchase_history,
            "recommendations": recommendations
        }
        results.append(customer_result)
    
    return results

        

if __name__ == "__main__":

    # Example customers
    customer_ids = ['Cust0001', 'Cust0002', 'Cust0003']
    result = main(customer_ids) 
    print(result)