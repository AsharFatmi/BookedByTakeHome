import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow import keras
from keras import regularizers
from typing import List, Tuple, Optional
from datetime import datetime
import logging
from config_manager import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Responsible for loading customer, item, and purchase data from CSV files."""
    def __init__(self, customers_path: str, items_path: str, purchases_path: str, config: ConfigManager):
        self.customers_path = customers_path
        self.items_path = items_path
        self.purchases_path = purchases_path
        self.config = config
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and perform initial data cleaning"""
        try:
            customers = pd.read_csv(self.customers_path)
            items = pd.read_csv(self.items_path)
            purchases = pd.read_csv(self.purchases_path)
            return customers, items, purchases
        except FileNotFoundError as e:
            logger.error(f"Error loading data files: {e}")
            raise

class DataPreprocessor:
    """Handles data preprocessing for the recommendation system, including encoding, scaling, 
    and creating user-item interaction matrices."""
    def __init__(self, config: ConfigManager):
        self.config = config
        self.scaler = MinMaxScaler()  # Used to normalize numerical features
        self.user_encoder = LabelEncoder()  # Encodes user IDs to integers
        self.item_encoder = LabelEncoder()  # Encodes item IDs to integers
        self.unknown_category = config.preprocessing_config['unknown_category']
        self.fill_value = config.preprocessing_config['fill_value']
        self.feature_range = None  # Store the feature range for inverse transform
        self.feature_names = None  # Store feature names for consistent scaling

    def preprocess_customers(self, customers: pd.DataFrame) -> pd.DataFrame:
        """Preprocess customer data by filling missing values and encoding categorical features"""
        customers = customers.copy()
        # Fill missing age with mean and missing gender with unknown category
        customers.fillna({'Age': customers['Age'].mean(), 'Gender': self.unknown_category}, inplace=True)
        customers['RegistrationDate'] = pd.to_datetime(customers['RegistrationDate'])
        # Convert boolean subscription fields to integers
        customers['EmailSubscription'] = customers['EmailSubscription'].astype(int)
        customers['SMSSubscription'] = customers['SMSSubscription'].astype(int)
        # One-hot encode gender
        customers = pd.get_dummies(customers, columns=['Gender'], prefix=['Gender'], dummy_na=False)
        return customers

    def preprocess_purchases(self, purchases: pd.DataFrame) -> pd.DataFrame:
        """Preprocess purchase data by converting date strings to datetime objects"""
        purchases = purchases.copy()
        purchases['PurchaseDate'] = pd.to_datetime(purchases['PurchaseDate'])
        return purchases

    def preprocess_items(self, items: pd.DataFrame) -> pd.DataFrame:
        """Preprocess item data by filling missing values"""
        items = items.copy()
        items.fillna("Unknown", inplace=True)
        return items

    def create_user_item_matrix(self, purchases: pd.DataFrame) -> pd.DataFrame:
        """Create user-item interaction matrix where rows are users, columns are items,
        and values represent the quantity purchased"""
        matrix = purchases.pivot_table(
            index='CustomerID', 
            columns='ItemID', 
            values='Quantity', 
            aggfunc='sum', 
            fill_value=self.fill_value
        )
        
        # Filter based on minimum interactions if configured
        # This helps reduce sparsity and focus on users with sufficient data
        min_interactions = self.config.preprocessing_config['min_interactions']
        if min_interactions > 0:
            user_interactions = (matrix > 0).sum(axis=1)
            matrix = matrix[user_interactions >= min_interactions]
            
        return matrix

    def scale_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Scale the user-item matrix to normalize values for model training"""
        # Store feature names for consistent scaling
        self.feature_names = matrix.columns.tolist()
        
        # Fit and transform the data
        scaled_values = self.scaler.fit_transform(matrix)
        # Store the feature range for later use in inverse transformation
        self.feature_range = (self.scaler.data_min_, self.scaler.data_max_)
        return pd.DataFrame(scaled_values, index=matrix.index, columns=self.feature_names)

    def transform_vector(self, vector: np.ndarray) -> np.ndarray:
        """Transform a vector using the fitted scaler with consistent feature handling"""
        if self.feature_names is None:
            raise ValueError("Scaler has not been fitted. Call scale_matrix first.")
            
        # Convert numpy array to DataFrame with correct feature names
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        vector_df = pd.DataFrame(vector, columns=self.feature_names)
        
        # Transform and return as numpy array
        return self.scaler.transform(vector_df)

    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Inverse transform predictions to the original scale using stored feature range"""
        if self.feature_range is None:
            raise ValueError("Scaler has not been fitted. Call scale_matrix first.")
        
        data_min, data_max = self.feature_range
        # Handle the case where some features might have zero range
        scale = (data_max - data_min)
        scale[scale == 0.0] = 1.0
        
        # Manual inverse transform to handle shape mismatches
        return predictions * scale + data_min

    def encode_ids(self, customers: pd.DataFrame, items: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
        """Encode user and item IDs to integer indices for model consumption"""
        self.user_encoder.fit(customers['CustomerID'])
        self.item_encoder.fit(items['ItemID'])
        
        matrix = matrix.copy()
        # Create a mapping dictionary for quick lookups
        user_id_map = {id_: idx for idx, id_ in enumerate(self.user_encoder.classes_)}
        item_id_map = {id_: idx for idx, id_ in enumerate(self.item_encoder.classes_)}
        
        # Create a new matrix with encoded indices
        encoded_matrix = pd.DataFrame(
            np.zeros((len(user_id_map), len(item_id_map))),
            index=range(len(user_id_map)),
            columns=range(len(item_id_map))
        )
        
        # Fill the encoded matrix with values from the original matrix
        for user_id in matrix.index:
            for item_id in matrix.columns:
                if user_id in user_id_map and item_id in item_id_map:
                    encoded_matrix.loc[user_id_map[user_id], item_id_map[item_id]] = matrix.loc[user_id, item_id]
        
        return encoded_matrix

class RecommenderModel:
    """Implements the autoencoder-based collaborative filtering model for recommendations.
    The autoencoder learns to compress user preference vectors into a lower-dimensional space
    and reconstruct them, enabling the discovery of latent features and patterns."""
    def __init__(self, num_items: int, config: ConfigManager):
        self.num_items = num_items
        self.config = config
        self.model_config = config.model_config
        self.training_config = config.training_config
        self.reg_config = config.regularization_config
        self.autoencoder, self.encoder = self._create_autoencoder()

    def _create_autoencoder(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """Create the autoencoder architecture with configurable layers and regularization"""
        input_layer = tf.keras.layers.Input(shape=(self.num_items,), name='item_input')
        
        # Get configuration values
        embedding_dim = self.model_config['embedding_dim']
        dropout_rate = self.model_config['dropout_rate']
        hidden_layers = self.model_config['hidden_layer_sizes']
        activation = self.model_config['activation']
        output_activation = self.model_config['output_activation']
        l2_strength = self.reg_config['l2_strength']
        
        # Encoder - compresses input into a lower-dimensional latent representation
        encoded = input_layer
        for multiplier in hidden_layers[:-1]:
            encoded = tf.keras.layers.Dense(
                embedding_dim * multiplier,
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_strength)  # L2 regularization to prevent overfitting
            )(encoded)
            encoded = tf.keras.layers.Dropout(dropout_rate)(encoded)  # Dropout for regularization
        
        # Final encoding layer - the bottleneck that captures the latent representation
        encoded = tf.keras.layers.Dense(
            embedding_dim * hidden_layers[-1],
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_strength)
        )(encoded)

        # Decoder - reconstructs the original input from the latent representation
        decoded = encoded
        for multiplier in reversed(hidden_layers[:-1]):
            decoded = tf.keras.layers.Dense(
                embedding_dim * multiplier,
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_strength)
            )(decoded)
            decoded = tf.keras.layers.Dropout(dropout_rate)(decoded)
        
        # Output layer - reconstructs the user-item interaction vector
        decoded = tf.keras.layers.Dense(
            self.num_items,
            activation=output_activation
        )(decoded)

        # Create the complete autoencoder model and the encoder component
        autoencoder = tf.keras.Model(input_layer, decoded)
        encoder = tf.keras.Model(input_layer, encoded)
        
        # Compile model with configured learning rate
        learning_rate = self.training_config['learning_rate']
        autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse'  # Mean squared error loss for reconstruction quality
        )
        return autoencoder, encoder

    def train(self, X_train: np.ndarray, X_test: np.ndarray) -> tf.keras.callbacks.History:
        """Train the autoencoder model with early stopping to prevent overfitting"""
        callbacks = []
        
        # Add early stopping if configured
        if self.training_config.get('early_stopping_patience', 0) > 0:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.training_config['early_stopping_patience'],
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        return self.autoencoder.fit(
            X_train, X_train,  # Autoencoder learns to reconstruct its own input
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            validation_data=(X_test, X_test),
            callbacks=callbacks,
            verbose=1
        )

class RecommenderSystem:
    """Combines the trained model and preprocessor to generate recommendations for users.
    Handles both cold-start scenarios (new users) and recommendations for existing users."""
    def __init__(self, model: RecommenderModel, preprocessor: DataPreprocessor, config: ConfigManager):
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.items_df = None
        self.customer_item_matrix = None
        self.customer_item_matrix_scaled = None

    def set_data(self, items_df: pd.DataFrame, customer_item_matrix: pd.DataFrame, 
                 customer_item_matrix_scaled: pd.DataFrame):
        """Set the necessary data for making recommendations"""
        self.items_df = items_df
        self.customer_item_matrix = customer_item_matrix
        self.customer_item_matrix_scaled = customer_item_matrix_scaled

    def recommend_for_user(self, customer_id: str, num_recommendations: Optional[int] = None) -> List[str]:
        """Generate recommendations for a specific user based on their purchase history.
        For new users or those with no history, falls back to recommending top-selling items."""
        if num_recommendations is None:
            num_recommendations = self.config.evaluation_config['top_k_recommendations']
            
        # Handle new users (cold start problem)
        if customer_id not in self.preprocessor.user_encoder.classes_:
            logger.info(f"Customer {customer_id} not found. Recommending top selling items.")
            return self._get_top_selling_items(num_recommendations)

        encoded_user_id = self.preprocessor.user_encoder.transform([customer_id])[0]
        
        # Get user vector from interaction matrix
        if encoded_user_id in self.customer_item_matrix_scaled.index:
            user_vector = self.customer_item_matrix_scaled.loc[encoded_user_id].values
        else:
            logger.info(f"Customer {customer_id} has no purchase history. Recommending top selling items.")
            return self._get_top_selling_items(num_recommendations)

        # Transform the user vector to the scaled space
        user_vector_scaled = self.preprocessor.transform_vector(user_vector)

        # Get predictions from the model - reconstruct user preferences
        reconstructed_user = self.model.autoencoder.predict(user_vector_scaled)
        
        # Transform predictions back to original scale
        reconstructed_user = self.preprocessor.inverse_transform_predictions(reconstructed_user)

        # Get already purchased items to exclude from recommendations
        already_purchased = self._get_user_purchased_items(customer_id)
        
        # Create mask for filtering out purchased items
        mask = np.ones(reconstructed_user.shape[1], dtype=bool)
        if already_purchased:
            purchased_indices = [self.preprocessor.item_encoder.transform([item_id])[0] for item_id in already_purchased]
            mask[purchased_indices] = False
        reconstructed_user[0, ~mask] = -np.inf  # Exclude purchased items from recommendations

        # Get recommendations based on highest predicted preferences
        available_indices = np.where(mask)[0]
        sorted_indices = available_indices[np.argsort(reconstructed_user[0, available_indices])[::-1]]
        recommended_indices = sorted_indices[:num_recommendations]
        recommended_item_ids = self.preprocessor.item_encoder.inverse_transform(recommended_indices)
        
        return self.items_df[self.items_df['ItemID'].isin(recommended_item_ids)]['ItemName'].tolist()

    def _get_top_selling_items(self, num_recommendations: int) -> List[str]:
        """Get top selling items as a fallback recommendation strategy"""
        top_selling = self.customer_item_matrix.sum().nlargest(num_recommendations).index
        return self.items_df[self.items_df['ItemID'].isin(top_selling)]['ItemName'].tolist()

    def _get_user_purchased_items(self, customer_id: str) -> List[str]:
        """Get items already purchased by user using original IDs"""
        if customer_id in self.customer_item_matrix.index:
            purchases = self.customer_item_matrix.loc[customer_id]
            return purchases[purchases > 0].index.tolist()
        return []

    def add_new_customer(self, customer_id: str, age: int, email_sub: int, sms_sub: int, 
                        registration_date: datetime, contact: str):
        """Add a new customer to the system with default zero interactions"""
        # Update customers DataFrame
        new_customer = pd.DataFrame({
            'CustomerID': [customer_id],
            'Age': [age],
            'EmailSubscription': [email_sub],
            'SMSSubscription': [sms_sub],
            'RegistrationDate': [registration_date],
            'ContactNumber': [contact]
        })
        
        # Add to matrices with zero interactions
        self.customer_item_matrix.loc[customer_id] = 0
        self.customer_item_matrix_scaled = pd.DataFrame(
            self.preprocessor.scaler.fit_transform(self.customer_item_matrix),
            index=self.customer_item_matrix.index,
            columns=self.customer_item_matrix.columns
        )
        
        # Update encoders with the new user ID
        self.preprocessor.user_encoder.fit(list(self.preprocessor.user_encoder.classes_) + [customer_id])

class Evaluator:
    """Handles model evaluation using metrics like precision and recall
    to assess recommendation quality."""
    def __init__(self, config: ConfigManager):
        self.config = config
        self.top_k = config.evaluation_config['top_k_recommendations']
        self.test_period_months = config.evaluation_config['test_period_months']

    def calculate_precision_recall(self, recommendations: List[str], actual_purchases: List[str]) -> Tuple[float, float]:
        """Calculate precision and recall metrics for a single user's recommendations.
        
        Precision: Fraction of recommended items that were actually purchased
        Recall: Fraction of actually purchased items that were recommended
        """
        if not actual_purchases:
            return 0.0, 0.0
            
        recommended_set = set(recommendations[:self.top_k])
        actual_set = set(actual_purchases)
        intersection = recommended_set.intersection(actual_set)
        
        precision = len(intersection) / self.top_k if self.top_k > 0 else 0.0
        recall = len(intersection) / len(actual_set) if actual_set else 0.0
        
        return precision, recall

    def evaluate_model(self, model: RecommenderModel, preprocessor: DataPreprocessor,
                      items_df: pd.DataFrame, customer_item_matrix: pd.DataFrame,
                      customer_item_matrix_scaled: pd.DataFrame,
                      test_purchases: pd.DataFrame, test_start_date: pd.Timestamp) -> Tuple[float, float]:
        """Evaluate the model's performance on test data by comparing recommendations
        with actual future purchases."""
        # Get customers who made purchases in the test period
        test_customers = test_purchases[test_purchases['PurchaseDate'] >= test_start_date]['CustomerID'].unique()
        
        total_precision = 0.0
        total_recall = 0.0
        num_test_customers = 0
        
        for customer_id in test_customers:
            # Create user vector from training data
            user_vector = np.zeros(len(preprocessor.item_encoder.classes_))
            if customer_id in customer_item_matrix.index:
                purchases = customer_item_matrix.loc[customer_id]
                for item_id, quantity in purchases[purchases > 0].items():
                    if item_id in preprocessor.item_encoder.classes_:
                        idx = preprocessor.item_encoder.transform([item_id])[0]
                        user_vector[idx] = quantity
            
            # Scale the user vector
            user_vector_scaled = preprocessor.transform_vector(user_vector)
            
            # Get predictions from the model
            reconstructed_user = model.autoencoder.predict(user_vector_scaled)
            # Transform predictions back to original scale
            reconstructed_user = preprocessor.inverse_transform_predictions(reconstructed_user)
            
            # Get recommendations - highest predicted values
            mask = np.ones(reconstructed_user.shape[1], dtype=bool)
            reconstructed_user[0, ~mask] = -np.inf
            recommended_indices = np.argsort(reconstructed_user[0])[::-1][:self.top_k]
            recommended_item_ids = preprocessor.item_encoder.inverse_transform(recommended_indices)
            
            # Get actual purchases during the test period
            actual_purchases = test_purchases[
                (test_purchases['CustomerID'] == customer_id) & 
                (test_purchases['PurchaseDate'] >= test_start_date)
            ]['ItemID'].tolist()
            
            # Calculate metrics if the user made purchases in the test period
            if actual_purchases:
                precision, recall = self.calculate_precision_recall(recommended_item_ids, actual_purchases)
                total_precision += precision
                total_recall += recall
                num_test_customers += 1
        
        # Return average precision and recall across all test customers
        if num_test_customers > 0:
            return (total_precision / num_test_customers, total_recall / num_test_customers)
        return 0.0, 0.0