import os
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from config_manager import ConfigManager
from recommendation_engine import RecommenderModel

def visualize_model_architecture():
    """Generate and save visualization of the model architecture."""
    # Create model directory if it doesn't exist
    os.makedirs("model_artifacts", exist_ok=True)
    
    # Initialize configuration
    config = ConfigManager("config.yaml")
    
    # Create model instance with sample number of items
    sample_num_items = 100  # Example number of items
    model = RecommenderModel(num_items=sample_num_items, config=config)
    
    # Generate and save full autoencoder architecture
    plot_model(
        model.autoencoder,
        to_file="model_artifacts/autoencoder_architecture.png",
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=300
    )
    
    # Generate and save encoder architecture
    plot_model(
        model.encoder,
        to_file="model_artifacts/encoder_architecture.png",
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=300
    )

if __name__ == "__main__":
    visualize_model_architecture() 