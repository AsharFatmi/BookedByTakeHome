"""
Customer classification module for segmenting customers based on their behavior.
Uses K-means clustering to identify distinct customer segments based on various metrics.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from data_analysis import DataAnalyzer
from datetime import datetime

class CustomerClassifier:
    """
    Class for performing customer segmentation using K-means clustering.
    Analyzes customer behavior patterns and groups them into meaningful segments.
    """

    def __init__(self, random_state=42):
        """
        Initialize the CustomerClassifier.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = None
        self.n_clusters = None  # Will be set after elbow method
        self.cluster_labels = {
            0: "Loyal High-Spenders",
            1: "Service-Focused Occasional Shoppers",
            2: "Price-Sensitive or Discount Shoppers",
            3: "Active and Engaged Value Shoppers",
            4: "Balanced Regulars"
        }

    def _extract_features(self, df_merged):
        """
        Extract features for clustering from merged DataFrame.
        
        Calculates various customer metrics including:
        - Recency, Frequency, Monetary Value (RFM)
        - Average Order Value (AOV)
        - Time between purchases
        - Purchase streaks
        - Service vs Product ratio
        
        Args:
            df_merged: Merged DataFrame containing customer purchase data
            
        Returns:
            Dictionary of feature DataFrames
        """
        features = {}
        reference_date = datetime(2023, 12, 31)
        
        # --- RFM Metrics ---
        # Recency, Frequency, Monetary
        rfm_metrics = df_merged.groupby('CustomerID').agg({
            'PurchaseDate': lambda x: (reference_date - pd.to_datetime(x.max())).days,  # Recency
            'ItemID': 'count',  # Frequency
            'TotalPrice': 'sum'  # Monetary Value
        }).reset_index()
        rfm_metrics.columns = ['CustomerID', 'Recency', 'Frequency', 'MonetaryValue']
        features['rfm'] = rfm_metrics

        # --- Customer Demographics ---
        # Get unique customer demographics (age and gender)
        demographics = df_merged[['CustomerID', 'Age', 'Gender']].drop_duplicates()
        # One-hot encode gender
        demographics = pd.get_dummies(demographics, columns=['Gender'], prefix=['Gender'])
        features['demographics'] = demographics

        # --- Average Order Value (AOV) ---
        features['aov'] = pd.DataFrame({
            'CustomerID': rfm_metrics['CustomerID'],
            'AOV': rfm_metrics['MonetaryValue'] / rfm_metrics['Frequency']
        })

        # --- Time Between Purchases ---
        # Sort purchases by customer and date
        df_sorted = df_merged.sort_values(['CustomerID', 'PurchaseDate'])
        df_sorted['TimeDiff'] = df_sorted.groupby('CustomerID')['PurchaseDate'].diff().dt.days
        avg_time_between = df_sorted.groupby('CustomerID')['TimeDiff'].mean().reset_index()
        avg_time_between.columns = ['CustomerID', 'AvgTimeBetweenPurchases']
        features['time_between'] = avg_time_between

        # --- Purchase Streak ---
        df_sorted['PurchaseDay'] = (pd.to_datetime(df_sorted['PurchaseDate']) - datetime(2023, 1, 1)).dt.days
        
        def longest_streak(days):
            """Calculate the longest consecutive purchase streak."""
            days = sorted(set(days))
            if not days:
                return 0
            streaks = []
            current_streak = 1
            for i in range(1, len(days)):
                if days[i] == days[i-1] + 1:
                    current_streak += 1
                else:
                    streaks.append(current_streak)
                    current_streak = 1
            streaks.append(current_streak)
            return max(streaks)
        
        streak_data = df_sorted.groupby('CustomerID')['PurchaseDay'].apply(longest_streak).reset_index()
        streak_data.columns = ['CustomerID', 'LongestPurchaseStreak']
        features['streak'] = streak_data

        # --- Service vs Product Ratio ---
        category_counts = df_merged.groupby(['CustomerID', 'ItemCategory']).size().unstack(fill_value=0)
        category_counts['TotalPurchases'] = category_counts.sum(axis=1)
        category_ratios = pd.DataFrame({
            'CustomerID': category_counts.index,
            'ServiceRatio': category_counts['Service'] / category_counts['TotalPurchases'],
            'ProductRatio': category_counts['Product'] / category_counts['TotalPurchases']
        }).reset_index(drop=True)
        features['category_ratio'] = category_ratios

        return features

    def _combine_features(self, features):
        """Combine all extracted features into a single DataFrame."""
        df_clustering = features['rfm']
        
        # Merge all other features
        for key in ['demographics', 'aov', 'time_between', 'streak', 'category_ratio']:
            df_clustering = pd.merge(df_clustering, features[key], on='CustomerID', how='left')
        
        # Fill NaN values
        for column in df_clustering.columns:
            if df_clustering[column].dtype in ['int64', 'float64']:
                if column in ['ServiceRatio', 'ProductRatio']:
                    df_clustering[column] = df_clustering[column].fillna(0)
                elif column == 'AvgTimeBetweenPurchases':
                    df_clustering[column] = df_clustering[column].fillna(df_clustering[column].max())
                elif column.startswith('Gender_'):
                    df_clustering[column] = df_clustering[column].fillna(0)
                else:
                    df_clustering[column] = df_clustering[column].fillna(0)
        
        return df_clustering

    def _preprocess_features(self, df_clustering):
        """Scale the features for clustering."""
        features = df_clustering.drop('CustomerID', axis=1)
        scaled_features = self.scaler.fit_transform(features)
        return pd.DataFrame(scaled_features, columns=features.columns)

    def plot_elbow_curve(self, df_scaled, max_clusters=20):
        """
        Plot the elbow curve to help determine optimal number of clusters.
        Returns the inertia values for different k.
        """
        inertias = []
        K = range(1, max_clusters + 1)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, max_iter=500, random_state=self.random_state, n_init=10)
            kmeans.fit(df_scaled)
            inertias.append(round(kmeans.inertia_, 0))
        
        # Create elbow curve plot
        plt.figure(figsize=(10, 6))
        plt.plot(K, inertias, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.grid(True)
        plt.savefig('elbow_curve.png')
        # plt.show()
        
        return inertias

    def set_n_clusters(self, n):
        """Set number of clusters and initialize cluster labels."""
        self.n_clusters = n
        self.cluster_labels = {
            i: f'Cluster {i}' for i in range(n)
        }
        # You can customize labels after analyzing the clusters

    def run_elbow_method(self, df_merged, max_clusters=20):
        """
        Run elbow method analysis to help determine optimal number of clusters.
        
        Args:
            df_merged: DataFrame with merged data
            max_clusters: Maximum number of clusters to test
        """
        # Extract and combine features
        features = self._extract_features(df_merged)
        df_clustering = self._combine_features(features)
        df_scaled = self._preprocess_features(df_clustering)
        
        # Run elbow analysis
        print("Running elbow method analysis...")
        self.plot_elbow_curve(df_scaled, max_clusters)
        return df_scaled

    def fit_predict(self, df_merged):
        """
        Main method to perform customer classification.
        
        Steps:
        1. Extract and combine features
        2. Preprocess features
        3. Run K-means clustering
        4. Generate cluster labels and summary
        
        Args:
            df_merged: Merged DataFrame containing all customer data
            
        Returns:
            DataFrame with cluster assignments
        """
        if df_merged is None:
            print("Error: df_merged is None. Data loading likely failed.")
            return None

        # Extract and combine features
        features = self._extract_features(df_merged)
        df_clustering = self._combine_features(features)
        df_scaled = self._preprocess_features(df_clustering)

        # Check if number of clusters is set
        if self.n_clusters is None:
            print("Number of clusters not set. Running elbow method analysis first...")
            self.run_elbow_method(df_merged)
            raise ValueError("Please set the number of clusters using set_n_clusters() after analyzing the elbow curve.")

        # Fit KMeans and predict clusters
        print(f"Running K-means clustering with {self.n_clusters} clusters...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, 
                           random_state=self.random_state, 
                           n_init=10)
        clusters = self.kmeans.fit_predict(df_scaled)

        # Add cluster labels and generate summary
        df_clustering['Cluster'] = clusters
        df_clustering['ClusterLabel'] = df_clustering['Cluster'].map(self.cluster_labels)
        self._generate_cluster_summary(df_clustering)

        # Merge results back to original DataFrame
        result = pd.merge(
            df_merged, 
            df_clustering[['CustomerID', 'Cluster', 'ClusterLabel']], 
            on='CustomerID', 
            how='left'
        )

        return result

    def _generate_cluster_summary(self, df_clustering):
        """Generate and print cluster summary statistics."""
        numeric_columns = ['Age', 'Recency', 'Frequency', 'MonetaryValue', 'AOV', 
                          'AvgTimeBetweenPurchases', 'LongestPurchaseStreak',
                          'ServiceRatio', 'ProductRatio']
        # Add gender columns if they exist
        gender_columns = [col for col in df_clustering.columns if col.startswith('Gender_')]
        numeric_columns.extend(gender_columns)
        
        cluster_summary = df_clustering[numeric_columns + ['Cluster']].groupby('Cluster').mean()
        print("\nCluster Summary (Mean Values):\n", cluster_summary)

    def visualize_clusters(self, df_clustering):
        """
        Create scatter plots to visualize clusters using different feature combinations.
        
        Args:
            df_clustering: DataFrame containing cluster assignments and features
        """
        import seaborn as sns
        
        # Set the style for all plots
        plt.style.use('default')  # Use default style instead of seaborn
        
        # Set color palette for better visibility
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_clustering['Cluster'].unique())))
        
        # Define feature pairs to plot
        feature_pairs = [
            ('Recency', 'Frequency'),
            ('Frequency', 'MonetaryValue'),
            ('MonetaryValue', 'AOV'),
            ('ServiceRatio', 'ProductRatio'),
            ('Age', 'MonetaryValue'),
            ('AvgTimeBetweenPurchases', 'Frequency')
        ]
        
        # Create a figure with subplots
        n_pairs = len(feature_pairs)
        n_cols = 2
        n_rows = (n_pairs + 1) // 2
        fig = plt.figure(figsize=(15, 5 * n_rows))
        
        # Get features used for clustering in the correct order
        clustering_features = ['Recency', 'Frequency', 'MonetaryValue', 'Age', 'AOV', 
                             'AvgTimeBetweenPurchases', 'LongestPurchaseStreak',
                             'ServiceRatio', 'ProductRatio', 'Gender_F', 'Gender_M']
        
        # Scale the features for cluster centers
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_clustering[clustering_features])
        
        for idx, (feature1, feature2) in enumerate(feature_pairs, 1):
            ax = plt.subplot(n_rows, n_cols, idx)
            
            # Create scatter plot
            scatter = plt.scatter(
                df_clustering[feature1],
                df_clustering[feature2],
                c=df_clustering['Cluster'],
                cmap='viridis',
                alpha=0.6,
                s=100
            )
            
            plt.xlabel(feature1, fontsize=10)
            plt.ylabel(feature2, fontsize=10)
            plt.title(f'Cluster Distribution: {feature1} vs {feature2}', fontsize=12, pad=10)
            
            # Add grid for better readability
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend
            legend1 = plt.legend(*scatter.legend_elements(),
                               title="Clusters",
                               loc="upper right",
                               title_fontsize=10,
                               fontsize=8)
            ax.add_artist(legend1)
            
            # Add cluster centers if kmeans is fitted
            if self.kmeans is not None:
                feature1_idx = clustering_features.index(feature1)
                feature2_idx = clustering_features.index(feature2)
                
                # Get cluster centers and inverse transform them
                cluster_centers_scaled = self.kmeans.cluster_centers_
                cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
                
                plt.scatter(
                    cluster_centers[:, feature1_idx],
                    cluster_centers[:, feature2_idx],
                    c='red',
                    marker='x',
                    s=200,
                    linewidths=3,
                    label='Cluster Centers'
                )
                plt.legend(loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('cluster_visualizations.png', dpi=300, bbox_inches='tight')
        # plt.show()
        
        # ==================================================================================
        # 3D VISUALIZATION: MonetaryValue vs AOV vs AvgTimeBetweenPurchases
        # ==================================================================================
        # This visualization shows the relationship between:
        # - Monetary Value (total amount spent by customer)
        # - AOV (Average Order Value)
        # - Average Time Between Purchases (customer purchasing frequency pattern)
        # The purpose is to identify distinct customer behavior patterns in these dimensions
        # ==================================================================================
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # DRAWING ORDER STRATEGY:
        # 1. First create placeholder for legend (empty scatter plot)
        # 2. Then draw actual data points (they will be behind)
        # 3. Finally draw cluster centers on top (with high zorder) so they're always visible
        
        # Step 1: Add cluster centers to legend first (but don't actually plot them yet)
        if self.kmeans is not None:
            # Calculate the actual positions of cluster centers in the original feature space
            cluster_centers = scaler.inverse_transform(self.kmeans.cluster_centers_)
            
            # Get indices for the features we want to visualize
            mon_idx = clustering_features.index('MonetaryValue')
            aov_idx = clustering_features.index('AOV')
            time_idx = clustering_features.index('AvgTimeBetweenPurchases')
            
            # Store the cluster centers for later actual plotting
            center_x = cluster_centers[:, mon_idx]  # Monetary Value dimension
            center_y = cluster_centers[:, aov_idx]  # AOV dimension
            center_z = cluster_centers[:, time_idx] # Avg Time Between Purchases dimension
            
            # Create an empty scatter plot just for the legend entry
            # This ensures the legend shows the cluster centers marker style
            ax.scatter(
                [], [], [],  # Empty data for legend only
                c='red',
                marker='X',  # X marker for centers (more visible than lowercase x)
                s=500,       # Large size to stand out
                linewidths=6,
                label='Cluster Centers'
            )
        
        # Step 2: Add the actual data points (drawn first, so they appear behind centers)
        scatter = ax.scatter(
            df_clustering['MonetaryValue'],
            df_clustering['AOV'],
            df_clustering['AvgTimeBetweenPurchases'],
            c=df_clustering['Cluster'],  # Color points by their cluster assignment
            cmap='viridis',             # Color map for distinguishing clusters
            alpha=0.6,                  # Transparency helps when points overlap
            s=100                       # Point size
        )
        
        # Step 3: Now add the actual cluster centers AFTER the data points so they're on top
        if self.kmeans is not None:
            # Plot the cluster centers with high visibility
            centers = ax.scatter(
                center_x, center_y, center_z,
                c='red',               # Bright color to stand out
                marker='X',            # X marker is more visible in 3D
                s=500,                 # Very large size to ensure visibility
                linewidths=6,          # Thick lines
                edgecolors='black',    # Black edge creates contrast with red fill
                zorder=100             # Very high zorder ensures drawing on top
            )
            
            # Add text labels for each cluster center for even better visibility
            for i in range(len(center_x)):
                ax.text(
                    center_x[i], center_y[i], center_z[i],  # Position at center
                    f'C{i}',                                # Label text (Cluster #)
                    fontsize=12,
                    fontweight='bold',
                    color='white',                          # White text on black background
                    backgroundcolor='black',                # for maximum readability
                    zorder=101                              # Even higher z-order than markers
                )
        
        # Set axis labels and title
        ax.set_xlabel('Monetary Value', fontsize=10, labelpad=10)
        ax.set_ylabel('AOV', fontsize=10, labelpad=10)
        ax.set_zlabel('Avg Time Between Purchases', fontsize=10, labelpad=10)
        ax.set_title('3D Cluster Visualization: Monetary vs AOV vs Time Between', fontsize=12, pad=20)
        
        # Add legends - we need to handle two separate legends
        # 1. For the data points colored by cluster
        # 2. For the cluster centers
        legend1 = ax.legend(*scatter.legend_elements(),
                           title="Clusters",
                           loc="upper right",
                           title_fontsize=10,
                           fontsize=8)
        ax.add_artist(legend1)  # This adds the first legend but keeps the current legend active
        ax.legend(loc='upper left', fontsize=8)  # Add second legend for cluster centers
        
        # Adjust the view angle for optimal viewing of clusters
        # Elevation (up/down angle) and azimuth (horizontal rotation angle)
        ax.view_init(elev=25, azim=30)  # These values chosen for best visibility of clusters
        
        # Save high resolution image
        plt.savefig('cluster_visualizations_3d.png', dpi=300, bbox_inches='tight')
        # plt.show()

    def save_cluster(self, df_merged, output_file="cluster_samples.csv"):
        """
        Save clustered data with features used in the analysis.
        
        Args:
            df_merged: DataFrame with cluster labels
            output_file: Output CSV filename
            
        Returns:
            DataFrame with cluster means
        """
        if 'Cluster' not in df_merged.columns:
            print("Error: DataFrame does not contain cluster information")
            return False
        
        # Extract features and combine with cluster information
        features = self._extract_features(df_merged)
        df_clustering = self._combine_features(features)
        df_clustering = pd.merge(
            df_clustering,
            df_merged[['CustomerID', 'Cluster', 'ClusterLabel']],
            on='CustomerID',
            how='left'
        )
        
        # Calculate cluster means for key metrics
        cluster_means = df_clustering.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'MonetaryValue': 'mean',
            'AOV': 'mean',
            'AvgTimeBetweenPurchases': 'mean',
            'LongestPurchaseStreak': 'mean',
            'ServiceRatio': 'mean',
            'ProductRatio': 'mean',
            'Age': 'mean'
        }).round(2).reset_index()

        # Save results
        cluster_means.to_csv('cluster_means.csv', index=False)
        df_clustering.to_csv(output_file, index=False)
        
        # Print summary statistics
        print(f"\nSaved clustering features and results ({len(df_clustering)} rows) to {output_file}")
        cluster_counts = df_clustering['Cluster'].value_counts().sort_index()
        print("\nRecords per cluster:")
        for cluster, count in cluster_counts.items():
            print(f"Cluster {cluster}: {count} records")
        
        print("\nFeatures saved:")
        for col in df_clustering.columns:
            print(f"- {col}")
        
        return cluster_means


if __name__ == "__main__":
    # Initialize data analyzer and load data
    analyzer = DataAnalyzer('customers.csv', 'items.csv', 'purchases.csv')

    if analyzer.load_data():
        # Initialize classifier
        classifier = CustomerClassifier()
        
        # Configuration
        run_elbow_method = False  # Set to True to run elbow method analysis
        max_clusters = 20  # Maximum number of clusters to test
        default_clusters = 5  # Default number of clusters if not running elbow method
        
        if run_elbow_method:
            # Run elbow method and get user input
            df_scaled = classifier.run_elbow_method(analyzer.df_merged, max_clusters)
            n_clusters = int(input("Enter the desired number of clusters based on the elbow curve: "))
        else:
            n_clusters = default_clusters
            
        # Set clusters and run classification
        classifier.set_n_clusters(n_clusters)
        df_merged = classifier.fit_predict(analyzer.df_merged)
        
        if df_merged is not None:
            print("\nFirst few rows with Cluster Labels:\n", df_merged.head())
            cluster_means = classifier.save_cluster(df_merged, "cluster_samples.csv")
            
            # Get the clustering data from the saved file which contains all features
            df_clustering = pd.read_csv("cluster_samples.csv")
            
            # Visualize the clusters
            classifier.visualize_clusters(df_clustering)