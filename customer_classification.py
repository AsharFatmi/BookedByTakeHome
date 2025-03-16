import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from data_analysis import DataAnalyzer
from datetime import datetime

class CustomerClassifier:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = None
        self.n_clusters = None  # Will be set after elbow method
        self.cluster_labels = None  # Will be set after determining clusters

    def _extract_features(self, df_merged):
        """Extract features for clustering from merged DataFrame."""
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
        plt.show()
        
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
        """Main method to perform customer classification."""
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

        # Add cluster labels
        df_clustering['Cluster'] = clusters
        df_clustering['ClusterLabel'] = df_clustering['Cluster'].map(self.cluster_labels)

        # Generate cluster summary
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

    def save_cluster(self, df_merged, output_file="cluster_samples.csv"):
        """
        Save clustered data with only the features used in the analysis.
        
        Args:
            df_merged: DataFrame with cluster labels
            output_file: Output CSV filename
        """
        if 'Cluster' not in df_merged.columns:
            print("Error: DataFrame does not contain cluster information")
            return False
        
        # Extract features again to get the same structure
        features = self._extract_features(df_merged)
        df_clustering = self._combine_features(features)
        
        # Add cluster information
        df_clustering = pd.merge(
            df_clustering,
            df_merged[['CustomerID', 'Cluster', 'ClusterLabel']],
            on='CustomerID',
            how='left'
        )
        
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

        # Save cluster means to CSV
        cluster_means.to_csv('cluster_means.csv', index=False)
        
        # Save to CSV
        df_clustering.to_csv(output_file, index=False)
        print(f"\nSaved clustering features and results ({len(df_clustering)} rows) to {output_file}")
        
        # Print summary of records per cluster
        cluster_counts = df_clustering['Cluster'].value_counts().sort_index()
        print("\nRecords per cluster:")
        for cluster, count in cluster_counts.items():
            print(f"Cluster {cluster}: {count} records")
        
        # Print the columns saved
        print("\nFeatures saved:")
        for col in df_clustering.columns:
            print(f"- {col}")
        
        return cluster_means


if __name__ == "__main__":
    # Initialize data analyzer and load data
    analyzer = DataAnalyzer('customers.csv', 'items.csv', 'purchases.csv')

    if analyzer.load_data():
    
        # Initialize classifier with n_clusters=0 to trigger elbow method
        # or with a specific number to run direct clustering
    
        n_clusters = 5  # Change this value to skip elbow method; Set to 5 as previous runs show 5 clusters are the most optimal
        classifier = CustomerClassifier()
        
        if n_clusters == 0:
            # Run elbow method and get user input
            classifier.run_elbow_method(analyzer.df_merged)
            n_clusters = int(input("Enter the desired number of clusters based on the elbow curve: "))
            
        # Set clusters and run classification
        classifier.set_n_clusters(n_clusters)
        df_merged = classifier.fit_predict(analyzer.df_merged)
        
        if df_merged is not None:
            print("\nFirst few rows with Cluster Labels:\n", df_merged.head())
            cluster_means = classifier.save_cluster(df_merged, "cluster_samples.csv")