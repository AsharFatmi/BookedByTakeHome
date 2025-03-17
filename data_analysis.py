"""
Data analysis module for processing and analyzing customer purchase data.
Provides functionality for loading, preprocessing, and analyzing sales data.
"""

import pandas as pd
import json

class DataAnalyzer:
    """
    Class for analyzing customer purchase data and generating insights.
    Handles data loading, preprocessing, and analysis of sales patterns.
    """

    def __init__(self, customers_file = 'customers.csv', items_file = 'items.csv', purchases_file = 'purchases.csv'):
        """
        Initialize the DataAnalyzer with input file paths.
        
        Args:
            customers_file (str): Path to customers CSV file
            items_file (str): Path to items CSV file
            purchases_file (str): Path to purchases CSV file
        """
        self.customers_file = customers_file
        self.items_file = items_file
        self.purchases_file = purchases_file
        self.df_customers = None
        self.df_items = None
        self.df_purchases = None
        self.df_merged = None

    def load_data(self):
        """
        Load and preprocess customer purchase data from CSV files.
        
        Performs the following operations:
        1. Loads customer, item, and purchase data
        2. Fills missing employee IDs
        3. Converts purchase dates to datetime
        4. Merges data into a single DataFrame
        
        Returns:
            bool: True if successful, False if any errors occur
        """
        try:
            # Load all data files
            self.df_customers = pd.read_csv(self.customers_file)
            self.df_items = pd.read_csv(self.items_file)
            self.df_purchases = pd.read_csv(self.purchases_file)

            # Clean and preprocess data
            self.df_purchases['EmployeeID'].fillna('Unknown', inplace=True)
            self.df_purchases['PurchaseDate'] = pd.to_datetime(self.df_purchases['PurchaseDate'])
            
            # Merge all dataframes
            self.df_merged = pd.merge(self.df_purchases, self.df_items, on='ItemID', how='left')
            self.df_merged = pd.merge(self.df_merged, self.df_customers, on='CustomerID', how='left')
            
            return True

        except FileNotFoundError as e:
            print(f"Error: File not found - {e}")
            return False
        except pd.errors.EmptyDataError:
            print("Error: One of the CSV files is empty.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False

    def analyze_sales_and_spending(self):
        """
        Analyzes the merged DataFrame to identify top and bottom selling items and categories,
        and calculate the average spending per customer.
        Returns a dictionary containing all analysis results.
        """
        if self.df_merged is None:
            return {"error": "Data not loaded. Please load data first."}

        analysis_results = {
            "top_selling_items": {},
            "bottom_selling_items": {},
            "top_selling_categories": {},
            "top_selling_subcategories": {},
            "bottom_selling_subcategories": {},
            "revenue_analysis": {},
            "customer_spending": {}
        }

        # 1. Top and Bottom Selling Items
        if 'ItemName' in self.df_merged.columns and not self.df_merged['ItemName'].isnull().all():
            item_counts = self.df_merged['ItemName'].value_counts()
            analysis_results["top_selling_items"]["by_quantity"] = item_counts.head(10).to_dict()
            analysis_results["bottom_selling_items"]["by_quantity"] = item_counts.tail(10).to_dict()
        else:
            analysis_results["top_selling_items"]["by_quantity"] = "Cannot be determined (missing ItemName data)"
            analysis_results["bottom_selling_items"]["by_quantity"] = "Cannot be determined (missing ItemName data)"

        # 2. Top-Selling Categories
        if 'ItemCategory' in self.df_merged.columns and not self.df_merged['ItemCategory'].isnull().all():
            analysis_results["top_selling_categories"]["by_quantity"] = self.df_merged['ItemCategory'].value_counts().head(10).to_dict()
        else:
            analysis_results["top_selling_categories"]["by_quantity"] = "Cannot be determined (missing ItemCategory data)"

        # 2.b Top and Bottom Selling Subcategories
        if 'ItemSubcategory' in self.df_merged.columns and not self.df_merged['ItemSubcategory'].isnull().all():
            subcat_counts = self.df_merged['ItemSubcategory'].value_counts()
            analysis_results["top_selling_subcategories"]["by_quantity"] = subcat_counts.head(10).to_dict()
            analysis_results["bottom_selling_subcategories"]["by_quantity"] = subcat_counts.tail(10).to_dict()
        else:
            analysis_results["top_selling_subcategories"]["by_quantity"] = "Cannot be determined (missing ItemSubcategory data)"
            analysis_results["bottom_selling_subcategories"]["by_quantity"] = "Cannot be determined (missing ItemSubcategory data)"

        # 3. Top and Bottom Revenue Generating Items
        if all(col in self.df_merged.columns for col in ['ItemName', 'TotalPrice']) and \
           not (self.df_merged['ItemName'].isnull().all() or self.df_merged['TotalPrice'].isnull().all()):
            revenue_by_item = self.df_merged.groupby('ItemName')['TotalPrice'].sum()
            analysis_results["revenue_analysis"]["top_items"] = revenue_by_item.nlargest(10).to_dict()
            analysis_results["revenue_analysis"]["bottom_items"] = revenue_by_item.nsmallest(10).to_dict()
        else:
            analysis_results["revenue_analysis"]["top_items"] = "Cannot be determined (missing ItemName or TotalPrice data)"
            analysis_results["revenue_analysis"]["bottom_items"] = "Cannot be determined (missing ItemName or TotalPrice data)"

        # 4. Top-Selling Categories by Revenue
        if all(col in self.df_merged.columns for col in ['ItemCategory', 'TotalPrice']) and \
           not (self.df_merged['ItemCategory'].isnull().all() or self.df_merged['TotalPrice'].isnull().all()):
            analysis_results["revenue_analysis"]["top_categories"] = self.df_merged.groupby('ItemCategory')['TotalPrice'].sum().nlargest(10).to_dict()
        else:
            analysis_results["revenue_analysis"]["top_categories"] = "Cannot be determined (missing ItemCategory or TotalPrice data)"

        # 4.b Top-Selling Subcategories by Revenue
        if all(col in self.df_merged.columns for col in ['ItemSubcategory', 'TotalPrice']) and \
           not (self.df_merged['ItemSubcategory'].isnull().all() or self.df_merged['TotalPrice'].isnull().all()):
            analysis_results["revenue_analysis"]["top_subcategories"] = self.df_merged.groupby('ItemSubcategory')['TotalPrice'].sum().nlargest(10).to_dict()
        else:
            analysis_results["revenue_analysis"]["top_subcategories"] = "Cannot be determined (missing ItemSubcategory or TotalPrice data)"

        # 5. Customer Spending Analysis
        if all(col in self.df_merged.columns for col in ['CustomerID', 'TotalPrice']) and \
           not (self.df_merged['CustomerID'].isnull().all() or self.df_merged['TotalPrice'].isnull().all()):
            avg_spending_per_customer = self.df_merged.groupby('CustomerID')['TotalPrice'].mean()
            
            analysis_results["customer_spending"].update({
                "per_customer": avg_spending_per_customer.to_dict(),
                "overall_average": float(avg_spending_per_customer.mean()),
                "distribution": {
                    "count": float(avg_spending_per_customer.count()),
                    "mean": float(avg_spending_per_customer.mean()),
                    "std": float(avg_spending_per_customer.std()),
                    "min": float(avg_spending_per_customer.min()),
                    "25%": float(avg_spending_per_customer.quantile(0.25)),
                    "50%": float(avg_spending_per_customer.quantile(0.50)),
                    "75%": float(avg_spending_per_customer.quantile(0.75)),
                    "max": float(avg_spending_per_customer.max())
                }
            })
        else:
            analysis_results["customer_spending"] = "Cannot be determined (missing CustomerID or TotalPrice data)"

        # 7. Average spending per customer segment
        if all(col in self.df_merged.columns for col in ['CustomerSegment', 'TotalPrice']) and \
           not (self.df_merged['CustomerSegment'].isnull().all() or self.df_merged['TotalPrice'].isnull().all()):
            analysis_results["customer_spending"]["by_segment"] = self.df_merged.groupby('CustomerSegment')['TotalPrice'].mean().to_dict()
        else:
            analysis_results["customer_spending"]["by_segment"] = "Cannot be determined (missing CustomerSegment or TotalPrice data)"

        return analysis_results

    def save_merged_data(self, output_file='merged.csv'):
        """
        Saves the merged DataFrame to a CSV file.
        """
        if self.df_merged is not None:
            self.df_merged.to_csv(output_file, index=False)
            print(f"Merged data saved to {output_file}")
        else:
            print("Error: No merged data to save. Please load data first.")


if __name__ == "__main__":
    # Initialize the analyzer and run analysis
    analyzer = DataAnalyzer()

    # Load and analyze the data
    if analyzer.load_data():
        # Get and print analysis results
        results = analyzer.analyze_sales_and_spending()

        # Save analysis results to JSON file for caching
        with open('analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Analysis results saved to analysis_results.json")