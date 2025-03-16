import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set a seed for reproducibility
random.seed(42)
np.random.seed(42)

# --- 1. Generate Customer IDs ---
num_customers = 500
customer_ids = [f"Cust{i:04d}" for i in range(1, num_customers + 1)]

# --- 2. Generate Service and Product IDs ---
#  Salon services and products
services = [
    "Haircut (Women's)", "Haircut (Men's)", "Haircut (Children's)",
    "Hair Coloring (Full)", "Hair Coloring (Roots)", "Highlights", "Balayage",
    "Hair Treatment (Deep Conditioning)", "Hair Treatment (Keratin)",
    "Hair Styling (Updo)", "Hair Styling (Blowout)",
    "Perm", "Relaxer",
    "Hair Extensions (Installation)", "Hair Extensions (Removal)",
    "Makeup Application", "Facial", "Waxing (Eyebrows)", "Waxing (Legs)", "Waxing (Bikini)"
]
products = [
    "Shampoo (Volume)", "Shampoo (Moisturizing)", "Shampoo (Color-Safe)", "Shampoo (Clarifying)", "Shampoo (Dry)",
    "Conditioner (Volume)", "Conditioner (Moisturizing)", "Conditioner (Color-Safe)", "Conditioner (Leave-In)", "Conditioner (Deep)",
    "Hair Mask", "Hair Serum", "Hair Spray", "Styling Gel", "Styling Mousse", "Styling Cream", "Styling Wax",
    "Dry Shampoo", "Heat Protectant Spray", "Texturizing Spray", "Curl Enhancer", "Frizz Control Serum",
    "Makeup Remover", "Face Wash", "Moisturizer", "Sunscreen", "Toner",
    "Exfoliating Scrub", "Face Mask (Clay)", "Face Mask (Sheet)", "Eye Cream", "Lip Balm",
    "Body Wash", "Body Lotion", "Body Scrub", "Hand Cream", "Cuticle Oil",
    "Nail Polish (Red)", "Nail Polish (Nude)", "Nail Polish (Glitter)", "Top Coat", "Base Coat", "Nail Polish Remover",
    "Foundation", "Concealer", "Mascara", "Eyeliner", "Eyeshadow", "Lipstick"

]

all_items = services + products
num_items = len(all_items)
item_ids = [f"Item{i:04d}" for i in range(1, num_items + 1)]
item_id_map = dict(zip(all_items, item_ids))  # Map item names to IDs

# --- 3. Generate Purchase Records ---
num_records = 5000
purchase_records = []

for _ in range(num_records):
    customer_id = random.choice(customer_ids)
    # Simulate appointments (more likely than product-only purchases)
    if random.random() < 0.7:  # 70% chance of an appointment
        item_name = random.choice(services)
        item_id = item_id_map[item_name]
        quantity = 1  # Usually 1 for services
    else:
        item_name = random.choice(products)
        item_id = item_id_map[item_name]
        quantity = random.randint(1, 3)  # Can buy multiple products

    # Simulate purchase date (within the last year)
    date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 364))
    purchase_records.append([customer_id, item_id, quantity, date, item_name])


# --- 4. Create DataFrames (Initially combined for feature engineering) ---
df = pd.DataFrame(purchase_records, columns=['CustomerID', 'ItemID', 'Quantity', 'PurchaseDate', 'ItemName'])

# --- 5. Add Salon-Specific Features (as before, but preparing for separation) ---

# a) Item Categories (Service vs. Product)
df['ItemCategory'] = df['ItemName'].apply(lambda x: 'Service' if x in services else 'Product')

# b) Item Subcategories
def get_subcategory(item_name):
    if "Haircut" in item_name:
        return "Haircut"
    elif "Coloring" in item_name or "Highlights" in item_name or "Balayage" in item_name:
        return "Coloring"
    elif "Treatment" in item_name:
        return "Treatment"
    elif "Styling" in item_name or "Perm" in item_name or "Relaxer" in item_name or "Extensions" in item_name:
        return "Styling"
    elif "Waxing" in item_name:
        return "Waxing"
    elif "Makeup" in item_name:
        return "Makeup"
    elif "Facial" in item_name:
        return "Facial"
    elif "Shampoo" in item_name:
        return "Shampoo"
    elif "Conditioner" in item_name:
        return "Conditioner"
    elif "Mask" in item_name or "Serum" in item_name:
        return "Treatment Products"
    elif "Styling" in item_name or "Spray" in item_name or 'Dry Shampoo' in item_name or 'Heat Protectant' in item_name or 'Texturizing' in item_name:
        return "Styling Products"
    elif "Makeup" in item_name or "Face Wash" in item_name or 'Moisturizer' in item_name or "Sunscreen" in item_name or 'Toner' in item_name:
          return "Skincare"
    return "Other"
df['ItemSubcategory'] = df['ItemName'].apply(get_subcategory)

# c) Item Prices
def get_price(item_category, item_subcategory):
    if item_category == 'Service':
        if item_subcategory == "Haircut":
            return round(random.uniform(20, 80), 2)
        elif item_subcategory == "Coloring":
            return round(random.uniform(60, 200), 2)
        elif item_subcategory == "Treatment":
            return round(random.uniform(30, 100), 2)
        elif item_subcategory == "Styling":
            return round(random.uniform(40, 150), 2)
        elif item_subcategory == "Waxing":
            return round(random.uniform(10, 60), 2)
        elif item_subcategory == "Makeup":
            return round(random.uniform(50, 120), 2)
        elif item_subcategory == "Facial":
            return round(random.uniform(70, 150), 2)

    elif item_category == 'Product':
        if item_subcategory == "Shampoo" or item_subcategory == "Conditioner":
            return round(random.uniform(10, 40), 2)
        elif item_subcategory == "Treatment Products":
            return round(random.uniform(20, 60), 2)
        elif item_subcategory == "Styling Products":
            return round(random.uniform(15, 45), 2)
        elif item_subcategory == 'Skincare':
            return round(random.uniform(15, 70),2)

    return round(random.uniform(10, 50), 2)  # Default

df['UnitPrice'] = df.apply(lambda row: get_price(row['ItemCategory'], row['ItemSubcategory']), axis=1)
df['TotalPrice'] = df['UnitPrice'] * df['Quantity']

# d) Customer Preferences
customer_pref_service_map = {cust_id: random.choice(services) for cust_id in customer_ids}
customer_pref_product_map = {cust_id: random.choice(products) for cust_id in customer_ids}
df['PreferredCategory'] = df.apply(lambda row: customer_pref_service_map.get(row['CustomerID'], '') if row['ItemCategory'] == 'Service' else customer_pref_product_map.get(row['CustomerID'], ''), axis=1)


# e) Customer Segments
customer_segments = ['Regular', 'Occasional', 'New']
customer_segment_map = {cust_id: random.choices(customer_segments, weights=[0.4, 0.4, 0.2])[0] for cust_id in customer_ids} # 40% Regular, 40% Occasional, 20% New
df['CustomerSegment'] = df['CustomerID'].map(customer_segment_map)

# f) Employee ID (for services)
num_employees = 10
employee_ids = [f"Emp{i:02d}" for i in range(1, num_employees + 1)]
df['EmployeeID'] = df.apply(lambda row: random.choice(employee_ids) if row['ItemCategory'] == 'Service' else None, axis=1)


# --- 6. Create Separate DataFrames ---

# 6a. Customers
df_customers = pd.DataFrame({
    'CustomerID': customer_ids,
    'CustomerSegment': [customer_segment_map[cust_id] for cust_id in customer_ids],
    'PreferredCategory': [customer_pref_service_map[cust_id] if customer_pref_service_map[cust_id] else customer_pref_product_map[cust_id]  for cust_id in customer_ids ] #pref service OR product

})

# 6b. Products/Items
df_items = pd.DataFrame({
    'ItemID': item_ids,
    'ItemName': all_items,
    'ItemCategory': ['Service' if item in services else 'Product' for item in all_items],
    'ItemSubcategory': [get_subcategory(item) for item in all_items],
    'UnitPrice': [get_price('Service', get_subcategory(item)) if item in services else get_price('Product', get_subcategory(item)) for item in all_items]
})

# 6c. Purchase Records
df_purchases = df[['CustomerID', 'ItemID', 'Quantity', 'PurchaseDate', 'TotalPrice', 'EmployeeID']].copy()
df_purchases.rename(columns={'TotalPrice': 'TotalPrice'}, inplace=True)


# --- 7. Save to CSV Files ---
df_customers.to_csv('customers.csv', index=False)
df_items.to_csv('items.csv', index=False)
df_purchases.to_csv('purchases.csv', index=False)

print("Customers Data:")
print(df_customers.head())
print("\nItems Data:")
print(df_items.head())
print("\nPurchases Data:")
print(df_purchases.head())