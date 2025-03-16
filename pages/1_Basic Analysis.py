import streamlit as st
from data_analysis import DataAnalyzer

st.set_page_config(
    page_title="Data Analysis - Basic",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide the Streamlit header and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def line_break():
    st.markdown("</br>", unsafe_allow_html=True)


st.title("Statistical Analysis and insights from the data")

# Initialize and use the DataAnalyzer directly
analyzer = DataAnalyzer()

# Load data and get analysis results
if analyzer.load_data():
    analysis_results = analyzer.analyze_sales_and_spending()
    
    line_break()
    line_break()
    st.header("Key Analysis Findings")
    line_break()
    line_break()

    # Top Selling Items
    st.subheader("Top 5 Selling Items by Quantity")
    top_items_col1, top_items_col2 = st.columns(2)
    with top_items_col1:
        top_items = analysis_results['top_selling_items']['by_quantity']
        items_markdown = ""
        for i, (item, quantity) in enumerate(list(top_items.items())[:5], 1):
            items_markdown += f"{i}. {item} ({quantity} units)\n"
        st.markdown(items_markdown)
    with top_items_col2:
        st.info("The salon's most popular services are hair coloring and basic haircuts, indicating strong demand for essential hair care services.")

    line_break()
    line_break()
    # Top Categories
    st.subheader("Sales by Category")
    cat_col1, cat_col2 = st.columns(2)
    with cat_col1:
        categories = analysis_results['top_selling_categories']['by_quantity']
        st.markdown("\n".join([f"- **{cat}**: {qty:,} units" for cat, qty in categories.items()]))
    with cat_col2:
        service_qty = categories['Service']
        total_qty = sum(categories.values())
        percentage = (service_qty / total_qty) * 100
        st.info(f"Services make up approximately {percentage:.0f}% of total sales volume, showing the business is primarily service-driven.")

    line_break()
    line_break()
    # Top Subcategories
    st.subheader("Top 5 Subcategories by Volume")
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        subcategories = analysis_results['top_selling_subcategories']['by_quantity']
        subcat_markdown = ""
        for i, (subcat, quantity) in enumerate(list(subcategories.items())[:5], 1):
            subcat_markdown += f"{i}. {subcat} ({quantity:,} units)\n"
        st.markdown(subcat_markdown)
    with sub_col2:
        st.info("Hair styling and coloring services dominate the subcategories, representing the core business offerings.")

    line_break()
    line_break()
    # Customer Spending Analysis
    st.subheader("Customer Spending Analysis")
    spend_col1, spend_col2 = st.columns(2)
    with spend_col1:
        dist = analysis_results['customer_spending']['distribution']
        st.markdown(f"""
        - **Average Spending**: ${dist['mean']:.2f} per customer
        - **Minimum Spending**: ${dist['min']:.2f}
        - **Maximum Spending**: ${dist['max']:.2f}
        - **Median Spending**: ${dist['50%']:.2f}
        """)
    with spend_col2:
        st.info(f"There's a healthy spread in customer spending, with most customers falling between ${dist['25%']:.2f} (25th percentile) and ${dist['75%']:.2f} (75th percentile), indicating a good mix of service levels.")

    line_break()
    line_break()
    # Revenue Leaders
    st.subheader("Top 5 Revenue Generating Items")
    rev_col1, rev_col2 = st.columns(2)
    with rev_col1:
        revenue_items = analysis_results['revenue_analysis']['top_items']
        rev_markdown = ""
        for i, (item, revenue) in enumerate(list(revenue_items.items())[:5], 1):
            rev_markdown += f"{i}. {item}: ${revenue:,.2f}\n"
        st.markdown(rev_markdown)
    with rev_col2:
        st.info("Hair coloring services are the highest revenue generators, with various coloring services dominating the top revenue list.")

    line_break()
    line_break()
    # Bottom Selling Items
    st.subheader("Bottom 5 Selling Items")
    bottom_col1, bottom_col2 = st.columns(2)
    with bottom_col1:
        bottom_items = analysis_results['bottom_selling_items']['by_quantity']
        bottom_markdown = ""
        for i, (item, quantity) in enumerate(list(bottom_items.items())[:5], 1):
            bottom_markdown += f"{i}. {item} ({quantity} units)\n"
        st.markdown(bottom_markdown)
    with bottom_col2:
        st.info("Retail products, particularly makeup and styling products, show lower sales volumes compared to services.")

    line_break()
    line_break()
    # Bottom Revenue Items
    st.subheader("Lowest Revenue Generating Items")
    bottom_rev_col1, bottom_rev_col2 = st.columns(2)
    with bottom_rev_col1:
        bottom_revenue = analysis_results['revenue_analysis']['bottom_items']
        bottom_rev_markdown = ""
        for i, (item, revenue) in enumerate(list(bottom_revenue.items())[:5], 1):
            bottom_rev_markdown += f"{i}. {item}: ${revenue:,.2f}\n"
        st.markdown(bottom_rev_markdown)
    with bottom_rev_col2:
        st.info("Basic retail items generate the lowest revenue, suggesting an opportunity to either optimize the retail product mix or focus more on high-margin services.")

    # Revenue by Category
    st.subheader("Revenue by Category")
    cat_rev_col1, cat_rev_col2 = st.columns(2)
    with cat_rev_col1:
        category_revenue = analysis_results['revenue_analysis']['top_categories']
        total_revenue = sum(category_revenue.values())
        for category, revenue in category_revenue.items():
            percentage = (revenue / total_revenue) * 100
            st.markdown(f"- **{category}**: ${revenue:,.2f} ({percentage:.1f}%)")
    with cat_rev_col2:
        service_rev = category_revenue['Service']
        service_percentage = (service_rev / total_revenue) * 100
        st.info(f"Services generate {service_percentage:.1f}% of total revenue, highlighting the salon's strength in service-based offerings.")

    line_break()
    line_break()
    # Revenue by Subcategory
    st.subheader("Top 5 Revenue Generating Subcategories")
    subcat_rev_col1, subcat_rev_col2 = st.columns(2)
    with subcat_rev_col1:
        subcategory_revenue = analysis_results['revenue_analysis']['top_subcategories']
        subcat_rev_markdown = ""
        for i, (subcat, revenue) in enumerate(list(subcategory_revenue.items())[:5], 1):
            subcat_rev_markdown += f"{i}. {subcat}: ${revenue:,.2f}\n"
        st.markdown(subcat_rev_markdown)
    with subcat_rev_col2:
        top_subcat_rev = list(subcategory_revenue.items())[0]
        st.info(f"{top_subcat_rev[0]} services are the highest revenue-generating subcategory at ${top_subcat_rev[1]:,.2f}, demonstrating the importance of these services to the business.")
else:
    st.error("Failed to load data. Please check if the CSV files exist and are accessible.")

    # Add footer with attribution
line_break()
line_break()
line_break()
st.markdown("""
<div style='text-align: center; padding: 20px; position: fixed; bottom: 0; left: 0; right: 0; background-color: #e6e6fa; color: #000080;'>
    <p>Submitted by Ashar</p>
</div>
""", unsafe_allow_html=True)
