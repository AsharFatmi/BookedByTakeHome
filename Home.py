import streamlit as st
import json

# Set page configuration
st.set_page_config(
    page_title="Report of the project",
    layout="wide",
    initial_sidebar_state="expanded"
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

# Add a title to the main page
st.title("Report of the Take Home Project")
line_break()
line_break()
line_break()
line_break()

# Main page content
col1, col2 = st.columns([2, 1])

with col1:
    st.write("This is an app for the report of the take home assignment submission")

with col2:
    st.write("By:- Ashar Fatmi")

line_break()
line_break()
st.write("Note:- This page is just the overview and the conclusion for the findings and analysis go to the specific task from the sidebar to view the detailed report")



line_break()
line_break()
line_break()
st.markdown("""
<div style='text-align: center; padding: 20px; position: fixed; bottom: 0; left: 0; right: 0; background-color: #e6e6fa; color: #000080;'>
    <p>Submitted by Ashar</p>
</div>
""", unsafe_allow_html=True) 