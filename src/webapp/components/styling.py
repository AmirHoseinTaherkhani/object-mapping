"""
Streamlit styling utilities for Video-to-Map webapp
Light professional theme for webapp and future personal website
"""
import streamlit as st

def inject_custom_css():
    """Inject custom CSS styling into Streamlit app"""
    css = """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --primary-bg: #ffffff;
        --secondary-bg: #f8f9fa;
        --accent-color: #0066cc;
        --accent-light: #e3f2fd;
        --text-primary: #2c3e50;
        --text-secondary: #6c757d;
        --border-color: #dee2e6;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --error-color: #dc3545;
        --shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        --shadow-hover: 0 4px 20px rgba(0, 0, 0, 0.15);
        --radius: 12px;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Main app styling */
    .stApp {
        background-color: var(--primary-bg);
        color: var(--text-primary);
        font-family: Inter, sans-serif;
    }

    /* Headers and text */
    h1, h2, h3 {
        color: var(--text-primary);
        font-family: Inter, sans-serif;
    }

    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, var(--accent-color) 0%, #004499 100%);
        color: white;
        border: none;
        border-radius: var(--radius);
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow);
        font-family: Inter, sans-serif;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-hover);
    }

    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed var(--border-color);
        border-radius: var(--radius);
        padding: 2rem;
        background: var(--secondary-bg);
        transition: all 0.3s ease;
    }

    .stFileUploader:hover {
        border-color: var(--accent-color);
        background: var(--accent-light);
    }

    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent-color) 0%, #004499 100%);
        border-radius: 10px;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: var(--secondary-bg);
        border-right: 1px solid var(--border-color);
    }

    /* Columns */
    .stColumns {
        gap: 1.5rem;
    }

    /* Info boxes */
    .stAlert {
        border-radius: var(--radius);
        border: none;
        box-shadow: var(--shadow);
    }
    
    /* Selectbox and other inputs */
    .stSelectbox > div > div {
        background-color: var(--primary-bg);
        border: 2px solid var(--border-color);
        border-radius: var(--radius);
        color: var(--text-primary);
    }

    /* Number input */
    .stNumberInput > div > div > input {
        background-color: var(--primary-bg);
        border: 2px solid var(--border-color);
        border-radius: var(--radius);
        color: var(--text-primary);
    }

    /* Sidebar title styling */
    .css-1d391kg h1 {
        color: var(--accent-color);
        font-weight: 700;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def render_app_header(title: str, subtitle: str = ""):
    """Render styled app header"""
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<p style="font-size: 1.2rem; color: #6c757d; margin: 0; font-family: Inter, sans-serif; font-weight: 400;">{subtitle}</p>'
    
    header_html = f"""
    <div style="
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 3rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
        border: 1px solid #dee2e6;
    ">
        <h1 style="
            font-size: 2.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
            font-family: Inter, sans-serif;
        ">{title}</h1>
        {subtitle_html}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_feature_card(title: str, content: str, icon: str = "🔧"):
    """Render styled feature card"""
    card_html = f"""
    <div style="
        background: #ffffff;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border: 1px solid #dee2e6;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    " onmouseover="this.style.boxShadow='0 4px 20px rgba(0, 0, 0, 0.15)'; this.style.transform='translateY(-2px)'" 
       onmouseout="this.style.boxShadow='0 2px 10px rgba(0, 0, 0, 0.1)'; this.style.transform='translateY(0)'">
        <h3 style="
            color: #0066cc;
            font-size: 1.5rem;
            margin-bottom: 1rem;
            font-weight: 600;
            font-family: Inter, sans-serif;
        ">{icon} {title}</h3>
        <p style="
            color: #6c757d;
            line-height: 1.6;
            margin: 0;
            font-family: Inter, sans-serif;
            font-size: 1rem;
        ">{content}</p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def render_metric_card(value: str, label: str, color: str = "#0066cc"):
    """Render styled metric card"""
    metric_html = f"""
    <div style="
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border: 1px solid #dee2e6;
        text-align: center;
        transition: all 0.3s ease;
    " onmouseover="this.style.boxShadow='0 4px 20px rgba(0, 0, 0, 0.15)'; this.style.transform='translateY(-2px)'" 
       onmouseout="this.style.boxShadow='0 2px 10px rgba(0, 0, 0, 0.1)'; this.style.transform='translateY(0)'">
        <div style="
            font-size: 2.2rem;
            font-weight: 700;
            color: {color};
            font-family: Inter, sans-serif;
        ">{value}</div>
        <div style="
            color: #6c757d;
            font-size: 0.95rem;
            margin-top: 0.5rem;
            font-family: Inter, sans-serif;
            font-weight: 500;
        ">{label}</div>
    </div>
    """
    st.markdown(metric_html, unsafe_allow_html=True)

def add_footer():
    """Add styled footer"""
    footer_html = """
    <div style="
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid #dee2e6;
        color: #6c757d;
        font-family: Inter, sans-serif;
    ">
        <p style="margin: 0; font-size: 0.95rem;">Video-to-Map Object Tracking System | Built with ❤️ and ML</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
