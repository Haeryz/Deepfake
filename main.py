"""
Main entry point for the Deepfake Detection Web Application
Loads all models and runs the Streamlit interface.
"""

import streamlit as st
from app.load_all_models import load_all_models
from app.interface import run_interface

def main():
    """
    Main function to run the deepfake detection application.
    """
    # Configure the page
    st.set_page_config(
        page_title="Deepfake Detector",
        page_icon="🕵️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load models with caching to avoid reloading on every rerun
    @st.cache_resource
    def load_models_cached():
        """Load all models and cache them for performance."""
        print("Loading models...")  # This will show in terminal
        return load_all_models()

    # Load the models
    models = load_models_cached()

    # Run the interface
    run_interface(models)

if __name__ == "__main__":
    main()