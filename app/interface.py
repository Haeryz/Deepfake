import streamlit as st
from PIL import Image
import pandas as pd

def run_interface(models_dict):
    """
    Run the Streamlit interface for deepfake detection.

    Args:
        models_dict: Dictionary containing loaded models from load_all_models()
    """
    st.title("🕵️ Deepfake Detection System")
    st.markdown("Upload an image to check if it's AI-generated or real using your trained models.")

    # Model selection
    available_models = {
        'custom_cnn': 'CustomCNN',
        'efficientnet': 'EfficientNet',
        'mobilenet': 'MobileNet'
    }

    selected_models = st.multiselect(
        "Select models to use for analysis:",
        options=list(available_models.keys()),
        default=list(available_models.keys()),
        format_func=lambda x: available_models[x],
        help="Choose which models to run. At least one model must be selected."
    )

    if not selected_models:
        st.warning("Please select at least one model to proceed.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

        # Add a button to run analysis
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner(f"Analyzing image with {len(selected_models)} selected model{'s' if len(selected_models) > 1 else ''}..."):
                # Import the prediction function
                from .load_all_models import predict_image

                # Filter models_dict to only include selected models and their processors
                filtered_models = {}
                for key in selected_models:
                    if key in models_dict:
                        filtered_models[key] = models_dict[key]
                        # Also include processor if it exists
                        processor_key = f"{key}_processor"
                        if processor_key in models_dict:
                            filtered_models[processor_key] = models_dict[processor_key]

                # Run predictions
                results = predict_image(image, filtered_models)

            # Display results
            st.success("Analysis Complete!")

            # Create a nice display for results
            st.subheader("📊 Detection Results")

            # Prepare data for display
            model_names = []
            predictions = []
            fake_probs = []
            real_probs = []

            for model_name, result in results.items():
                model_names.append(result['model'])
                predictions.append(result['prediction'])
                fake_probs.append(f"{result['fake_probability']:.3f}")
                real_probs.append(f"{result['real_probability']:.3f}")

            # Create a dataframe for nice table display
            results_df = pd.DataFrame({
                'Model': model_names,
                'Prediction': predictions,
                'Fake Probability': fake_probs,
                'Real Probability': real_probs
            })

            # Display as table
            st.dataframe(results_df, use_container_width=True)

            # Add some interpretation
            fake_count = sum(1 for r in results.values() if r['prediction'] == 'FAKE')
            real_count = len(results) - fake_count

            st.subheader("🎯 Summary")
            total_models = len(results)
            if fake_count > real_count:
                st.error(f"⚠️ {fake_count} out of {total_models} model{'s' if total_models > 1 else ''} detected this as FAKE (AI-generated)")
            elif real_count > fake_count:
                st.success(f"✅ {real_count} out of {total_models} model{'s' if total_models > 1 else ''} detected this as REAL")
            else:
                st.warning(f"🤔 Mixed results - {fake_count} model{'s' if fake_count > 1 else ''} detected FAKE, {real_count} detected REAL")

            # Show detailed probabilities
            st.subheader("📈 Detailed Probabilities")
            if len(results) == 1:
                # Single model - show centered
                model_key = list(results.keys())[0]
                st.metric(
                    label=f"{results[model_key]['model']}",
                    value=results[model_key]['prediction'],
                    delta=f"Fake: {results[model_key]['fake_probability']:.3f}"
                )
            elif len(results) == 2:
                # Two models - show in two columns
                col1, col2 = st.columns(2)
                model_keys = list(results.keys())
                with col1:
                    st.metric(
                        label=f"{results[model_keys[0]]['model']}",
                        value=results[model_keys[0]]['prediction'],
                        delta=f"Fake: {results[model_keys[0]]['fake_probability']:.3f}"
                    )
                with col2:
                    st.metric(
                        label=f"{results[model_keys[1]]['model']}",
                        value=results[model_keys[1]]['prediction'],
                        delta=f"Fake: {results[model_keys[1]]['fake_probability']:.3f}"
                    )
            else:
                # Three models - show in three columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label=f"{results['custom_cnn']['model']}",
                        value=results['custom_cnn']['prediction'],
                        delta=f"Fake: {results['custom_cnn']['fake_probability']:.3f}"
                    )
                with col2:
                    st.metric(
                        label=f"{results['efficientnet']['model']}",
                        value=results['efficientnet']['prediction'],
                        delta=f"Fake: {results['efficientnet']['fake_probability']:.3f}"
                    )
                with col3:
                    st.metric(
                        label=f"{results['mobilenet']['model']}",
                        value=results['mobilenet']['prediction'],
                        delta=f"Fake: {results['mobilenet']['fake_probability']:.3f}"
                    )

    else:
        st.info("👆 Please upload an image to get started.")

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and PyTorch*")