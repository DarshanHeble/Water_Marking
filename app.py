import streamlit as st
import numpy as np
from PIL import Image
import io
from watermark_utils import (
    add_visible_watermark,
    add_invisible_watermark,
    add_semi_visible_watermark,
    extract_invisible_watermark,
    verify_watermark,
    analyze_attack_resistance,
    optimize_watermark_strength,
)
from quality_metrics import calculate_psnr, calculate_ssim, analyze_resistance

st.set_page_config(page_title="Digital Watermarking Platform", layout="wide")

st.title("Digital Watermarking Platform")
st.markdown(
    """
This platform demonstrates various digital watermarking techniques with quality analysis.
Choose a technique and customize the watermark parameters below.
"""
)

# Sidebar controls
st.sidebar.header("Watermark Settings")
watermark_type = st.sidebar.selectbox(
    "Select Watermark Type", ["Visible", "Invisible", "Semi-visible"]
)

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and display original image
    original_image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)

    try:
        # Apply watermark based on selection
        if watermark_type == "Visible":
            watermark_text = st.sidebar.text_input("Watermark Text", "Copyright 2024")
            opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 0.3)
            watermarked_image = add_visible_watermark(
                original_image, watermark_text, opacity
            )

        elif watermark_type == "Invisible":
            message = st.sidebar.text_input("Hidden Message", "Secret Watermark")
            strength = st.sidebar.slider("Watermark Strength", 0.1, 1.0, 0.5)

            # Optimize watermark strength
            if st.sidebar.button("Optimize Watermark Strength"):
                with st.spinner("Optimizing watermark strength..."):
                    optimal_strength = optimize_watermark_strength(
                        original_image, message
                    )
                    st.sidebar.success(f"Optimal strength: {optimal_strength:.2f}")
                    strength = optimal_strength

            watermarked_image = add_invisible_watermark(
                original_image, message, strength
            )
            extraction_result = extract_invisible_watermark(watermarked_image)

            if extraction_result["integrity_verified"]:
                st.sidebar.success(
                    f"✅ Extracted message: {extraction_result['message']}\nIntegrity verified!"
                )
            else:
                st.sidebar.warning(
                    f"⚠️ Extracted message: {extraction_result['message']}\nIntegrity check failed!"
                )

        else:  # Semi-visible
            pattern_strength = st.sidebar.slider("Pattern Strength", 0.05, 0.3, 0.1)
            watermarked_image = add_semi_visible_watermark(
                original_image, pattern_strength
            )

        with col2:
            st.subheader("Watermarked Image")
            st.image(watermarked_image, use_container_width=True)

        # Watermark Verification
        st.subheader("Watermark Verification")
        verification_result = verify_watermark(original_image, watermarked_image)

        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric(
                "Watermark Presence",
                (
                    "Detected"
                    if verification_result["watermark_present"]
                    else "Not Detected"
                ),
            )
        with col4:
            st.metric(
                "Similarity Score", f"{verification_result['similarity_score']:.3f}"
            )
        with col5:
            st.metric(
                "Tampering Status",
                "Tampered" if verification_result["tampering_detected"] else "Original",
            )

        # Show difference map
        st.subheader("Watermark Location Map")
        st.image(
            verification_result["difference_map"],
            use_container_width=True,
            caption="Brighter areas show where the watermark is embedded",
        )

        # Attack Resistance Analysis
        st.subheader("Attack Resistance Analysis")
        with st.expander("View Attack Resistance Details"):
            resistance_results = analyze_attack_resistance(watermarked_image)

            for attack_type, results in resistance_results.items():
                cols = st.columns(2)
                with cols[0]:
                    st.metric(
                        f"{attack_type.title()} PSNR", f"{results['psnr']:.2f} dB"
                    )
                with cols[1]:
                    status = "✅ Survived" if results["survived"] else "❌ Failed"
                    st.metric(f"{attack_type.title()} Test", status)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
