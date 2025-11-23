import streamlit as st
from PIL import Image
import io
from utils import simulate_cvd_pil, daltonize_pil, add_pattern_overlay

st.set_page_config(page_title="Color Blindness Accessibility Tool", layout="wide")
st.title("Color Blindness Accessibility Tool — Python (Streamlit)")


st.sidebar.header("Settings")
mode = st.sidebar.selectbox("Choose deficiency", ["protanopia", "deuteranopia", "tritanopia"])
strength = st.sidebar.slider("Correction strength", 0.0, 2.0, 1.0, 0.1)
view = st.sidebar.radio("View", ["Original / Simulated / Corrected", "Side-by-side"])


st.sidebar.markdown("### Pattern overlay (optional)")
apply_pattern = st.sidebar.checkbox("Apply pattern overlay to corrected image", value=False)
pattern_style = st.sidebar.selectbox("Pattern style", ["dots", "stripes"])
pattern_tile = st.sidebar.slider("Pattern density (tile size)", 6, 30, 12)
pattern_threshold = st.sidebar.slider("Detect threshold", 10, 200, 60)
pattern_color_choice = st.sidebar.selectbox("Pattern color", ["black", "white"])
pattern_alpha = st.sidebar.slider("Pattern opacity", 50, 255, 140)

pattern_color = (0,0,0,pattern_alpha) if pattern_color_choice == "black" else (255,255,255,pattern_alpha)

st.write("Upload an image or use your camera (camera works in supported browsers).")


uploaded = st.file_uploader("Upload an image (png/jpg)", type=["png", "jpg", "jpeg"])
camera_file = st.camera_input("Or take a photo")

image = None
if uploaded:
    image = Image.open(uploaded).convert("RGB")
elif camera_file:
    image = Image.open(camera_file).convert("RGB")

if image is None:
    st.info("Upload an image or use the camera to get started.")
    st.stop()


with st.spinner("Processing..."):
    simulated = simulate_cvd_pil(image, mode)
    corrected = daltonize_pil(image, mode, strength=strength)


if view == "Original / Simulated / Corrected":
    st.header("Original")
    st.image(image, use_column_width=True)
    st.header(f"Simulated ({mode})")
    st.image(simulated, use_column_width=True)
else:
    a, b = st.columns(2)
    a.subheader("Original"); a.image(image, use_column_width=True)
    b.subheader(f"Simulated ({mode})"); b.image(simulated, use_column_width=True)


if apply_pattern:
    overlayed = add_pattern_overlay(original_pil=image,
                                    corrected_pil=corrected,
                                    mode=mode,
                                    threshold=pattern_threshold,
                                    pattern=pattern_style,
                                    tile=pattern_tile,
                                    color=pattern_color)
    st.header("Corrected + Pattern Overlay")
    st.image(overlayed, use_column_width=True)
    buf2 = io.BytesIO()
    overlayed.save(buf2, format="PNG")
    buf2.seek(0)
    st.download_button("Download corrected + pattern (PNG)",
                       data=buf2,
                       file_name="corrected_with_pattern.png",
                       mime="image/png")
else:
    st.header("Corrected")
    st.image(corrected, use_column_width=True)
    buf = io.BytesIO()
    corrected.save(buf, format="PNG")
    buf.seek(0)
    st.download_button("Download corrected (PNG)",
                       data=buf,
                       file_name="corrected.png",
                       mime="image/png")

st.markdown("---")
st.info("Tip: Try images with red/green charts or traffic lights to see improvement.")