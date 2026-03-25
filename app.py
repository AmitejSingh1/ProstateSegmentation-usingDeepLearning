"""
Streamlit web application for PyTorch Prostate Segmentation using VGG16-UNet.

Run locally:
    streamlit run app.py
"""

import os
import numpy as np
import cv2
import streamlit as st
import torch
from PIL import Image

from model import build_vgg16_unet
from config import H, W, MODEL_PATH

# Update expected model path to PyTorch .pth format
PTH_MODEL_PATH = MODEL_PATH.replace('.keras', '.pth')

# Add this URL after you upload the model.pth to a GitHub release
MODEL_URL = "https://github.com/AmitejSingh1/ProstateSegmentation-usingDeepLearning/releases/download/v1.0/model.pth"

# ---- Page Config ----
st.set_page_config(
    page_title="Prostate Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---- Custom CSS ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%); }

    .main-header { text-align: center; padding: 2rem 0 1rem; }
    .main-header h1 {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p { color: #a0a0c0; font-size: 1.05rem; font-weight: 300; }

    .glass-card {
        background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px; padding: 1.5rem; backdrop-filter: blur(12px); margin-bottom: 1rem;
    }
    .glass-card h3 { color: #c0c0e0; font-weight: 600; margin-bottom: 0.8rem; font-size: 1.1rem; }

    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
        border: 1px solid rgba(102, 126, 234, 0.25); border-radius: 12px;
        padding: 1rem 1.2rem; text-align: center;
    }
    .metric-card .label { color: #a0a0c0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-card .value { color: #e0e0ff; font-size: 1.6rem; font-weight: 700; }

    .stFileUploader {
        border: 2px dashed rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important; background: rgba(102, 126, 234, 0.05) !important;
    }

    section[data-testid="stSidebar"] { background: rgba(15, 12, 41, 0.95); border-right: 1px solid rgba(255, 255, 255, 0.06); }
    .status-ready { color: #6ee7b7; font-weight: 600; }
    .status-missing { color: #fca5a5; font-weight: 600; }

    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pytorch_model():
    """Load the trained PyTorch segmentation model (cached)."""
    if not os.path.exists(PTH_MODEL_PATH):
        # Try to download the model from GitHub Releases if we are on the cloud
        try:
            st.info(f"Downloading 98MB model weights from GitHub Release...")
            # Make sure files directory exists
            os.makedirs(os.path.dirname(PTH_MODEL_PATH), exist_ok=True)
            torch.hub.download_url_to_file(MODEL_URL, PTH_MODEL_PATH)
            st.success("Download complete!")
        except Exception as e:
            st.error(f"Failed to auto-download model: {e}")
            return None, None
            
    if not os.path.exists(PTH_MODEL_PATH):
        return None, None
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_vgg16_unet()
    model.load_state_dict(torch.load(PTH_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def preprocess_image(image_array):
    """Preprocess uploaded image for PyTorch model input."""
    # Resize
    img = cv2.resize(image_array, (W, H))

    # Handle channels
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    # Normalize & to PyTorch layout (CHW)
    img_normalized = img / 255.0
    img_normalized = img_normalized.astype(np.float32)
    img_tensor = np.transpose(img_normalized, (2, 0, 1))
    
    return img, img_tensor


def predict_mask(model, device, img_tensor):
    """Run PyTorch inference and return binary mask."""
    x = torch.tensor(img_tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        raw_pred = model(x).squeeze().cpu().numpy()
        
    mask = (raw_pred > 0.5).astype(np.float32)
    return mask, raw_pred


def create_overlay(image, mask, alpha=0.4):
    """Create a colored overlay of the mask on the original image."""
    overlay = image.copy()
    color = np.array([180, 100, 234])  # Purple-blue BGR
    mask_bool = mask > 0.5
    overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + color * alpha).astype(np.uint8)

    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (130, 230, 255), 2)
    return overlay


def mask_to_download(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)
    _, buffer = cv2.imencode(".png", mask_uint8)
    return buffer.tobytes()


def calculate_3d_volume(img_array3d, model, device, threshold, area_ratio, voxel_vol_mm3):
    total_pixels = 0
    batch_size = 16
    num_slices = img_array3d.shape[0]
    
    # Process slices in batches for speed
    for i in range(0, num_slices, batch_size):
        batch = img_array3d[i:i+batch_size]
        tensors = []
        for slice_2d in batch:
            norm = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            color = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
            _, img_tensor = preprocess_image(color)
            tensors.append(img_tensor)
            
        batch_tensor = torch.tensor(np.array(tensors)).to(device)
        with torch.no_grad():
            preds = model(batch_tensor).squeeze(1).cpu().numpy()
            
        if len(preds.shape) == 2:
            preds = np.expand_dims(preds, axis=0)
            
        masks = (preds > threshold).astype(np.float32)
        total_pixels += masks.sum()
        
    est_org_pixels = total_pixels * area_ratio
    vol_mm3 = est_org_pixels * voxel_vol_mm3
    vol_cc = vol_mm3 / 1000.0  # 1000 mm^3 = 1 cc (mL)
    return vol_cc


# ===================== MAIN APP =====================

st.markdown("""
<div class="main-header">
    <h1>🔬 Prostate Segmentation</h1>
    <p>AI-powered prostate boundary detection from micro-ultrasound images using PyTorch</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    model, device = load_pytorch_model()
    if model is not None:
        st.markdown(f'<p class="status-ready">✅ Model Loaded</p>', unsafe_allow_html=True)
        st.markdown(f"<small style='color:#a0a0c0'>Engine: PyTorch <b>{device.type.upper()}</b></small>", unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-missing">❌ No Model Found</p>', unsafe_allow_html=True)
        st.info(f"Expected at:\n`{PTH_MODEL_PATH}`\n\nTrain a model first:\n```\npython train.py\n```")

    st.markdown("---")
    threshold = st.slider("Segmentation Threshold", 0.1, 0.9, 0.5, 0.05)
    overlay_alpha = st.slider("Overlay Opacity", 0.1, 0.8, 0.4, 0.05)
    
    st.markdown("---")
    st.markdown("### 📏 Advanced Clinical Settings")
    st.info("Some NIfTI ultrasound files lack physical dimension metadata. You can manually set the pixel scale here if your calculated volume is incorrect.")
    
    colA, colB = st.columns(2)
    with colA:
        manual_spacing_xy = st.number_input("In-Plane (X/Y mm)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)
    with colB:
        manual_spacing_z = st.number_input("Slice Thickness (Z mm)", min_value=0.01, max_value=5.0, value=1.0, step=0.1)

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    <div class="glass-card">
        <p style="color: #a0a0c0; font-size: 0.85rem; line-height: 1.6;">
            <strong style="color: #c0c0e0;">Architecture:</strong> VGG16-UNet<br>
            <strong style="color: #c0c0e0;">Framework:</strong> PyTorch<br>
            <strong style="color: #c0c0e0;">Input:</strong> 512 × 512 × 3<br>
        </p>
    </div>
    """, unsafe_allow_html=True)


if model is None:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 3rem;">
        <h3 style="font-size: 1.4rem;">🚀 Get Started</h3>
        <p style="color: #a0a0c0; font-size: 1rem; max-width: 600px; margin: 0 auto;">
            No trained PyTorch model found. Follow these steps to set up:
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h3>1️⃣ Prepare Data</h3><p style="color: #a0a0c0; font-size: 0.9rem;">Convert NIfTI volumes to PNG slices</p>
            <code style="color: #667eea;">python preprocess.py</code>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h3>2️⃣ Train Model</h3><p style="color: #a0a0c0; font-size: 0.9rem;">Train the PyTorch VGG16-UNet</p>
            <code style="color: #667eea;">python train.py --epochs 50</code>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h3>3️⃣ Segment!</h3><p style="color: #a0a0c0; font-size: 0.9rem;">Get instant results here</p>
            <code style="color: #667eea;">streamlit run app.py</code>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown('<div class="glass-card"><h3>📤 Upload Micro-Ultrasound Image</h3></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a PNG, JPEG, or NIfTI volume", type=["png", "jpg", "jpeg", "nii", "nii.gz"])

    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        raw_image = None
        is_nifti = False
        volume_cc = None
        
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            is_nifti = True
            import tempfile
            import SimpleITK as sitk
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
                
            try:
                sitk_img = sitk.ReadImage(tmp_path)
                
                # Use overridden spacing if changed, otherwise trust the file
                spacing = sitk_img.GetSpacing()  # (x, y, z) in mm
                
                sp_x = manual_spacing_xy if manual_spacing_xy != 1.0 else spacing[0]
                sp_y = manual_spacing_xy if manual_spacing_xy != 1.0 else spacing[1]
                sp_z = manual_spacing_z if manual_spacing_z != 1.0 else spacing[2]
                
                voxel_vol_mm3 = sp_x * sp_y * sp_z
                
                img_arr = sitk.GetArrayFromImage(sitk_img)
                org_z, org_y, org_x = img_arr.shape
                area_ratio = (org_x * org_y) / (W * H)
            finally:
                os.remove(tmp_path)
                
            st.markdown('<div class="glass-card"><h3>🗜️ Volume Slice Selection</h3></div>', unsafe_allow_html=True)
            num_slices = img_arr.shape[0]
            if num_slices > 1:
                z_slice = st.slider(f"Select Z-Slice (0 - {num_slices-1})", 0, num_slices - 1, num_slices // 2)
            else:
                z_slice = 0
                st.info("Uploaded volume only contains 1 slice.")
                
            slice_data = img_arr[z_slice]
            
            # Normalize to 0-255 uint8 for processing
            norm_slice = cv2.normalize(slice_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            raw_image = cv2.cvtColor(norm_slice, cv2.COLOR_GRAY2BGR)
            
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            raw_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if raw_image is not None:
            # If NIfTI, optionally calculate entire volume
            if is_nifti and st.sidebar.button("Calculate Total 3D Volume", use_container_width=True):
                with st.spinner("Running deep learning inference across all slices..."):
                    volume_cc = calculate_3d_volume(img_arr, model, device, threshold, area_ratio, voxel_vol_mm3)

            display_img, img_tensor = preprocess_image(raw_image)
            mask, raw_pred = predict_mask(model, device, img_tensor)

            mask = (raw_pred > threshold).astype(np.float32)
            overlay = create_overlay(display_img, mask, alpha=overlay_alpha)

            st.markdown('<div class="glass-card"><h3>🎯 Segmentation Results</h3></div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Original Image**")
                st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.markdown("**Predicted Mask**")
                st.image(mask, use_container_width=True, clamp=True)
            with col3:
                st.markdown("**Overlay**")
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown('<div class="glass-card"><h3>📈 Prediction Statistics</h3></div>', unsafe_allow_html=True)

            total_pixels = mask.shape[0] * mask.shape[1]
            prostate_pixels = int(mask.sum())
            prostate_pct = (prostate_pixels / total_pixels) * 100
            confidence = float(raw_pred[mask > 0.5].mean()) * 100 if prostate_pixels > 0 else 0

            m1, m2, m3, m4 = st.columns(4)
            with m1: st.markdown(f'<div class="metric-card"><div class="label">Image Size</div><div class="value">{H}×{W}</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="metric-card"><div class="label">Slice Area</div><div class="value">{prostate_pct:.1f}%</div></div>', unsafe_allow_html=True)
            
            if is_nifti and volume_cc is not None:
                with m3: st.markdown(f'<div class="metric-card"><div class="label">Total Vol (cc)</div><div class="value" style="color:#6ee7b7;">{volume_cc:.1f} cc</div></div>', unsafe_allow_html=True)
            else:
                with m3: st.markdown(f'<div class="metric-card"><div class="label">Slice Pixels</div><div class="value">{prostate_pixels:,}</div></div>', unsafe_allow_html=True)
                
            with m4: st.markdown(f'<div class="metric-card"><div class="label">Avg Confidence</div><div class="value">{confidence:.1f}%</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            dl1, dl2, _ = st.columns([1, 1, 2])
            with dl1:
                st.download_button("⬇️ Download Mask", data=mask_to_download(mask), file_name=f"mask_{uploaded_file.name}", mime="image/png")
            with dl2:
                overlay_bytes = cv2.imencode(".png", overlay)[1].tobytes()
                st.download_button("⬇️ Download Overlay", data=overlay_bytes, file_name=f"overlay_{uploaded_file.name}", mime="image/png")
