import streamlit as st
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.ops import nms as torch_nms
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time
import gdown

def download_file_from_google_drive(file_id, destination):
    """Downloads a large file from Google Drive using gdown."""
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)

# ---------------------------------------------------------
# Page Configuration & Aesthetics Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Lunar Safe Landing Site Analyzer",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set custom CSS styles for a premium space/dark UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600;700&display=swap');

    /* Global styling */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0b0f19;
        background-image: radial-gradient(circle at 50% 50%, #161e33 0%, #0b0f19 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background-color: rgba(11, 15, 25, 0.4);
        backdrop-filter: blur(10px);
    }

    [data-testid="stSidebar"] {
        background-color: #0a0d17;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Sidebar headers and widgets */
    .css-1d391tw, .css-1avcm0n {
        color: #f1f5f9;
    }
    
    /* Headers with gradient text */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        color: #ffffff;
        background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }

    /* KPI metric cards */
    .kpi-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 25px;
        margin-top: 10px;
    }

    .kpi-card {
        flex: 1;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 20px 15px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.3);
        box-shadow: 0 6px 24px rgba(96, 165, 250, 0.12);
        background: rgba(255, 255, 255, 0.04);
    }

    .kpi-value {
        font-size: 2.0rem;
        font-weight: 800;
        color: #60a5fa;
        font-family: 'Outfit', sans-serif;
        margin-bottom: 4px;
        text-shadow: 0 0 12px rgba(96, 165, 250, 0.25);
    }

    .kpi-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Custom status box */
    .status-box {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 8px;
        padding: 12px 18px;
        margin: 15px 0;
        font-size: 0.95rem;
        color: #34d399;
    }

    /* Tabs customized */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #94a3b8;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(96, 165, 250, 0.08) !important;
        border-color: rgba(96, 165, 250, 0.2) !important;
        color: #60a5fa !important;
    }
</style>
""", unsafe_allow_html=True)

# Apply beautiful dark style to Matplotlib
def apply_matplotlib_style():
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#0b0f19'
    plt.rcParams['axes.facecolor'] = '#0b0f19'
    plt.rcParams['savefig.facecolor'] = '#0b0f19'
    plt.rcParams['axes.edgecolor'] = (1.0, 1.0, 1.0, 0.1)
    plt.rcParams['grid.color'] = (1.0, 1.0, 1.0, 0.05)
    plt.rcParams['font.family'] = 'sans-serif'

apply_matplotlib_style()

# ---------------------------------------------------------
# Helper Functions & Computation Logic
# ---------------------------------------------------------

@st.cache_resource
def load_yolo_model(weights_path):
    """Loads and caches the YOLOv8 model."""
    try:
        return YOLO(weights_path)
    except Exception as e:
        st.error(f"Error loading model from {weights_path}: {e}")
        return None

def get_window_positions(image_h, image_w, window_size, overlap):
    """Computes top-left (x, y) coordinates for sliding windows."""
    step = int(window_size * (1 - overlap))

    y_starts = list(range(0, image_h - window_size + 1, step))
    if not y_starts or y_starts[-1] + window_size < image_h:
        y_starts.append(max(0, image_h - window_size))

    x_starts = list(range(0, image_w - window_size + 1, step))
    if not x_starts or x_starts[-1] + window_size < image_w:
        x_starts.append(max(0, image_w - window_size))

    return [(x, y) for y in y_starts for x in x_starts]

def run_model_inference(image_bgr, model, window_size, overlap):
    """Runs sliding window inference on the full image."""
    img_h, img_w = image_bgr.shape[:2]
    positions = get_window_positions(img_h, img_w, window_size, overlap)
    
    all_boxes = []
    all_scores = []
    
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    total = len(positions)
    for idx, (x, y) in enumerate(positions):
        patch = image_bgr[y:y+window_size, x:x+window_size]
        # Run inference on crop patch (disable verbosity for cleaner stdout)
        # Using base_conf=0.1 to capture candidates, we filter by CONF_THRESHOLD later in pandas
        results = model(patch, conf=0.1, verbose=False)
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                
                # Offset coordinates back to global image frame
                boxes[:, 0] += x
                boxes[:, 2] += x
                boxes[:, 1] += y
                boxes[:, 3] += y
                
                all_boxes.extend(boxes.tolist())
                all_scores.extend(scores.tolist())
                
        # Update UI progress
        progress_bar.progress((idx + 1) / total)
        status_text.text(f"Scanning Moon surface... Window {idx+1}/{total} processed. Detections found: {len(all_boxes)}")
        
    # Clean up progress bar
    progress_bar.empty()
    status_text.empty()
    
    return all_boxes, all_scores

def apply_nms(boxes, scores, iou_threshold=0.5):
    """Performs Non-Maximum Suppression via torchvision.ops.nms."""
    if len(boxes) == 0:
        return [], []

    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)

    keep = torch_nms(boxes_t, scores_t, iou_threshold)

    kept_boxes = boxes_t[keep].numpy().tolist()
    kept_scores = scores_t[keep].numpy().tolist()

    return kept_boxes, kept_scores

def normalize(grid):
    """Min-max scaling helper to normalize matrices to [0, 1]."""
    min_val = grid.min()
    max_val = grid.max()

    if max_val == min_val:
        return np.zeros_like(grid, dtype=float)

    return (grid - min_val) / (max_val - min_val)

# ---------------------------------------------------------
# Sidebar Panel Layout
# ---------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/FullMoon2010.jpg/280px-FullMoon2010.jpg", width=120)
st.sidebar.markdown("<h2 style='margin-top:0px;'>Analyzer Panel</h2>", unsafe_allow_html=True)

# 1. Image Selection
st.sidebar.subheader("Image Input")
source_type = st.sidebar.radio("Source Image:", ["Use Sample Image", "Upload any new image from Lunar images collected by telescopes : https://data.im-ldi.com/mds?MDS_SEARCH=%7B%22datasets%22%3A%5B%22luna_lroc_fi%22%2C%22luna_lroc_pds_nac_edrcdr%22%2C%22luna_lroc_pds_wac_edrcdr%22%2C%22luna_lroc_pds_rdr%22%5D%2C%22query%22%3A%7B%7D%2C%22map%22%3A%7B%7D%7D"])

sample_choice = 1
uploaded_file = None

if source_type == "Use Sample Image":
    sample_choice = st.sidebar.selectbox("Select Sample:", [1, 2, 3, 4, 5], format_func=lambda x: f"Sample Lunar Region {x}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload Lunar Image (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

# 2. Main Parameters
st.sidebar.subheader("Geological Calibration")
pixels_per_km = st.sidebar.number_input("Pixels per Kilometer (Scale):", min_value=1.0, max_value=200.0, value=10.0, step=0.5)

st.sidebar.subheader("YOLOv8 Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold:", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
iou_threshold = st.sidebar.slider("NMS IoU Threshold:", min_value=0.1, max_value=1.0, value=0.5, step=0.05)

st.sidebar.subheader("Sliding Window Settings")
window_size = st.sidebar.number_input("Window Size (pixels):", min_value=128, max_value=2000, value=640, step=32)
overlap = st.sidebar.slider("Overlap Fraction:", min_value=0.0, max_value=0.8, value=0.2, step=0.05)

st.sidebar.subheader("Grid Hazard Settings")
grid_n = st.sidebar.slider("Grid Dimension (N x N):", min_value=4, max_value=30, value=8, step=1)
safety_threshold = st.sidebar.slider("Safety Cutoff Score:", min_value=0.1, max_value=0.9, value=0.3, step=0.05)

# ---------------------------------------------------------
# Core Loading & State Management
# ---------------------------------------------------------
# Load model
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

if not os.path.exists(WEIGHTS_PATH):
    # Check if a Google Drive File ID is configured in Secrets
    gdrive_id = None
    try:
        gdrive_id = st.secrets.get("GDRIVE_FILE_ID", None)
    except Exception:
        pass

    if gdrive_id:
        # Download automatically in the background
        with st.spinner("Downloading YOLOv8 weights from Google Drive... (89 MB)"):
            try:
                download_file_from_google_drive(gdrive_id, WEIGHTS_PATH)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to automatically download weights from Google Drive: {e}")
                st.stop()
    else:
        st.sidebar.warning("⚠️ Model weights file 'best.pt' not found locally.")
        gdrive_id_input = st.sidebar.text_input("Paste Google Drive File ID for best.pt:")
        if st.sidebar.button("Download Model from Google Drive"):
            if not gdrive_id_input:
                st.sidebar.error("Please enter a valid Google Drive File ID.")
            else:
                with st.spinner("Downloading YOLOv8 weights from Google Drive... (89 MB)"):
                    try:
                        download_file_from_google_drive(gdrive_id_input, WEIGHTS_PATH)
                        st.sidebar.success("Model downloaded successfully! Reloading...")
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Download failed: {e}")
        st.sidebar.info("💡 Tip: To download automatically on deployment, add GDRIVE_FILE_ID to your Streamlit App Secrets.")
        st.info("👈 Please enter your Google Drive File ID for `best.pt` in the sidebar to download and load the model.")
        st.stop()

model = load_yolo_model(WEIGHTS_PATH)

if model is None:
    st.error("Could not load the trained YOLO weights (`best.pt`). Please check the file and try again.")
    st.stop()

# Identify image path or uploaded data
image_bgr = None
image_name = ""

if source_type == "Use Sample Image":
    sample_path = os.path.join(os.path.dirname(__file__), "samples", f"sample{sample_choice}.png")
    if os.path.exists(sample_path):
        image_bgr = cv2.imread(sample_path)
        image_name = f"sample{sample_choice}.png"
    else:
        st.error(f"Sample file not found at {sample_path}")
        st.stop()
else:
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, 1)
        image_name = uploaded_file.name
    else:
        st.info("👈 Please upload a lunar image file from the sidebar to begin analysis.")
        st.stop()

# Compute a hash key to control caching in session state
image_key = f"{image_name}_{image_bgr.shape[0]}_{image_bgr.shape[1]}"
sliding_settings_key = f"{window_size}_{overlap}"

if "current_image_key" not in st.session_state or st.session_state.current_image_key != image_key or "current_sliding_key" not in st.session_state or st.session_state.current_sliding_key != sliding_settings_key:
    # Reset cached raw detections
    st.session_state.current_image_key = image_key
    st.session_state.current_sliding_key = sliding_settings_key
    st.session_state.raw_boxes = None
    st.session_state.raw_scores = None

# Convert BGR to RGB for matplotlib
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
img_h, img_w = image_bgr.shape[:2]

# ---------------------------------------------------------
# Main Page Title & Description
# ---------------------------------------------------------
st.markdown("<h1 style='text-align: center; margin-bottom: 5px;'>🌙 Lunar Landing Site Safe Zone Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 25px;'>Automated detection of lunar impact craters and geological safety mapping for space exploration landing profiles.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Processing Pipeline Execution
# ---------------------------------------------------------
# Step 1: Run sliding window inference if not cached in session state
if st.session_state.raw_boxes is None:
    with st.status("Analyzing Lunar Surface Terrain (Running Sliding Window Inference)...", expanded=True) as status:
        raw_boxes, raw_scores = run_model_inference(image_bgr, model, window_size, overlap)
        st.session_state.raw_boxes = raw_boxes
        st.session_state.raw_scores = raw_scores
        status.update(label="Inference completed! Processing data...", state="complete", expanded=False)

# Retrive raw outputs
raw_boxes = st.session_state.raw_boxes
raw_scores = st.session_state.raw_scores

# Step 2: Filter by user-defined confidence threshold
filtered_boxes = []
filtered_scores = []
for box, score in zip(raw_boxes, raw_scores):
    if score >= conf_threshold:
        filtered_boxes.append(box)
        filtered_scores.append(score)

# Step 3: Apply Non-Maximum Suppression (NMS)
final_boxes, final_scores = apply_nms(filtered_boxes, filtered_scores, iou_threshold)

# Step 4: Build pandas DataFrame and convert scale to km
records = []
for box, score in zip(final_boxes, final_scores):
    x1, y1, x2, y2 = box
    w_px = x2 - x1
    h_px = y2 - y1
    records.append({
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
        'confidence': score,
        'center_x_px': (x1 + x2) / 2,
        'center_y_px': (y1 + y2) / 2,
        'width_px': w_px,
        'height_px': h_px
    })

# If craters are detected
df = pd.DataFrame(records) if len(records) > 0 else pd.DataFrame(columns=[
    'x1', 'y1', 'x2', 'y2', 'confidence', 'center_x_px', 'center_y_px', 'width_px', 'height_px'
])

if len(df) > 0:
    df['center_x_km'] = df['center_x_px'] / pixels_per_km
    df['center_y_km'] = df['center_y_px'] / pixels_per_km
    df['width_km'] = df['width_px'] / pixels_per_km
    df['height_km'] = df['height_px'] / pixels_per_km
    df['diameter_km'] = (df['width_km'] + df['height_km']) / 2

# Compute grid properties
cell_h = img_h / grid_n
cell_w = img_w / grid_n
cell_area_km2 = (cell_h / pixels_per_km) * (cell_w / pixels_per_km)

# Step 5: Compute Grid-Based CDI (Crater Density Index)
count_grid = np.zeros((grid_n, grid_n))
for r_idx in range(grid_n):
    for c_idx in range(grid_n):
        cell_x1 = c_idx * cell_w
        cell_y1 = r_idx * cell_h
        cell_x2 = (c_idx + 1) * cell_w
        cell_y2 = (r_idx + 1) * cell_h

        if len(df) > 0:
            for _, row in df.iterrows():
                crater_x1, crater_y1, crater_x2, crater_y2 = row[['x1', 'y1', 'x2', 'y2']]
                # Intersection check between cell grid and bounding box
                if (crater_x1 < cell_x2 and crater_x2 > cell_x1 and
                    crater_y1 < cell_y2 and crater_y2 > cell_y1):
                    count_grid[r_idx, c_idx] += 1

cdi_grid = count_grid / cell_area_km2

# Step 6: Compute Size-Weighted Hazard Index (SWHI)
swhi_raw = np.zeros((grid_n, grid_n))
for r_idx in range(grid_n):
    for c_idx in range(grid_n):
        cell_x1 = c_idx * cell_w
        cell_y1 = r_idx * cell_h
        cell_x2 = (c_idx + 1) * cell_w
        cell_y2 = (r_idx + 1) * cell_h

        if len(df) > 0:
            for _, row in df.iterrows():
                crater_x1, crater_y1, crater_x2, crater_y2 = row[['x1', 'y1', 'x2', 'y2']]
                if (crater_x1 < cell_x2 and crater_x2 > cell_x1 and
                    crater_y1 < cell_y2 and crater_y2 > cell_y1):
                    swhi_raw[r_idx, c_idx] += np.log(row['diameter_km'])

swhi_grid = swhi_raw / cell_area_km2

# Step 7: Compute Combined Safety Score
cdi_norm = normalize(cdi_grid)
swhi_norm = normalize(swhi_grid)
safety_score = (cdi_norm + swhi_norm) / 2

# Identify safe cells and top 3 safest cells
safe_mask = safety_score <= safety_threshold
flat_scores = safety_score.flatten()
safest_indices = np.argsort(flat_scores)
top_cells = []
for idx in safest_indices:
    r = idx // grid_n
    c = idx % grid_n
    top_cells.append((r, c))
    if len(top_cells) == 3:
        break

# Show completion banner
st.markdown(
    f"<div class='status-box'>🚀 <b>Analysis complete!</b> Detected <b>{len(df)}</b> unique craters. "
    f"Found <b>{int(safe_mask.sum())}</b> landing cells meeting the safety criteria out of {grid_n * grid_n} total grid sections.</div>", 
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Dynamic KPI Dashboard Cards
# ---------------------------------------------------------
total_craters = len(df)
avg_diameter = df['diameter_km'].mean() if total_craters > 0 else 0.0
min_diameter = df['diameter_km'].min() if total_craters > 0 else 0.0
max_diameter = df['diameter_km'].max() if total_craters > 0 else 0.0
safe_cell_pct = (safe_mask.sum() / (grid_n * grid_n)) * 100

st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card">
        <div class="kpi-value">{total_craters}</div>
        <div class="kpi-label">Craters Detected</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{avg_diameter:.3f} km</div>
        <div class="kpi-label">Avg Crater Diameter</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{safe_mask.sum()} / {grid_n * grid_n}</div>
        <div class="kpi-label">Safe Landing Cells ({safe_cell_pct:.1f}%)</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Interactive Tabbed Visualizations
# ---------------------------------------------------------
tab_safety_zones, tab_detections, tab_cdi, tab_swhi, tab_safety_score, tab_data_report = st.tabs([
    "📍 Safe Landing Zones",
    "🟡 Detected Craters",
    "📊 Crater Density (CDI)",
    "🌋 Hazard Index (SWHI)",
    "🌡️ Combined Safety Heatmap",
    "🔬 Geological Assessment Report"
])

with tab_safety_zones:
    st.subheader("Safe and Hazardous Landing Site Overlays")
    st.write("Displays the moon surface with semi-transparent indicators (**green** = safe zone, **red** = hazard). "
             "The top 3 safest landing grid cells are highlighted with a white star.")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_rgb)
    
    # Overlay grid cells
    for r_idx in range(grid_n):
        for c_idx in range(grid_n):
            x_start = c_idx * cell_w
            y_start = r_idx * cell_h
            
            color = 'green' if safe_mask[r_idx, c_idx] else 'red'
            rect = patches.Rectangle(
                (x_start, y_start), cell_w, cell_h,
                linewidth=1, edgecolor=color, facecolor=color, alpha=0.22
            )
            ax.add_patch(rect)
            
    # Highlight top 3 safest cells
    for rank, (r, c) in enumerate(top_cells):
        cx = c * cell_w + cell_w / 2
        cy = r * cell_h + cell_h / 2
        ax.plot(cx, cy, 'w*', markersize=14, markeredgecolor='black', markeredgewidth=1)
        ax.text(cx, cy - cell_h * 0.3, f'#{rank+1}',
                color='white', fontsize=11, ha='center', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
                
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab_detections:
    st.subheader("Deep Learning YOLOv8 Crater Detections")
    st.write(f"Showing all **{total_craters}** craters detected with a confidence rating score ≥ **{conf_threshold:.2f}**.")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_rgb)
    
    for _, row in df.iterrows():
        rect = patches.Rectangle(
            (row['x1'], row['y1']),
            row['width_px'],
            row['height_px'],
            linewidth=1.2, edgecolor='yellow', facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)
        
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab_cdi:
    st.subheader("Crater Density Index (CDI)")
    st.write("Number of impact craters normalized per square kilometer (craters/km²). Highlighted using a thermal scale.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Gridded Area Visualization**")
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(image_rgb)
        for i in range(grid_n + 1):
            ax.axhline(y=i * cell_h, color='white', linewidth=0.5, alpha=0.4)
            ax.axvline(x=i * cell_w, color='white', linewidth=0.5, alpha=0.4)
        ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    with col2:
        st.write("**Crater Density Heatmap**")
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cdi_grid, cmap='hot', interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Craters per km²', color='white')
        ax.set_title("CDI Heatmap", fontsize=11)
        ax.set_xlabel("Col (West to East)")
        ax.set_ylabel("Row (North to South)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab_swhi:
    st.subheader("Size-Weighted Hazard Index (SWHI)")
    st.write("SWHI uses the natural logarithm of crater diameter to weight hazards. "
             "A cell containing larger craters has a higher SWHI score even if total count is small.")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(swhi_grid, cmap='hot', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label('Sum of ln(diameter_km) / cell area (km²)', color='white')
    ax.set_title("SWHI Heatmap", fontsize=11)
    ax.set_xlabel("Col (West to East)")
    ax.set_ylabel("Row (North to South)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab_safety_score:
    st.subheader("Combined Safety Score Heatmap")
    st.write("Normalized safety score (`0 = completely safe (green)` to `1 = highly hazardous (red)`), combining CDI and SWHI values.")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(safety_score, cmap='RdYlGn_r', interpolation='nearest', vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label('Combined Safety Index (0 to 1)', color='white')
    
    # Mark top 3 safest cells
    for rank, (r, c) in enumerate(top_cells):
        ax.plot(c, r, 'w*', markersize=12, markeredgecolor='black', markeredgewidth=0.7)
        ax.text(c, r - 0.35, f'#{rank+1}', color='white', fontsize=10, ha='center', fontweight='bold')
        
    ax.set_title("Combined Safety Index (CDI & SWHI Normalized)", fontsize=11)
    ax.set_xlabel("Col (West to East)")
    ax.set_ylabel("Row (North to South)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab_data_report:
    st.subheader("Lunar Region Scientific Assessment Report")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Top Landing Site Recommendations")
        recommendations = []
        for rank, (r, c) in enumerate(top_cells):
            score = safety_score[r, c]
            crater_count = int(count_grid[r, c])
            swhi_val = swhi_grid[r, c]
            cdi_val = cdi_grid[r, c]
            recommendations.append({
                "Rank": f"#{rank+1}",
                "Grid Coordinate": f"Row {r}, Col {c}",
                "Safety Score": f"{score:.4f}",
                "Crater Count": crater_count,
                "CDI (craters/km²)": f"{cdi_val:.2f}",
                "SWHI Index": f"{swhi_val:.2f}"
            })
        st.table(pd.DataFrame(recommendations))
        
    with col2:
        st.markdown("### Geological Statistics Summary")
        stats_data = {
            "Metric": [
                "Region Physical Width", 
                "Region Physical Height", 
                "Smallest Detected Crater", 
                "Largest Detected Crater", 
                "Mean Crater Diameter",
                "Crater Density in Safest Cell",
                "Crater Density in Hazard Cell"
            ],
            "Value": [
                f"{img_w / pixels_per_km:.2f} km",
                f"{img_h / pixels_per_km:.2f} km",
                f"{min_diameter:.4f} km ({min_diameter * 1000:.1f} m)" if total_craters > 0 else "N/A",
                f"{max_diameter:.4f} km ({max_diameter * 1000:.1f} m)" if total_craters > 0 else "N/A",
                f"{avg_diameter:.4f} km ({avg_diameter * 1000:.1f} m)" if total_craters > 0 else "N/A",
                f"{cdi_grid.min():.2f} craters/km²",
                f"{cdi_grid.max():.2f} craters/km²"
            ]
        }
        st.table(pd.DataFrame(stats_data))

    # Scientific Reflection (Answers to Section 12 dynamic reflections)
    st.markdown("---")
    st.markdown("### Scientific Reflection & Interpretation")
    
    # 1. Crater count density interpretation
    density_text = ""
    if total_craters > 150:
        density_text = "This indicates high crater density, consistent with an ancient lunar highland terrain that has experienced a heavy impact bombardment history."
    elif total_craters > 50:
        density_text = "This indicates moderate crater density, typical of transition boundaries between lunar maria (plains) and cratered highlands."
    else:
        density_text = "This indicates low crater density, which is representative of geologically younger smooth lunar maria plains."
        
    # 2. Safest zones text
    top_cell_desc = []
    for r, c in top_cells:
        pos_y = "Upper" if r < grid_n / 3 else ("Lower" if r > 2 * grid_n / 3 else "Central")
        pos_x = "Left" if c < grid_n / 3 else ("Right" if c > 2 * grid_n / 3 else "Middle")
        top_cell_desc.append(f"{pos_y}-{pos_x} (Row {r}, Col {c})")
    top_cell_desc_str = ", ".join(top_cell_desc)

    st.markdown(f"""
    1. **Crater Count & Density Assessment**: 
       * The model detected **{total_craters}** craters after Non-Maximum Suppression. {density_text}
    
    2. **Crater Size Range**: 
       * Crater diameters range from **{min_diameter:.3f} km ({min_diameter*1000:.1f} m)** to **{max_diameter:.3f} km ({max_diameter*1000:.1f} m)**, with an average of **{avg_diameter:.3f} km**. 
       * These are small to medium-scale secondary craters. In contrast to major craters like Tycho (~85 km), these features are typical of localized regional landing spot surveys.
    
    3. **Optimal Landing Zones**: 
       * The top 3 safest cells are situated at: **{top_cell_desc_str}**. 
       * These locations show minimal crater counts and hazards, representing flat landing candidates.
    
    4. **Crater Density (CDI) vs. Hazard Index (SWHI) Patterns**: 
       * The CDI and SWHI maps highlight complementary patterns. Some grid cells contain high numbers of small craters (elevated CDI but moderate SWHI), while others have fewer but much larger craters (low CDI but elevated SWHI). Using both metrics prevents landing in regions dominated by single large obstacles, which raw counts fail to represent.
    
    5. **System Limitations**:
       * *Shadows & Contrast*: Lighting angle significantly affects detections. Very shallow sun angles produce long shadows that may throw off the YOLO model (false positives) or mask craters (false negatives).
       * *Slope & Slope-Scale Hazards*: This 2D image analysis doesn't measure topography gradients or rock distribution smaller than the model's minimum detection size (~{min_diameter*1000:.1f} m). Incorporating digital elevation models (DEM) and higher resolution LIDAR readings would dramatically improve landing site validation.
    """)

# ---------------------------------------------------------
# Crater Table Display
# ---------------------------------------------------------
if len(df) > 0:
    st.markdown("### Detected Craters Catalog")
    st.dataframe(
        df[['center_x_km', 'center_y_km', 'diameter_km', 'confidence']].rename(columns={
            'center_x_km': 'X Center (km)',
            'center_y_km': 'Y Center (km)',
            'diameter_km': 'Diameter (km)',
            'confidence': 'Detection Confidence'
        }).sort_values('Diameter (km)', ascending=False),
        use_container_width=True
    )
