import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64

# --------------------------
# CONFIG
# --------------------------
MODEL_PATH = "best_resnet18_model_new.pth"
CLASS_NAMES = ["STAGE 1", "STAGE 2", "STAGE 3", "STAGE 4", "STAGE 5", "STAGE 6"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the reference UI image included in the container (optional, used for demo banner)
BANNER_IMAGE_PATH = r"/mnt/data/WhatsApp Image 2025-12-06 at 17.10.31_9327fa88.jpg"

# --------------------------
# MODEL LOADING
# --------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception as e:
        st.warning(f"Could not load model weights from {MODEL_PATH}. Running with random weights. ({e})")

    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform

model, transform = load_model()

# --------------------------
# STYLES (to mimic the screenshot)
# --------------------------
st.set_page_config(page_title="CVMI Bone Maturity Stage Classification", layout="wide")

st.markdown(
    """
    <style>
    .banner{
        background: linear-gradient(90deg,#0f6674,#2fa3a0);
        color: white;
        padding: 40px 30px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .title{font-size:34px; font-weight:700; margin:0}
    .subtitle{font-size:15px; opacity:0.95}
    .card{background:white; border-radius:12px; padding:18px; box-shadow: 0 4px 20px rgba(0,0,0,0.08);}
    .small-muted{color:#6b7280; font-size:13px}
    .stage-pill{display:inline-block; padding:8px 14px; border-radius:8px; background:#e6f7f6; margin-right:6px}
    .stage-pill.active{background:#0f6674; color:white; font-weight:700}
    .progress-bar{height:12px; background:#e6f3f3; border-radius:8px;}
    .progress-fill{height:12px; background:#0f6674; border-radius:8px}
    </style>
    """,
    unsafe_allow_html=True,
)

# Banner
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"<div class='banner'><div class='title'>CVMI Bone Maturity Stage Classification</div><div class='subtitle'>Automated X-ray analysis for precise growth stage prediction (Stages 1-6)</div></div>", unsafe_allow_html=True)
with col2:
    # optional small logo or user icon
    st.write("")

# Layout: left = image card, right = diagnostic card
left_col, right_col = st.columns([1, 1.05])

with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h4>Patient Image</h4>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"], key='uploader')
    sample_shown = False
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)
    else:
        # if user didn't upload, show the provided example image (if exists)
        try:
            sample = Image.open(BANNER_IMAGE_PATH).convert("RGB")
            st.image(sample, caption="Example X-Ray (sample)", use_column_width=True)
            sample_shown = True
        except Exception:
            st.info("Upload an X-ray image to get started.")

    st.markdown("<div class='small-muted' style='margin-top:10px'>Uploaded X-Ray</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h4>Diagnostic Result</h4>", unsafe_allow_html=True)

    # Placeholder for prediction UI
    # If predict pressed, populate these values
    predicted_label = None
    confidence = None
    desc = None

    # Predict button in right card pulls image from uploader or sample
    if st.button("Predict Stage"):
        if uploaded_file is None and not sample_shown:
            st.warning("Please upload an X-ray image first.")
        else:
            # choose the image object
            img_for_pred = image if uploaded_file is not None else sample
            img_tensor = transform(img_for_pred).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_idx = int(torch.argmax(probs))
                predicted_label = CLASS_NAMES[pred_idx]
                confidence = float(probs[pred_idx].cpu().item()) * 100.0

            # Simple heuristic description mapping (customize as needed)
            descriptions = {
                0: "Inferior borders of C2, C3, and C4 are flat. Significant growth potential remains.",
                1: "Some change in inferior borders; growth potential present.",
                2: "Further maturation, partial fusion observed.",
                3: "Near fusion, reduced growth potential.",
                4: "Almost complete fusion; little growth remains.",
                5: "Fully matured; fusion complete.",
            }
            desc = descriptions.get(pred_idx, "")

    # If we already have prediction from previous run (session state), load it
    if "last_pred" in st.session_state:
        predicted_label = st.session_state.get("last_pred")
        confidence = st.session_state.get("last_conf")
        desc = st.session_state.get("last_desc")

    # Display stage steps
    step_cols = st.columns(6)
    active_idx = None
    if predicted_label is not None:
        active_idx = CLASS_NAMES.index(predicted_label)
        # store in session state for reuse
        st.session_state["last_pred"] = predicted_label
        st.session_state["last_conf"] = confidence
        st.session_state["last_desc"] = desc

    for i in range(6):
        cls = f"<div class='stage-pill {'active' if active_idx==i else ''}'>{i+1}</div>"
        step_cols[i].markdown(cls, unsafe_allow_html=True)

    # Predicted Stage text
    if predicted_label is not None:
        st.markdown(f"<h2 style='margin-top:12px'>Predicted Stage: <strong>{predicted_label}</strong></h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='margin-top:12px; color:#374151;'>Predicted Stage: --</h3>", unsafe_allow_html=True)

    # Description box
    st.markdown("<div style='border:1px solid #e6eef0; padding:12px; border-radius:8px; margin-top:10px'>", unsafe_allow_html=True)
    st.markdown(f"<strong>Description:</strong> {desc if desc else 'No description available yet.'}", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Confidence bar
    if confidence is not None:
        conf_frac = max(0.0, min(confidence / 100.0, 1.0))
        st.markdown("<div style='margin-top:12px'><strong>Confidence:</strong></div>", unsafe_allow_html=True)
        st.markdown("<div class='progress-bar'><div class='progress-fill' style='width:" + f"{conf_frac*100:.2f}%" + "'></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:6px'>{confidence:.1f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='margin-top:12px'><strong>Confidence:</strong> --</div>", unsafe_allow_html=True)

    # Buttons: Download Report and New Analysis
    col_dl, col_new = st.columns([1, 1])

    with col_dl:
        if predicted_label is not None:
            report_text = f"Predicted Stage: {predicted_label}\nConfidence: {confidence:.2f}%\nDescription: {desc}\n"
            st.download_button("Download Report", data=report_text.encode('utf-8'), file_name="cvmi_report.txt")
        else:
            st.button("Download Report", disabled=True)

    with col_new:
        if st.button("New Analysis"):
            # clear uploader and session state
            if 'uploader' in st.session_state:
                try:
                    del st.session_state['uploader']
                except Exception:
                    pass
            for k in ["last_pred", "last_conf", "last_desc"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# Footer small note
st.markdown("<div style='margin-top:18px; color:#6b7280; font-size:13px'>Tip: Upload a lateral cervical X-ray similar to the example for best results.</div>", unsafe_allow_html=True)
