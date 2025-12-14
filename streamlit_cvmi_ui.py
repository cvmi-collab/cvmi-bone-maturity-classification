from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64

# ---- IMPORTANT: set_page_config MUST be the first Streamlit command ----
st.set_page_config(page_title="CVMI Bone Maturity Stage Classification", layout="wide")

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
        # warn only when UI is already initialized (safe now because set_page_config was done)
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
    st.write("")

# Layout: left = image card, right = diagnostic card
left_col, right_col = st.columns([1, 1.05])

with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h4>Patient Image</h4>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"], key='uploader')
    sample_shown = False
    image = None
    sample = None

    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)
    else:
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

    predicted_label = None
    confidence = None
    desc = None

    if st.button("Predict Stage"):
        if uploaded_file is None and not sample_shown:
            st.warning("Please upload an X-ray image first.")
        else:
            img_for_pred = image if uploaded_file is not None else sample
            img_tensor = transform(img_for_pred).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_idx = int(torch.argmax(probs))
                predicted_label = CLASS_NAMES[pred_idx]
                confidence = float(probs[pred_idx].cpu().item()) * 100.0

            descriptions = {
                0: "Inferior borders of C2, C3, and C4 are flat. Significant growth potential remains.",
                1: "Some change in inferior borders; growth potential present.",
                2: "Further maturation, partial fusion observed.",
                3: "Near fusion, reduced growth potential.",
                4: "Almost complete fusion; little growth remains.",
                5: "Fully matured; fusion complete.",
            }
            desc = descriptions.get(pred_idx, "")

            # store useful things in session for later use (download, page refresh)
            st.session_state["last_pred"] = predicted_label
            st.session_state["last_conf"] = confidence
            st.session_state["last_desc"] = desc
            # store the image bytes for later PDF creation
            buf = io.BytesIO()
            img_for_pred.save(buf, format="PNG")
            st.session_state["last_img_bytes"] = buf.getvalue()

    # load previous prediction if present
    if "last_pred" in st.session_state and predicted_label is None:
        predicted_label = st.session_state.get("last_pred")
        confidence = st.session_state.get("last_conf")
        desc = st.session_state.get("last_desc")

    step_cols = st.columns(6)
    active_idx = None
    if predicted_label is not None:
        active_idx = CLASS_NAMES.index(predicted_label)

    for i in range(6):
        cls_html = f"<div class='stage-pill {'active' if active_idx==i else ''}'>{i+1}</div>"
        step_cols[i].markdown(cls_html, unsafe_allow_html=True)

    if predicted_label is not None:
        st.markdown(f"<h2 style='margin-top:12px'>Predicted Stage: <strong>{predicted_label}</strong></h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='margin-top:12px; color:#374151;'>Predicted Stage: --</h3>", unsafe_allow_html=True)

    st.markdown("<div style='border:1px solid #e6eef0; padding:12px; border-radius:8px; margin-top:10px'>", unsafe_allow_html=True)
    st.markdown(f"<strong>Description:</strong> {desc if desc else 'No description available yet.'}", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if confidence is not None:
        conf_frac = max(0.0, min(confidence / 100.0, 1.0))
        st.markdown("<div style='margin-top:12px'><strong>Confidence:</strong></div>", unsafe_allow_html=True)
        st.markdown("<div class='progress-bar'><div class='progress-fill' style='width:" + f"{conf_frac*100:.2f}%" + "'></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:6px'>{confidence:.1f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='margin-top:12px'><strong>Confidence:</strong> --</div>", unsafe_allow_html=True)

    col_dl, col_new = st.columns([1, 1])

    # ---------- PDF download implementation ----------
    with col_dl:
        if predicted_label is not None:
            # Build PDF with ReportLab, embedding the patient image if available
            try:
                # Temporary files
                temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                temp_img = None

                # If we saved image bytes in session state, create a temp image file
                if "last_img_bytes" in st.session_state:
                    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    with open(temp_img.name, "wb") as f:
                        f.write(st.session_state["last_img_bytes"])

                # Create PDF
                c = canvas.Canvas(temp_pdf.name, pagesize=letter)
                width, height = letter

                # Title
                c.setFont("Helvetica-Bold", 16)
                c.drawString(40, height - 50, "CVMI Bone Maturity Stage Report")

                # Draw horizontal rule
                c.setLineWidth(0.5)
                c.line(40, height - 56, width - 40, height - 56)

                # If image exists, draw it on the top-right area
                if temp_img is not None:
                    # Reserve area and maintain aspect ratio
                    img_box_w = 200
                    img_box_h = 200
                    # position it top-right
                    img_x = width - img_box_w - 40
                    img_y = height - img_box_h - 80
                    try:
                        c.drawImage(temp_img.name, img_x, img_y, width=img_box_w, height=img_box_h, preserveAspectRatio=True, mask='auto')
                    except Exception:
                        pass

                # Text body
                c.setFont("Helvetica", 11)
                y = height - 100
                line_height = 16

                c.drawString(40, y, f"Predicted Stage : {predicted_label}")
                y -= line_height
                c.drawString(40, y, f"Confidence      : {confidence:.2f}%")
                y -= line_height * 1.2

                c.drawString(40, y, "Description:")
                y -= line_height
                # wrap description text manually
                desc_text = desc if desc else "No description provided."
                max_chars = 90
                while desc_text:
                    part = desc_text[:max_chars]
                    c.drawString(50, y, part)
                    desc_text = desc_text[max_chars:]
                    y -= line_height
                    if y < 80:
                        c.showPage()
                        y = height - 80
                        c.setFont("Helvetica", 11)

                # Footer / note
                if y < 120:
                    c.showPage()
                    y = height - 80
                c.setFont("Helvetica-Oblique", 9)
                c.drawString(40, 50, "Generated by CVMI Bone Maturity Stage Classification")

                c.save()

                # Read bytes and present download button
                with open(temp_pdf.name, "rb") as f:
                    pdf_bytes = f.read()

                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name="cvmi_report.pdf",
                    mime="application/pdf"
                )

            finally:
                # clean up temp files (do not remove before download_button reads bytes; we already read them)
                if temp_img is not None:
                    try:
                        os.unlink(temp_img.name)
                    except Exception:
                        pass
                if temp_pdf is not None:
                    try:
                        os.unlink(temp_pdf.name)
                    except Exception:
                        pass
        else:
            st.button("Download PDF Report", disabled=True)

    # ---------- New Analysis ----------
    with col_new:
        if st.button("New Analysis"):
            # clear uploader and session state
            if 'uploader' in st.session_state:
                try:
                    del st.session_state['uploader']
                except Exception:
                    pass
            for k in ["last_pred", "last_conf", "last_desc", "last_img_bytes"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='margin-top:18px; color:#6b7280; font-size:13px'>Tip: Upload a lateral cervical X-ray similar to the example for best results.</div>", unsafe_allow_html=True)
