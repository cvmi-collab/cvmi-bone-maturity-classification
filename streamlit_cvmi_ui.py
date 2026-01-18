import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="CVMI Bone Maturity Stage Classification",
    layout="wide"
)

# --------------------------
# PATH SETUP
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_resnet18_model_new.pth")

COLLEGE_LOGO = os.path.join(BASE_DIR, "assets", "college_logo.png.png")
DEPT_LOGO = os.path.join(BASE_DIR, "assets", "department_logo.png.png")
REFERENCE_IMAGE = os.path.join(BASE_DIR, "assets", "cvm_reference.png")

CLASS_NAMES = [
    "STAGE 1", "STAGE 2", "STAGE 3",
    "STAGE 4", "STAGE 5", "STAGE 6"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# MODEL LOADING
# --------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return model, transform


model, transform = load_model()

# --------------------------
# STYLES
# --------------------------
st.markdown(
    """
    <style>
    .card{
        background:white;
        border-radius:12px;
        padding:18px;
        box-shadow:0 4px 20px rgba(0,0,0,0.08);
    }
    .stage-pill{
        display:inline-block;
        padding:8px 14px;
        border-radius:8px;
        background:#e6f7f6;
        margin-right:6px;
    }
    .stage-pill.active{
        background:#0f6674;
        color:white;
        font-weight:700;
    }
    a{
        color:#2563eb;
        text-decoration:none;
        font-weight:600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------
# HEADER (LOGOS + DEPARTMENT)
# --------------------------
c1, c2, c3 = st.columns([1, 2, 1])

# --------------------------
# TITLE / TEXT (NO BANNER)
# --------------------------
st.markdown(
    """
    <h2 style="margin-bottom:6px;">
        CVMI Bone Maturity Stage Classification
    </h2>

    <p style="font-size:16px; margin-top:0;">
        <strong>CERVICAL VERTEBRAL METHOD (CVM) STAGING</strong><br>
        <strong>CERVICAL STAGE (CS):</strong> 1, 2, 3, 4, 5, 6
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

with c1:
    if os.path.exists(COLLEGE_LOGO):
        st.image(COLLEGE_LOGO, width=110)

with c2:
    st.markdown(
        "<h3 style='text-align:center;'>Department of Orthodontics & Dentofacial Orthopedics</h3>",
        unsafe_allow_html=True
    )

with c3:
    if os.path.exists(DEPT_LOGO):
        st.image(DEPT_LOGO, width=110)

# --------------------------
# MAIN LAYOUT
# --------------------------
left_col, right_col = st.columns([1, 1.05])

# -------- LEFT PANEL --------
with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Patient Image")

    uploaded_file = st.file_uploader(
        "Upload an image (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)
    else:
        st.info("Upload an X-ray image to get started.")

        if os.path.exists(REFERENCE_IMAGE):
            st.markdown(
                "<a href='assets/cvm_reference.png' target='_blank'>Link</a>",
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

# -------- RIGHT PANEL --------
with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Diagnostic Result")

    predicted_label = None
    desc = None

    if st.button("Predict Stage"):
        if uploaded_file is None:
            st.warning("Please upload an X-ray image first.")
        else:
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(img_tensor)
                pred_idx = int(torch.argmax(outputs))

            predicted_label = CLASS_NAMES[pred_idx]

            descriptions = {
                0: "Inferior borders of C2, C3, and C4 are flat. Significant growth potential remains.",
                1: "Concavity appears at C2. Growth potential present.",
                2: "Concavity seen in C2 and C3. Active growth phase.",
                3: "Concavity in C2–C4. Growth spurt nearing completion.",
                4: "Almost complete fusion. Minimal growth remains.",
                5: "Complete skeletal maturation."
            }

            desc = descriptions[pred_idx]

            st.session_state["last_pred"] = predicted_label
            st.session_state["last_desc"] = desc

    if "last_pred" in st.session_state:
        predicted_label = st.session_state["last_pred"]
        desc = st.session_state["last_desc"]

    cols = st.columns(6)

    if predicted_label:
        active_idx = CLASS_NAMES.index(predicted_label)
        for i in range(6):
            cls = "stage-pill active" if i == active_idx else "stage-pill"
            cols[i].markdown(
                f"<div class='{cls}'>{i + 1}</div>",
                unsafe_allow_html=True
            )
        st.markdown(f"### Predicted Stage: **{predicted_label}**")
    else:
        st.markdown("### Predicted Stage: --")

    st.markdown(
        f"**Description:** {desc if desc else 'No description available yet.'}"
    )

    st.markdown(
        "<div style='margin-top:10px;color:#6b7280;font-size:13px;'>"
        "Total 1000 digit cephalogram were used to train the AI Model"
        "</div>",
        unsafe_allow_html=True
    )

    if st.button("New Analysis"):
        st.session_state.clear()
        st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# FOOTER
# --------------------------
st.markdown(
    "<div style='margin-top:18px;color:#6b7280;font-size:13px;'>"
    "Tip: Upload a lateral cervical X-ray similar to the example for best results."
    "</div>",
    unsafe_allow_html=True
)
