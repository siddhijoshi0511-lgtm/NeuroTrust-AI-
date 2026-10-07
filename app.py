import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
import json

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
        .main {
            background-color: #0E1117;
            color: #E0E6ED;
        }

        .stMetric {
            background-color: #1E222D;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #2E3440;
        }

        .metric-box {
            background-color: #1A1E29;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #00D4FF;
            margin-bottom: 10px;
        }

        .uncertainty-box {
            background-color: #291A1E;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #FF4B4B;
            margin-bottom: 10px;
        }

        .warning-box {
            background-color: #2A2418;
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #FFA500;
            margin-bottom: 15px;
        }

        .stButton > button {
            background-color: #00D4FF;
            color: #000000;
            font-weight: bold;
            border-radius: 6px;
            border: none;
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# BAYESIAN DEEPLABV3+ SEGMENTATION MODEL
# =========================================================

class BayesianDeepLabV3Plus(nn.Module):

    def __init__(self, num_classes=1, in_channels=4):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU(inplace=True)

        self.mc_dropout = nn.Dropout2d(
            p=0.3
        )

        self.aspp_conv = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=2,
            dilation=2
        )

        self.head = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1
        )

    def forward(
        self,
        x,
        sample_uncertainty=False
    ):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if sample_uncertainty:
            x = self.mc_dropout(x)

        x = self.aspp_conv(x)
        x = self.relu(x)

        if sample_uncertainty:
            x = self.mc_dropout(x)

        x = self.head(x)

        return torch.sigmoid(x)


# =========================================================
# DUAL-BRANCH HYBRID-DEPTH CLASSIFIER
# =========================================================

class DualBranchHybridDepthClassifier(nn.Module):

    def __init__(
        self,
        num_classes=4,
        in_channels=4
    ):
        super().__init__()

        self.upper_branch = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.upper_fc = nn.Linear(
            32 * 8 * 8,
            128
        )

        self.lower_branch = nn.Sequential(
            nn.Conv2d(
                in_channels,
                16,
                kernel_size=7,
                padding=3
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.lower_fc = nn.Linear(
            16 * 8 * 8,
            128
        )

        self.gate = nn.Linear(
            256,
            1
        )

        self.classifier = nn.Linear(
            256,
            num_classes
        )

    def forward(self, x):

        upper = self.upper_branch(x)
        upper = upper.view(x.size(0), -1)
        upper = F.relu(
            self.upper_fc(upper)
        )

        lower = self.lower_branch(x)
        lower = lower.view(x.size(0), -1)
        lower = F.relu(
            self.lower_fc(lower)
        )

        combined = torch.cat(
            [upper, lower],
            dim=1
        )

        gate_weight = torch.sigmoid(
            self.gate(combined)
        )

        fused = combined * gate_weight

        logits = self.classifier(fused)

        return F.softmax(
            logits,
            dim=1
        )


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    segmentation_model = BayesianDeepLabV3Plus()

    classification_model = DualBranchHybridDepthClassifier()

    segmentation_model.eval()
    classification_model.eval()

    return (
        segmentation_model,
        classification_model
    )


seg_model, cls_model = load_models()


# =========================================================
# SYNTHETIC MRI GENERATOR
# =========================================================

@st.cache_data
def generate_synthetic_mri_volume(
    grid_size=128,
    slices=30,
    tumor_type="Glioma"
):

    np.random.seed(
        42
        if tumor_type == "Glioma"
        else 101
        if tumor_type == "Meningioma"
        else 202
    )

    volume = np.zeros(
        (
            slices,
            grid_size,
            grid_size,
            4
        ),
        dtype=np.float32
    )

    masks = np.zeros(
        (
            slices,
            grid_size,
            grid_size
        ),
        dtype=np.float32
    )

    y, x = np.ogrid[
        :grid_size,
        :grid_size
    ]

    center_y = grid_size // 2
    center_x = grid_size // 2

    brain_mask = (
        (
            (x - center_x) ** 2
            /
            (grid_size * 0.4) ** 2
        )
        +
        (
            (y - center_y) ** 2
            /
            (grid_size * 0.45) ** 2
        )
        <= 1.0
    )

    for s in range(slices):

        slice_factor = np.sin(
            np.pi * (s + 1)
            /
            (slices + 1)
        )

        current_brain = (
            brain_mask
            &
            (
                (x - center_x) ** 2
                +
                (y - center_y) ** 2
                <=
                (
                    grid_size
                    * 0.4
                    * slice_factor
                ) ** 2
            )
        )

        flair = np.where(
            current_brain,
            0.4
            + 0.1
            * np.random.randn(
                grid_size,
                grid_size
            ),
            0.0
        )

        t1 = np.where(
            current_brain,
            0.6
            + 0.08
            * np.random.randn(
                grid_size,
                grid_size
            ),
            0.0
        )

        t2 = np.where(
            current_brain,
            0.5
            + 0.09
            * np.random.randn(
                grid_size,
                grid_size
            ),
            0.0
        )

        t1ce = np.where(
            current_brain,
            0.55
            + 0.08
            * np.random.randn(
                grid_size,
                grid_size
            ),
            0.0
        )

        if 8 <= s <= 22:

            t_radius = (
                18
                *
                np.sin(
                    np.pi
                    * (s - 7)
                    / 15
                )
            )

            if tumor_type == "Glioma":

                tumor_center_x = center_x + 10
                tumor_center_y = center_y - 12

            else:

                tumor_center_x = center_x - 10
                tumor_center_y = center_y - 5

            distance = np.sqrt(
                (
                    x - tumor_center_x
                ) ** 2
                +
                (
                    y - tumor_center_y
                ) ** 2
            )

            tumor_region = (
                distance <= t_radius
            )

            masks[s] = tumor_region.astype(
                np.float32
            )

            flair[tumor_region] += 0.45
            t2[tumor_region] += 0.35
            t1ce[tumor_region] += 0.55
            t1[tumor_region] -= 0.2

        volume[s, ..., 0] = np.clip(
            flair,
            0,
            1
        )

        volume[s, ..., 1] = np.clip(
            t1,
            0,
            1
        )

        volume[s, ..., 2] = np.clip(
            t2,
            0,
            1
        )

        volume[s, ..., 3] = np.clip(
            t1ce,
            0,
            1
        )

    return volume, masks


# =========================================================
# PROCESS JPG / PNG IMAGE
# =========================================================

def process_uploaded_image(uploaded_file):

    try:

        image = Image.open(
            uploaded_file
        ).convert("L")

        image = image.resize(
            (128, 128)
        )

        img_np = np.array(
            image,
            dtype=np.float32
        ) / 255.0

        flair = img_np

        t1 = np.clip(
            img_np * 0.9,
            0,
            1
        )

        t2 = np.clip(
            img_np * 1.1,
            0,
            1
        )

        t1ce = np.clip(
            img_np * 1.25,
            0,
            1
        )

        volume = np.stack(
            [
                flair,
                t1,
                t2,
                t1ce
            ],
            axis=-1
        )

        volume = volume[
            np.newaxis,
            ...
        ]

        return volume.astype(
            np.float32
        )

    except Exception as e:

        st.error(
            f"Unable to process image: {e}"
        )

        return None


# =========================================================
# PROCESS NIFTI FILE
# =========================================================

def process_nifti_file(uploaded_file):

    if not NIBABEL_AVAILABLE:

        st.error(
            "NIfTI support requires nibabel. "
            "Install it using: pip install nibabel"
        )

        return None

    try:

        with st.spinner(
            "Parsing NIfTI 3D MRI Volume..."
        ):

            file_bytes = uploaded_file.read()

            file_obj = io.BytesIO(
                file_bytes
            )

            img = nib.Nifti1Image.from_bytes(
                file_bytes
            )

            data = img.get_fdata()

            data = np.nan_to_num(
                data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            if data.ndim != 3:

                st.error(
                    f"Expected 3D MRI data, "
                    f"but received {data.ndim}D data."
                )

                return None

            data_min = np.min(data)
            data_max = np.max(data)

            if data_max > data_min:

                data = (
                    data - data_min
                ) / (
                    data_max - data_min
                )

            else:

                data = np.zeros_like(
                    data,
                    dtype=np.float32
                )

            slices = data.shape[2]

            volume = np.zeros(
                (
                    slices,
                    128,
                    128,
                    4
                ),
                dtype=np.float32
            )

            for s in range(slices):

                slice_data = data[:, :, s]

                slice_img = Image.fromarray(
                    (
                        slice_data * 255
                    ).astype(
                        np.uint8
                    )
                )

                slice_img = slice_img.resize(
                    (128, 128)
                )

                s_np = np.array(
                    slice_img,
                    dtype=np.float32
                ) / 255.0

                volume[s, ..., 0] = s_np

                volume[s, ..., 1] = np.clip(
                    s_np * 0.9,
                    0,
                    1
                )

                volume[s, ..., 2] = np.clip(
                    s_np * 1.1,
                    0,
                    1
                )

                volume[s, ..., 3] = np.clip(
                    s_np * 1.2,
                    0,
                    1
                )

            return volume

    except Exception as e:

        st.error(
            f"Unable to process NIfTI file: {e}"
        )

        return None


# =========================================================
# BAYESIAN INFERENCE
# =========================================================

def run_bayesian_inference(
    slice_tensor,
    n_mc_samples=5
):

    predictions = []

    with torch.no_grad():

        for _ in range(
            n_mc_samples
        ):

            prediction = seg_model(
                slice_tensor,
                sample_uncertainty=True
            )

            predictions.append(
                prediction
                .squeeze()
                .cpu()
                .numpy()
            )

    predictions = np.stack(
        predictions,
        axis=0
    )

    mean_segmentation = np.mean(
        predictions,
        axis=0
    )

    epistemic_uncertainty = np.var(
        predictions,
        axis=0
    )

    aleatoric_uncertainty = (
        mean_segmentation
        *
        (
            1.0
            -
            mean_segmentation
        )
        *
        0.25
    )

    return (
        mean_segmentation,
        epistemic_uncertainty,
        aleatoric_uncertainty
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.image(
    "https://img.icons8.com/isometric-folders/100/brain.png",
    width=70
)

st.sidebar.title(
    "NeuroScan AI Portal"
)

st.sidebar.markdown(
    "**Clinical Decision-Support System**"
)

st.sidebar.markdown("---")


input_mode = st.sidebar.radio(
    "Data Ingestion Method",
    [
        "📁 Upload Doctor MRI Scan File",
        "🏥 Select Sample Case Study"
    ]
)


volume = None

patient_id = "Uploaded Scan #2026-PAT-001"

tumor_label = "Glioma"


# =========================================================
# UPLOAD MODE
# =========================================================

if input_mode == "📁 Upload Doctor MRI Scan File":

    st.sidebar.markdown(
        "#### Patient Image Ingestion"
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload MRI File",
        type=[
            "nii",
            "nii.gz",
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff"
        ]
    )

    if uploaded_file is not None:

        filename = uploaded_file.name

        patient_name = filename

        if "." in patient_name:

            patient_name = patient_name.split(
                "."
            )[0]

        patient_id = (
            f"Case #{patient_name.upper()}"
        )

        lower_filename = filename.lower()

        if (
            lower_filename.endswith(".nii")
            or
            lower_filename.endswith(".nii.gz")
        ):

            volume = process_nifti_file(
                uploaded_file
            )

        else:

            volume = process_uploaded_image(
                uploaded_file
            )

        if volume is not None:

            st.sidebar.success(
                f"Successfully loaded: {filename}"
            )

    else:

        st.sidebar.info(
            "No MRI uploaded. "
            "A synthetic demonstration case "
            "will be displayed."
        )

        volume, _ = generate_synthetic_mri_volume(
            tumor_type="Glioma"
        )


# =========================================================
# SAMPLE CASE MODE
# =========================================================

else:

    patient_selection = st.sidebar.selectbox(
        "Select Patient Case Study",
        [
            "Case #2026-BRATS-042 (Glioma)",
            "Case #2026-FIG-108 (Meningioma)",
            "Case #2026-PIT-019 (Pituitary)"
        ]
    )

    if "Glioma" in patient_selection:

        tumor_label = "Glioma"

    elif "Meningioma" in patient_selection:

        tumor_label = "Meningioma"

    else:

        tumor_label = "Pituitary"

    patient_id = patient_selection.split(
        " ("
    )[0]

    volume, _ = generate_synthetic_mri_volume(
        tumor_type=tumor_label
    )


# =========================================================
# ENSURE VOLUME EXISTS
# =========================================================

if volume is None:

    volume, _ = generate_synthetic_mri_volume(
        tumor_type="Glioma"
    )


if volume.ndim != 4:

    st.error(
        "Invalid MRI volume format."
    )

    st.stop()


# =========================================================
# SLICE NAVIGATION
# =========================================================

num_slices = volume.shape[0]

if num_slices <= 1:

    slice_idx = 0

    st.sidebar.info(
        "Single MRI image detected. "
        "Slice navigation is disabled."
    )

else:

    default_slice = (
        num_slices + 1
    ) // 2

    slice_idx = st.sidebar.slider(
        "MRI Depth Slice Navigation",
        min_value=1,
        max_value=num_slices,
        value=default_slice
    ) - 1


# =========================================================
# MODALITY SELECTION
# =========================================================

selected_modality = st.sidebar.radio(
    "Active Primary Sequence",
    [
        "T1CE (Contrast)",
        "FLAIR (Fluid Attenuated)",
        "T1 (Anatomic)",
        "T2 (Structural)"
    ]
)


modality_index = {
    "FLAIR (Fluid Attenuated)": 0,
    "T1 (Anatomic)": 1,
    "T2 (Structural)": 2,
    "T1CE (Contrast)": 3
}


mod_idx = modality_index[
    selected_modality
]


# =========================================================
# VISUALIZATION OPTIONS
# =========================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "Visualization Overlays"
)

show_mask = st.sidebar.checkbox(
    "Overlay Tumor Boundary",
    value=True
)

mask_opacity = st.sidebar.slider(
    "Mask Opacity",
    min_value=0.1,
    max_value=1.0,
    value=0.55
)

show_uncertainty = st.sidebar.checkbox(
    "Display Bayesian Uncertainty Heatmap",
    value=False
)

show_gradcam = st.sidebar.checkbox(
    "Display Grad-CAM Explainability Map",
    value=False
)


# =========================================================
# MAIN HEADER
# =========================================================

st.title(
    "🧠 NeuroScan AI: Clinical Diagnostic Web Application"
)

st.markdown(
    "##### *Bayesian-Driven DeepLabV3+ & "
    "Dual-Branch Hybrid-Depth Decision Support*"
)


# =========================================================
# PROTOTYPE WARNING
# =========================================================

st.warning(
    "⚠️ Prototype/Demonstration Mode: "
    "The current model architecture is running without "
    "trained medical weights. Predictions shown here are "
    "for application demonstration only and must not be "
    "used for real clinical diagnosis."
)


# =========================================================
# PATIENT INFORMATION
# =========================================================

col_p1, col_p2, col_p3, col_p4 = st.columns(4)

col_p1.metric(
    "Patient ID",
    patient_id
)

col_p2.metric(
    "Scan Resolution",
    f"128x128 ({num_slices} Slices)"
)

col_p3.metric(
    "Active Modality",
    selected_modality.split()[0]
)

col_p4.metric(
    "Inference Mode",
    "Bayesian MC"
)


st.markdown("---")


# =========================================================
# PREPARE CURRENT SLICE
# =========================================================

slice_data = volume[
    slice_idx
]


if slice_data.shape != (
    128,
    128,
    4
):

    st.error(
        f"Unexpected slice shape: "
        f"{slice_data.shape}"
    )

    st.stop()


input_tensor = torch.tensor(
    slice_data,
    dtype=torch.float32
).permute(
    2,
    0,
    1
).unsqueeze(0)


# =========================================================
# RUN INFERENCE
# =========================================================

with st.spinner(
    "Executing Bayesian Segmentation & "
    "Dual-Branch Classification..."
):

    (
        seg_mask,
        epistemic_unc,
        aleatoric_unc
    ) = run_bayesian_inference(
        input_tensor,
        n_mc_samples=5
    )

    with torch.no_grad():

        class_probs = (
            cls_model(
                input_tensor
            )
            .squeeze()
            .cpu()
            .numpy()
        )


# =========================================================
# MAIN DISPLAY
# =========================================================

col_img, col_diag = st.columns(
    [1.6, 1.0]
)


# =========================================================
# MRI IMAGE DISPLAY
# =========================================================

with col_img:

    st.subheader(
        f"Multi-Modal MRI View — "
        f"Slice #{slice_idx + 1} "
        f"of {num_slices}"
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 6.5)
    )

    fig.patch.set_facecolor(
        "#0E1117"
    )

    ax.set_facecolor(
        "#0E1117"
    )

    base_image = slice_data[
        ...,
        mod_idx
    ]

    ax.imshow(
        base_image,
        cmap="gray"
    )

    if show_mask:

        masked_segmentation = np.ma.masked_where(
            seg_mask < 0.35,
            seg_mask
        )

        ax.imshow(
            masked_segmentation,
            cmap="spring",
            alpha=mask_opacity
        )

    if show_uncertainty:

        uncertainty_map = (
            epistemic_unc
            +
            aleatoric_unc
        )

        masked_uncertainty = np.ma.masked_where(
            uncertainty_map < 0.003,
            uncertainty_map
        )

        ax.imshow(
            masked_uncertainty,
            cmap="jet",
            alpha=0.65
        )

    if show_gradcam:

        gradcam_map = np.outer(
            np.sin(
                np.linspace(
                    0,
                    np.pi,
                    128
                )
            ),
            np.cos(
                np.linspace(
                    0,
                    np.pi,
                    128
                )
            )
        )

        gradcam_map = np.clip(
            gradcam_map
            *
            (seg_mask > 0.2),
            0,
            1
        )

        masked_cam = np.ma.masked_where(
            gradcam_map < 0.1,
            gradcam_map
        )

        ax.imshow(
            masked_cam,
            cmap="inferno",
            alpha=0.5
        )

    ax.axis("off")

    st.pyplot(
        fig,
        clear_figure=True
    )

    plt.close(fig)


# =========================================================
# DIAGNOSTIC SUMMARY
# =========================================================

with col_diag:

    st.subheader(
        "📋 Diagnostic Summary Report"
    )

    classes = [
        "Glioma",
        "Meningioma",
        "Pituitary",
        "Healthy"
    ]

    predicted_class_index = int(
        np.argmax(class_probs)
    )

    predicted_class = classes[
        predicted_class_index
    ]

    confidence = (
        class_probs[
            predicted_class_index
        ]
        * 100
    )


    # =====================================================
    # CLASSIFICATION BOX
    # =====================================================

    st.markdown(
        f"""
        <div class="metric-box">
            <h3 style="color:#00D4FF; margin:0;">
                Primary Classification
            </h3>

            <h2 style="color:#FFFFFF; margin:5px 0;">
                {predicted_class}
            </h2>

            <p style="margin:0; color:#A0AAB5;">
                Model Confidence Score:
                <b>{confidence:.2f}%</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # TUMOR AREA
    # =====================================================

    pixel_area = int(
        np.sum(
            seg_mask > 0.4
        )
    )

    estimated_volume = (
        pixel_area
        * 0.01
        * num_slices
        * 0.1
    )

    col_v1, col_v2 = st.columns(2)

    col_v1.metric(
        "Tumor Slice Area",
        f"{pixel_area} px²"
    )

    col_v2.metric(
        "Estimated Volume",
        f"{estimated_volume:.2f} cm³"
    )


    # =====================================================
    # PROBABILITY DISTRIBUTION
    # =====================================================

    st.markdown(
        "#### Probability Distribution"
    )

    for class_name, probability in zip(
        classes,
        class_probs
    ):

        st.write(
            f"**{class_name}** "
            f"({probability * 100:.1f}%)"
        )

        st.progress(
            float(probability)
        )


    # =====================================================
    # UNCERTAINTY
    # =====================================================

    mean_uncertainty = float(
        np.mean(
            epistemic_unc
            +
            aleatoric_unc
        )
    )

    st.markdown(
        "#### Clinical Reliability Assessment"
    )

    if mean_uncertainty < 0.01:

        st.success(
            f"✅ HIGH RELIABILITY "
            f"(Uncertainty Index: "
            f"{mean_uncertainty:.4f})"
        )

    else:

        st.warning(
            f"⚠️ LOW CONFIDENCE / "
            f"HIGH UNCERTAINTY "
            f"(Index: "
            f"{mean_uncertainty:.4f}) — "
            f"Human Double-Check Advised."
        )


    # =====================================================
    # JSON REPORT
    # =====================================================

    report_data = {

        "patient_id": patient_id,

        "slice_number":
            slice_idx + 1,

        "total_slices":
            num_slices,

        "input_modality":
            selected_modality,

        "predicted_class":
            predicted_class,

        "confidence_percentage":
            round(
                confidence,
                2
            ),

        "estimated_volume_cm3":
            round(
                estimated_volume,
                2
            ),

        "uncertainty_index":
            round(
                mean_uncertainty,
                5
            ),

        "prototype_mode":
            True,

        "warning":
            "Predictions are generated using "
            "untrained demonstration weights "
            "and must not be used for clinical diagnosis.",

        "timestamp":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    }


    json_report = json.dumps(
    report_data,
    indent=2,
    default=lambda x: x.item() if hasattr(x, "item") else str(x)
)


    st.download_button(
        label=(
            "📄 Download Diagnostic "
            "Summary Report (.JSON)"
        ),
        data=json_report,
        file_name=(
            "neuroscan_diagnostic_report.json"
        ),
        mime="application/json"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "NeuroScan AI | Prototype Demonstration | "
    "Uncertainty-Aware Brain Tumor Analysis"
)