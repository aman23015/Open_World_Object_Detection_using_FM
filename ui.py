import streamlit as st
from PIL import Image, ImageDraw
import torch
import numpy as np
from FOMO import build_model
from torchvision import transforms
from types import SimpleNamespace

@st.cache_resource
def load_model():
    from types import SimpleNamespace

    args = SimpleNamespace(
        model_name="google/owlvit-base-patch16",
        dataset="Aerial",
        device="cuda:1",
        data_task="RWD",
        data_root="/home/aaditya23006/AMAN/SML/DATA",
        classnames_file="known_classnames.txt",
        prev_classnames_file="known_classnames.txt",
        unknown_classnames_file="",
        templates_file="best_templates.txt",
        attributes_file="attributes.json",
        output_dir="tmp/rwod",
        pred_per_im=100,
        image_resize=768,
        num_few_shot=100,
        num_att_per_class=25,
        image_conditioned=True,
        image_conditioned_file="few_shot_data.json",
        use_attributes=True,
        att_selection=False,
        att_refinement=False,
        att_adapt=True,
        post_process_method="regular",
        unk_proposal=False,
        unk_method="sigmoid-max-mcm",
        batch_size=4,
        num_workers=2,
        train_set="",        # Needed to bypass loading dataset
        test_set="",         # Same here
        seed=42,
        eval=False,
        viz=False,
        prev_output_file="",
        output_file="",
        TCP="29500",
        distributed=False,
        neg_sup_ep=30,      # number of epochs for attribute refinement/selection
        neg_sup_lr=5e-5,    # learning rate used during attribute training

    )

    
    model, post = build_model(args)
    model.load_state_dict(torch.load(f"{args.output_dir}/best_model_Surgical_google_owlvit-base-patch16.pth"))
    model.eval()
    return model, post, args


def draw_boxes_on_image(pil_image, boxes, labels, scores):
    draw = ImageDraw.Draw(pil_image)
    for box, label, score in zip(boxes, labels, scores):
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0, y0), f"{label} {score:.2f}", fill="white")
    return pil_image


def run_inference(model, postprocessor, args, image: Image.Image):
    from transformers import OwlViTProcessor

    processor = model.processor
    inputs = processor(images=image, return_tensors="pt").to(args.device)
    pixel_values = inputs["pixel_values"]

    with torch.no_grad():
        outputs = model(pixel_values)

    orig_size = torch.tensor([image.size[::-1]])  # (H, W)
    results = postprocessor(outputs, orig_size)

    return results


# ----------------- Streamlit App ----------------- #

st.set_page_config(page_title="FOMO Open World Detector", layout="centered")
st.title("🔍 Open World Object Detection with OWL-ViT + FOMO")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    with st.spinner("Loading model..."):
        model, postprocessor, args = load_model()

    with st.spinner("Running detection..."):
        results = run_inference(model, postprocessor, args, image)

    boxes = results[0]['boxes'].cpu().numpy()
    scores = results[0]['scores'].cpu().numpy()
    labels = results[0]['labels'].cpu().numpy()

    # Filter boxes by score threshold
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.3)
    keep = scores > threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    st.subheader("Detected Objects")
    if len(boxes) > 0:
        image_with_boxes = draw_boxes_on_image(image.copy(), boxes, labels, scores)
        st.image(image_with_boxes, caption="Detections", use_container_width=True)
    else:
        st.warning("No objects detected with confidence above threshold.")
