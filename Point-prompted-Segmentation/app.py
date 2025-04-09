# Interactive Image Segmentation with SAM and Custom Unet_Point Model

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import torch
import os
import subprocess
import sys
# clone segment-anything
if not os.path.exists("segment-anything"):
    subprocess.run(["git", "clone", "https://github.com/facebookresearch/segment-anything.git"])

# add to Python path
sys.path.append(os.path.abspath("segment-anything"))
from segment_anything import SamPredictor, sam_model_registry

from unet_point import UNetPointSeg

# ---------- Heatmap Generation ----------
def generate_heatmap_from_point(image_shape, point, sigma=10.0):
    H, W = image_shape
    x, y = point
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    dist_sq = (yy - y) ** 2 + (xx - x) ** 2
    heatmap = np.exp(-dist_sq / (2 * sigma ** 2))
    heatmap = heatmap / np.max(heatmap)
    return heatmap.astype(np.float32)

# ---------- Load SAM model ----------
def load_sam_model(model_type="vit_b", checkpoint_path="sam_vit_b_01ec64.pth"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    predictor = SamPredictor(sam)
    return predictor

# ---------- Load custom Unet_Point model ----------
def load_other_model():
    global your_model
    model = UNetPointSeg()
    parameters_path = "best_param.pth"

    # Always map to CPU first (this is safest for CPU-only environments like HF Spaces)
    state_dict = torch.load(parameters_path, map_location=torch.device("cpu"))

    # Remove 'module.' prefix if present
    new_state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(new_state_dict)
    model.eval()

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    your_model = model
    return "Unet_Point model loaded!"

# ---------- Global model object ----------
predictor = None
your_model = None

# ---------- Initialize SAM ----------
def initialize_sam_model():
    global predictor
    predictor = load_sam_model()
    return "SAM model loaded!"

# ---------- Split function ----------
def segment_from_point(image, evt: gr.SelectData, selected_model):

    # Step 1: Get the original size
    H_orig, W_orig = image.shape[:2]

    # Step 2: Get the original click coordinates
    x_orig, y_orig = evt.index

    # Step 3: resize the image to 224x224
    image = np.array(Image.fromarray(image).resize((224, 224)))

    # Step 4: Map the original coordinates to the new image
    x = int(x_orig * 224 / W_orig)
    y = int(y_orig * 224 / H_orig)
    point = np.array([[x, y]])
    point_label = np.array([1])  # 1 = foreground

    # Step 5: Visualize click points
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 0, 0))
    clicked_image = np.array(pil_image)

    if selected_model == "SAM":
        if predictor is None:
            return image, image, "Error: SAM model not initialized."
        predictor.set_image(image.astype(np.uint8))
        masks, scores, logits = predictor.predict(
            point_coords=point,
            point_labels=point_label,
            multimask_output=True
        )
        best_mask = masks[np.argmax(scores)]
        mask_image = (best_mask * 255).astype(np.uint8)
        mask_image = np.stack([mask_image] * 3, axis=-1)
        return clicked_image, mask_image, f"[SAM] Selected point: ({x}, {y})"

    elif selected_model == "Unet_Point":
        if your_model is None:
            return image, image, "Error: Unet_Point model not initialized."
        H, W, C = image.shape
        heatmap = generate_heatmap_from_point((H, W), (x, y), sigma=10.0)

        image_tensor = torch.from_numpy(image / 255.0).float().permute(2, 0, 1)  # (3, H, W)
        heatmap_tensor = torch.from_numpy(heatmap).float().unsqueeze(0)          # (1, H, W)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = torch.cat([image_tensor, heatmap_tensor], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = your_model(input_tensor)  # (1, 1, H, W)
            mask_pred = output[0, 0].cpu().numpy()  # (H, W)
            mask_binary = (mask_pred > 0.5).astype(np.uint8) * 255
            mask_image = np.stack([mask_binary] * 3, axis=-1)

        return clicked_image, mask_image, f"[Unet_Point] Selected point: ({x}, {y})"

    else:
        return image, image, "Error: Invalid model selected."

# ---------- Gradio interface ----------
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Interactive Image Segmentation")

    with gr.Row():
        with gr.Column():
            init_button = gr.Button("Initialize SAM Model")
            init_output = gr.Textbox(label="SAM Model Status")

            init_unet_button = gr.Button("Initialize Unet_Point Model")
            init_unet_output = gr.Textbox(label="Unet_Point Model Status")

            model_selector = gr.Radio(
                choices=["SAM", "Unet_Point"],
                value="SAM",
                label="Choose Segmentation Model"
            )

            input_image = gr.Image(label="Upload Image", type="numpy")

        with gr.Column():
            clicked_image = gr.Image(label="Image with Selected Point", interactive=False)
            segmented_image = gr.Image(label="Segmentation Result", interactive=False)
            coord_text = gr.Textbox(label="Coordinates")

    # Event Binding
    init_button.click(initialize_sam_model, inputs=[], outputs=init_output)
    init_unet_button.click(load_other_model, inputs=[], outputs=init_unet_output)

    input_image.select(
        segment_from_point,
        inputs=[input_image, model_selector],
        outputs=[clicked_image, segmented_image, coord_text]
    )

    gr.Markdown("""
    ## 🧭 Instructions:
    1. Click **"Initialize SAM Model"** or **"Initialize Unet_Point Model"**
    2. Upload an image
    3. Select a model
    4. Click on the image to perform segmentation
    """)

if __name__ == "__main__":
    demo.launch(show_error=True)
