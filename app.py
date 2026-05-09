import gradio as gr
import os
import sys
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ------------------------------------------------------------
# 1. CONFIGURATION – EDIT THESE PATHS FOR DEVELOPMENT
# ------------------------------------------------------------
# When running as script, these are used.
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
DEV_VALIDATION_FOLDER = os.path.join(os.path.dirname(__file__), "images")
DEV_QUADRANT_MODEL = os.path.join(os.path.dirname(__file__), "quadrant_best.pt")
DEV_TOOTH_MODEL    = os.path.join(os.path.dirname(__file__), "tooth_best.pt")
DEV_DISEASE_MODEL  = os.path.join(os.path.dirname(__file__), "disease_best.pt")

# Confidence thresholds
CONF_QUADRANT = 0.25
CONF_TOOTH    = 0.25
CONF_DISEASE  = 0.25

# ------------------------------------------------------------
# 2. HANDLE PATHS WHEN PACKAGED WITH PYBINDS (PyInstaller)
# ------------------------------------------------------------
if getattr(sys, 'frozen', False):
    # Running as compiled .exe
    BASE_DIR = sys._MEIPASS
    # For a packaged app, you may want to keep validation folder external,
    # or bundle a small set of sample images. Here we assume external:
    VALIDATION_FOLDER = DEV_VALIDATION_FOLDER   # or ask user to set
    QUADRANT_MODEL_PATH = os.path.join(BASE_DIR, "quadrant_best.pt")
    TOOTH_MODEL_PATH    = os.path.join(BASE_DIR, "tooth_best.pt")
    DISEASE_MODEL_PATH  = os.path.join(BASE_DIR, "disease_best.pt")
else:
    # Running as normal Python script
    VALIDATION_FOLDER = DEV_VALIDATION_FOLDER
    QUADRANT_MODEL_PATH = DEV_QUADRANT_MODEL
    TOOTH_MODEL_PATH    = DEV_TOOTH_MODEL
    DISEASE_MODEL_PATH  = DEV_DISEASE_MODEL

# ------------------------------------------------------------
# 3. LOAD MODELS (once)
# ------------------------------------------------------------
print("Loading YOLO models...")
quadrant_model = YOLO(QUADRANT_MODEL_PATH)
tooth_model    = YOLO(TOOTH_MODEL_PATH)
disease_model  = YOLO(DISEASE_MODEL_PATH)
print("All models loaded.")

# Get class names from disease model (for label drawing)
disease_class_names = disease_model.names   # dict {0: "caries", 1: "periapical", ...}

# ------------------------------------------------------------
# 4. HELPER: run detection and return (boxes, class_ids, confs)
# ------------------------------------------------------------
def get_detections(model, image_path, conf_threshold):
    """Return list of (box, class_id, confidence)"""
    results = model(image_path, conf=conf_threshold, verbose=False)
    detections = []
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        for box, cid, conf in zip(boxes, cls_ids, confs):
            x1, y1, x2, y2 = map(int, box[:4])
            detections.append(((x1, y1, x2, y2), cid, conf))
    return detections

def get_boxes_only(model, image_path, conf_threshold):
    """Simpler: return list of boxes (for quadrant/tooth)"""
    detections = get_detections(model, image_path, conf_threshold)
    return [box for box, _, _ in detections]

# ------------------------------------------------------------
# 5. MAIN PIPELINE WITH LABELS
# ------------------------------------------------------------
def run_pipeline(selected_image_path: str, history):
    if not selected_image_path or not os.path.exists(selected_image_path):
        history.append({"role": "assistant", "content": "❌ Image not found. Please select a valid file."})
        return history, None

    original = Image.open(selected_image_path).convert("RGB")
    
    # Step 1
    history.append({"role": "assistant", "content": "🦷 Step 1: Running quadrant detection..."})
    quadrant_boxes = get_boxes_only(quadrant_model, selected_image_path, CONF_QUADRANT)
    history.append({"role": "assistant", "content": f"   → Found {len(quadrant_boxes)} quadrant(s)."})
    
    # Step 2
    history.append({"role": "assistant", "content": "🦷 Step 2: Running tooth detection..."})
    tooth_boxes = get_boxes_only(tooth_model, selected_image_path, CONF_TOOTH)
    history.append({"role": "assistant", "content": f"   → Found {len(tooth_boxes)} tooth region(s)."})
    
    # Step 3 – disease detection with class names
    history.append({"role": "assistant", "content": "🔍 Step 3: Running disease detection..."})
    disease_detections = get_detections(disease_model, selected_image_path, CONF_DISEASE)
    history.append({"role": "assistant", "content": f"   → Found {len(disease_detections)} potential disease(s)."})
    
    # Step 4 – draw on original image: red boxes + text labels
    final_img = original.copy()
    draw = ImageDraw.Draw(final_img)
    
    try:
        # Try to load a larger font (size 24-28)
        font = ImageFont.truetype("arialbd.ttf", 32)  # bold arial
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = ImageFont.load_default()
    
    for (box, class_id, conf) in disease_detections:
        x1, y1, x2, y2 = box
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        # Get class name
        class_name = disease_class_names.get(class_id, f"class_{class_id}")
        label = f"{class_name} ({conf:.2f})"
        # Calculate text size (approximate)
        bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        # Draw background rectangle for text
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="red")
        # Draw text
        draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)
    
    history.append({"role": "assistant", "content": f"✅ Pipeline complete. {len(disease_detections)} disease(s) highlighted with labels."})
    
    return history, final_img

# ------------------------------------------------------------
# 6. GRADIO UI (no 'type' argument, messages format)
# ------------------------------------------------------------
def get_image_list():
    if not os.path.exists(VALIDATION_FOLDER):
        return []
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    files = [f for f in os.listdir(VALIDATION_FOLDER) if f.lower().endswith(exts)]
    return sorted(files)

with gr.Blocks(title="Dental X‑ray Pipeline – 3‑Model Detection") as demo:
    gr.Markdown("""
    # 🦷 Dental X‑ray Analysis Pipeline
    **Quadrant → Tooth → Disease detection**  
    Select an X‑ray from the validation folder. The pipeline will:
    1. Detect quadrants (first “mask”)
    2. Detect teeth within (second “mask”)
    3. Detect diseases (caries, periapical, etc.)
    4. Draw **disease boxes with class labels** on the original X‑ray.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_dropdown = gr.Dropdown(
                choices=get_image_list(),
                label="📁 Choose an X‑ray from validation folder",
                interactive=True
            )
            refresh_btn = gr.Button("🔄 Refresh list")
            run_btn = gr.Button("🚀 Run Pipeline", variant="primary")
        
        with gr.Column(scale=2):
            output_image = gr.Image(label="Final Result – Disease detections on original X‑ray", type="pil")
    
    chatbot = gr.Chatbot(label="Pipeline Log", height=300)
    history_state = gr.State([])   # list of message dicts
    
    def refresh_list():
        return gr.Dropdown(choices=get_image_list())
    refresh_btn.click(refresh_list, None, image_dropdown)
    
    def on_run(selected_filename, history):
        if not selected_filename:
            history = history or []
            history.append({"role": "assistant", "content": "⚠️ Please select an image first."})
            return history, None, history
        full_path = os.path.join(VALIDATION_FOLDER, selected_filename)
        new_history, final_img = run_pipeline(full_path, history or [])
        return new_history, final_img, new_history
    
    run_btn.click(
        on_run,
        inputs=[image_dropdown, history_state],
        outputs=[chatbot, output_image, history_state]
    )
    
    demo.load(lambda: [{"role": "assistant", "content": "Ready. Select an X‑ray and click 'Run Pipeline'."}], None, chatbot)

# ------------------------------------------------------------
# 7. LAUNCH (auto-open browser, quiet)
# ------------------------------------------------------------
if __name__ == "__main__":
    import webbrowser, threading
    def open_browser():
        webbrowser.open("http://127.0.0.1:7861")
    threading.Timer(1.5, open_browser).start()
    demo.launch(server_name="127.0.0.1", server_port=7861, quiet=True)