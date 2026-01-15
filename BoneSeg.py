import os
import sys
import types
from pathlib import Path


mock_modules = [
    'torch._dynamo',
    'torch._dynamo.utils',
    'torch._dynamo.config',
    'torch._dynamo.convert_frame',
    'torch._dynamo.eval_frame',
    'torch._dynamo.resume_execution',
    'torch._numpy'
]

for mod_name in mock_modules:
    mock_mod = types.ModuleType(mod_name)
    sys.modules[mod_name] = mock_mod

sys.modules['torch._dynamo.utils'].is_compile_supported = lambda: False
sys.modules['torch._dynamo'].optimize = lambda *args, **kwargs: (lambda x: x)

import numpy as np
from PIL import Image
from nd2reader import ND2Reader
from tkinter import filedialog, messagebox
import customtkinter as ctk
import torch
import csv
from skimage.measure import label, regionprops
import threading


torch.jit._state.disable()
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys._MEIPASS)
    SAM2_ROOT = BASE_DIR / "sam2"
else:
    BASE_DIR = Path(__file__).parent.absolute()
    SAM2_ROOT = BASE_DIR / "sam2"

if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))

os.environ["PYTHONPATH"] = str(SAM2_ROOT)

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir

GlobalHydra.instance().clear()
config_dir = str(SAM2_ROOT / "sam2" / "configs")
initialize_config_dir(config_dir=config_dir, version_base=None)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class SegmentationApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("BoneSeg")
        
        icon_path = os.path.join(BASE_DIR, "icon.ico")
        if os.path.exists(icon_path):
            try:
                self.iconbitmap(icon_path)
            except:
                pass

        self.geometry("1400x900")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))
        self.after(10, lambda: self.state("zoomed"))
        
        self.left_frame = ctk.CTkFrame(self, width=350, corner_radius=10)
        self.left_frame.pack(side="left", fill="y", padx=10, pady=10)

        self.right_frame = ctk.CTkFrame(self, corner_radius=10)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.bottom_frame = ctk.CTkFrame(self, height=150, corner_radius=10)
        self.bottom_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        self.image_label = ctk.CTkLabel(self.right_frame, text="Load an image to start")
        self.image_label.pack(expand=True)

        self.file_path = ctk.StringVar()
        self.segmentation_tool = ctk.StringVar(value="SAM2 - Predictor")
        self.sam2_version = ctk.StringVar(value="trained")
        self.object_choice = ctk.StringVar(value="lacunae")
        self.channel_choice = ctk.StringVar(value="DAPI")
        self.latest_stats = None

        self.build_left_panel()

        self.results_text = ctk.CTkTextbox(self.bottom_frame, height=140)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)

        self.current_image_np = None
        self.segmented_image_np = None
        self.current_image_tk = None

    def build_left_panel(self):
        logo_path = os.path.join(BASE_DIR, "logo.png")
        if os.path.exists(logo_path):
            logo_img = Image.open(logo_path)
            self.logo_tk = ctk.CTkImage(logo_img, size=(100, 100))
            ctk.CTkLabel(self.left_frame, image=self.logo_tk, text="").grid(row=0, column=0, pady=10)
        
        self.left_frame.grid_columnconfigure(0, weight=1)
        row = 1

        ctk.CTkLabel(self.left_frame, text="ND2 File:").grid(row=row, column=0, sticky="w", padx=10, pady=(10, 0))
        row += 1
        self.nd2_entry = ctk.CTkEntry(self.left_frame, textvariable=self.file_path, width=280)
        self.nd2_entry.grid(row=row, column=0, sticky="w", padx=10, pady=5)
        row += 1
        ctk.CTkButton(self.left_frame, text="Browse", command=self.select_file).grid(row=row, column=0, sticky="w", padx=10, pady=5)
        row += 1

        ctk.CTkLabel(self.left_frame, text="Channel:").grid(row=row, column=0, sticky="w", padx=10, pady=(15, 0))
        row += 1
        self.channel_combo = ctk.CTkComboBox(self.left_frame, variable=self.channel_choice, values=["DAPI", "CFP", "BrightField", "Cy3", "YFP", "GFP"])
        self.channel_combo.grid(row=row, column=0, sticky="w", padx=10)
        row += 1

        ctk.CTkLabel(self.left_frame, text="Segmentation Method:").grid(row=row, column=0, sticky="w", padx=10, pady=(15, 0))
        row += 1
        self.seg_combo = ctk.CTkComboBox(self.left_frame, variable=self.segmentation_tool, values=["SAM2 - Predictor"])
        self.seg_combo.grid(row=row, column=0, sticky="w", padx=10)
        row += 1

        self.sam2_lbl = ctk.CTkLabel(self.left_frame, text="SAM2 Version:")
        self.sam2_lbl.grid(row=row, column=0, sticky="w", padx=10, pady=(15, 0))
        row += 1
        self.sam2_combo = ctk.CTkComboBox(self.left_frame, variable=self.sam2_version, values=["trained"])
        self.sam2_combo.grid(row=row, column=0, sticky="w", padx=10)
        row += 1

        self.object_lbl = ctk.CTkLabel(self.left_frame, text="Target Object:")
        self.object_lbl.grid(row=row, column=0, sticky="w", padx=10, pady=(15, 0))
        row += 1
        self.object_combo = ctk.CTkComboBox(self.left_frame, variable=self.object_choice, values=["lacunae", "all objects"])
        self.object_combo.grid(row=row, column=0, sticky="w", padx=10)
        row += 1

        self.segment_button = ctk.CTkButton(self.left_frame, text="Run Segmentation", fg_color="green", hover_color="darkgreen", command=self.run_segmentation)
        self.segment_button.grid(row=row, column=0, sticky="w", padx=10, pady=25)
        row += 1

        self.save_img_button = ctk.CTkButton(self.left_frame, text="Save Masked Image", state="disabled", command=self.save_current_segmented_image)
        self.save_img_button.grid(row=row, column=0, sticky="w", padx=10, pady=5)
        row += 1

        self.save_results_button = ctk.CTkButton(self.left_frame, text="Save Results (.csv)", state="disabled", command=self.save_current_results)
        self.save_results_button.grid(row=row, column=0, sticky="w", padx=10, pady=5)

    def select_file(self):
        file = filedialog.askopenfilename(title="Open ND2 File", filetypes=[("ND2 files", "*.nd2")])
        if file:
            self.file_path.set(file)
            self.load_and_display_image(file)

    def load_and_display_image(self, path):
        try:
            with ND2Reader(path) as images:
                Z = images.sizes.get('z', 1)
                channels = images.metadata.get('channels', [])
                channel_map = {name: idx for idx, name in enumerate(channels)}
                target_chan = self.channel_choice.get()
                
                if target_chan not in channel_map:
                    messagebox.showerror("Error", f"Channel {target_chan} not found in file.")
                    return
                
                c_idx = channel_map[target_chan]
                frames = [images.get_frame_2D(c=c_idx, z=z) for z in range(Z)]
                img2d = np.max(np.stack(frames, axis=0), axis=0)
            
            img_min, img_max = img2d.min(), img2d.max()
            img_uint8 = ((img2d - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)
            
            self.current_image_np = img_uint8
            self.display_image(img_uint8)
            self.results_text.insert("end", f"Loaded: {os.path.basename(path)}\n")
        except Exception as e:
            messagebox.showerror("Loading Error", str(e))

    def display_image(self, img_np):
        img_pil = Image.fromarray(img_np)
        self.current_image_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(800, 800))
        self.image_label.configure(image=self.current_image_tk, text="")

    def run_segmentation(self):
        if self.current_image_np is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        self.segment_sam2_predictor()

    def segment_sam2_predictor(self):
        model_cfg = "sam2.1/sam2.1_hiera_b+.yaml"
        ckpt_path = self.get_sam2_checkpoint()
        
        if not os.path.exists(ckpt_path):
            messagebox.showerror("Error", f"Checkpoint not found at: {ckpt_path}")
            return
            
        try:
            self.results_text.insert("end", "Initializing SAM2...\n")
            self.update_idletasks()
            
            model = build_sam2(model_cfg, ckpt_path, device=device)
            predictor = SAM2ImagePredictor(model)
            rgb = np.stack([self.current_image_np] * 3, axis=-1)
            predictor.set_image(rgb)
            
            self.results_text.insert("end", "SAM2 Ready. Left-Click on an object to segment.\n")
            self.enable_click_predictor(rgb, predictor)
        except Exception as e:
            messagebox.showerror("SAM2 Error", str(e))

    def enable_click_predictor(self, rgb, predictor):
        def on_click(event):
            self.config(cursor="watch")
            self.update_idletasks()
            
            x = int(event.x * (self.current_image_np.shape[1] / 800))
            y = int(event.y * (self.current_image_np.shape[0] / 800))
            
            point_coords = np.array([[[x, y]]], dtype=np.int32)
            point_labels = np.array([[1]], dtype=np.int32)
            
            with torch.inference_mode():
                masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels)
            
            best_mask = masks[np.argmax(scores)]
            self.apply_mask_overlay(best_mask)
            self.compute_and_display_metrics(best_mask)
            
            self.config(cursor="")

        self.image_label.bind("<Button-1>", on_click)

    def apply_mask_overlay(self, mask):
        overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
        overlay[mask > 0] = [255, 0, 0] # Rouge pour le masque
        
        alpha = 0.4
        base_rgb = np.stack([self.current_image_np] * 3, axis=-1)
        blended = (base_rgb * (1 - alpha) + overlay * alpha).astype(np.uint8)
        
        self.segmented_image_np = blended
        self.display_image(blended)
        self.save_img_button.configure(state="normal")
        self.save_results_button.configure(state="normal")

    def get_sam2_checkpoint(self):
        obj = "blanc" if self.object_choice.get() == "lacunae" else "objet"
        chan = self.channel_choice.get()
        return str(BASE_DIR / "sam2" / "checkpoints" / "finetuned_checkpoints" / f"checkpoint_{obj}_{chan}.pt")

    def compute_and_display_metrics(self, mask):
        props = regionprops(label(mask))
        if not props: return
        
        num_masks = len(props)
        img_float = self.current_image_np.astype(np.float32)
        
        mean_in = np.mean(img_float[mask > 0])
        mean_out = np.mean(img_float[mask == 0])
        ratio = mean_in / (mean_out + 1e-8)
        
        self.latest_stats = {
            "sample": os.path.basename(self.file_path.get()),
            "count": num_masks,
            "mean_in": round(float(mean_in), 2),
            "mean_out": round(float(mean_out), 2),
            "ratio": round(float(ratio), 3)
        }
        self.results_text.insert("end", f"Detected: {num_masks} | Signal/Noise Ratio: {self.latest_stats['ratio']}\n")
        self.results_text.see("end")

    def save_current_segmented_image(self):
        file = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file and self.segmented_image_np is not None:
            Image.fromarray(self.segmented_image_np).save(file)
            messagebox.showinfo("Success", "Masked image saved.")

    def save_current_results(self):
        if not self.latest_stats: return
        file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file:
            file_exists = os.path.isfile(file)
            with open(file, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Sample", "Mask Count", "Intensity In", "Intensity Out", "Ratio"])
                writer.writerow([self.latest_stats["sample"], self.latest_stats["count"], 
                                 self.latest_stats["mean_in"], self.latest_stats["mean_out"], 
                                 self.latest_stats["ratio"]])
            messagebox.showinfo("Success", "Statistics saved to CSV.")

if __name__ == "__main__":
    app = SegmentationApp()
    app.mainloop()

