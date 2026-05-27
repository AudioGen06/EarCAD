#!/usr/bin/env python3
"""
Neonatal Ear Deformity Intelligent Diagnosis System - Complete Standalone Version
Includes: Binary classification (normal/abnormal), deformity scoring, six-class classification, heatmap generation, YOLO object detection cropping
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import warnings
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

warnings.filterwarnings('ignore')

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")  # YOLO model path


# Setup colored output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


# ==================== 0. YOLO Object Detection Module ====================
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print(f"{Colors.YELLOW}⚠ ultralytics not installed, object detection cropping unavailable{Colors.END}")


class YOLODetector:
    """YOLO object detector for ear detection and cropping"""

    def __init__(self, model_path, device):
        self.device = device
        self.model = None
        self.enabled = False

        if not YOLO_AVAILABLE:
            print(f"{Colors.YELLOW}⚠ YOLO module unavailable, please install: pip install ultralytics{Colors.END}")
            return

        if not os.path.exists(model_path):
            print(f"{Colors.YELLOW}⚠ YOLO model not found: {model_path}{Colors.END}")
            return

        try:
            self.model = YOLO(model_path)
            self.enabled = True
            print(f"{Colors.GREEN}✓ YOLO object detection module loaded successfully{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}✗ YOLO model loading failed: {e}{Colors.END}")

    def detect_and_crop(self, image_path, conf_threshold=0.5):
        """
        Detect ear in the image and crop
        Returns: (cropped_image, bbox, success)
        """
        if not self.enabled:
            return None, None, False

        try:
            # Run detection
            results = self.model(image_path, conf=conf_threshold)

            # Get detection results
            if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                return None, None, False

            # Take the box with highest confidence
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)
            best_box = boxes.xyxy.cpu().numpy()[best_idx]

            x1, y1, x2, y2 = map(int, best_box[:4])

            # Add a small margin (optional, add 5% margin)
            margin_w = int((x2 - x1) * 0.05)
            margin_h = int((y2 - y1) * 0.05)
            x1 = max(0, x1 - margin_w)
            y1 = max(0, y1 - margin_h)
            x2 = x2 + margin_w
            y2 = y2 + margin_h

            # Load and crop image
            original_image = Image.open(image_path).convert('RGB')
            cropped_image = original_image.crop((x1, y1, x2, y2))

            return cropped_image, (x1, y1, x2, y2), True

        except Exception as e:
            print(f"{Colors.YELLOW}⚠ YOLO detection failed: {e}{Colors.END}")
            return None, None, False


# ==================== 1. Deformity Scoring Module (Nonlinear Mapping Version) ====================
class ContrastiveModel(nn.Module):
    """Contrastive learning model (for deformity scoring)"""

    def __init__(self, base_encoder=models.resnet50, feature_dim=128):
        super(ContrastiveModel, self).__init__()
        self.encoder = base_encoder(weights=None)
        self.encoder.fc = nn.Identity()

        # Add projection layer
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        return features

    def get_projection(self, x):
        features = self.encoder(x)
        proj = self.projection(features)
        return F.normalize(proj, dim=1)


class EarSimilarityScorer:
    """Ear deformity scorer - nonlinear mapping: steep rise in key regions"""

    def __init__(self, model, normal_center, transform, device):
        self.model = model
        self.normal_center = normal_center
        self.transform = transform
        self.device = device
        self.model.eval()

    def score_image(self, image):
        """
        Original scoring: returns similarity score (higher = more like normal ear)
        Supports PIL image or path
        """
        try:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                image = image.convert('RGB')
            else:
                raise ValueError("image must be path or PIL Image")

            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.cpu().numpy().squeeze()

            features = features / (np.linalg.norm(features) + 1e-8)
            similarity = np.dot(features, self.normal_center)

            # Similarity score: higher = more like normal ear (0-100)
            similarity_score = (similarity + 1) * 50
            similarity_score = max(0, min(100, similarity_score))

            return similarity_score, similarity

        except Exception as e:
            print("error: {} - {}".format(image, str(e)))
            return 50.0, 0.0

    def get_deformity_score(self, similarity_score):
        """
        Nonlinear deformity mapping
        Critical range: between 55-75% similarity, deformity rises sharply
        """
        if similarity_score >= 75:
            ratio = (100 - similarity_score) / 25
            deformity = ratio * 20
        elif similarity_score >= 55:
            ratio = (75 - similarity_score) / 20
            deformity = 20 + ratio * 65
        else:
            ratio = (55 - similarity_score) / 55
            deformity = 85 + ratio * 15

        deformity = max(0, min(100, deformity))
        return deformity

    def get_level(self, deformity_score):
        """Get level based on deformity score (higher score = more severe deformity)"""
        if deformity_score <= 15:
            return "normal"
        elif deformity_score <= 35:
            return "mild"
        elif deformity_score <= 60:
            return "moderate"
        elif deformity_score <= 85:
            return "severe"
        else:
            return "very severe"


# ==================== 2. Binary Classification Module ====================
class BinaryClassifier:
    """Binary classification model (normal/abnormal) - converted from 7-class model, with bias toward normal threshold"""

    def __init__(self, model_path, device, normal_threshold=0.05):
        self.device = device
        self.normal_threshold = normal_threshold
        self.model = self.load_model(model_path)
        self.transform = self.create_transform()
        self.model.eval()

    def load_model(self, model_path):
        """Load ConvNeXt-Tiny model (7-class)"""
        print(f"{Colors.YELLOW}Loading binary classification model: {model_path}{Colors.END}")

        checkpoint = torch.load(model_path, map_location=self.device)

        if 'norm_mean' in checkpoint and 'norm_std' in checkpoint:
            self.norm_mean = checkpoint['norm_mean']
            self.norm_std = checkpoint['norm_std']
        else:
            self.norm_mean = [0.485, 0.456, 0.406]
            self.norm_std = [0.229, 0.224, 0.225]

        model = models.convnext_tiny(weights=None)
        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features, 7)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)

        if 'val_acc' in checkpoint:
            print(f"  Validation accuracy: {checkpoint['val_acc']:.4f}")
        print(f"  Normal judgment threshold: Normal probability < {self.normal_threshold} -> abnormal")

        return model

    def create_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        ])

    def predict(self, image):
        """Predict single image, supports PIL image or path"""
        try:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                image = image.convert('RGB')
            else:
                raise ValueError("image must be path or PIL Image")

            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                all_probs = probs[0].cpu().numpy()

            normal_prob = all_probs[0]
            is_normal = (normal_prob >= self.normal_threshold)
            confidence = normal_prob if is_normal else (1 - normal_prob)

            return {
                'success': True,
                'is_normal': is_normal,
                'result': 'normal' if is_normal else 'abnormal',
                'confidence': confidence,
                'normal_prob': normal_prob,
                'raw_pred_label': pred_class,
                'all_probs': all_probs
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}


# ==================== 3. Six-Class Classification Module ====================
class SixClassClassifier:
    """Six-class classification model (deformity ear subtypes)"""

    def __init__(self, model_path, device):
        self.device = device
        self.class_names = ['category_1', 'category_2', 'category_3', 'category_4,5', 'category_6', 'category_7']
        self.model = self.load_model(model_path)
        self.transform = self.create_transform()
        self.model.eval()

    def get_target_layer(self):
        if hasattr(self.model, 'features'):
            return self.model.features[-1]
        elif hasattr(self.model, 'layer4'):
            return self.model.layer4[-1]
        else:
            return None

    def load_model(self, model_path):
        print(f"{Colors.YELLOW}Loading six-class classification model: {model_path}{Colors.END}")

        checkpoint = torch.load(model_path, map_location=self.device)

        if 'norm_mean' in checkpoint and 'norm_std' in checkpoint:
            self.norm_mean = checkpoint['norm_mean']
            self.norm_std = checkpoint['norm_std']
        else:
            self.norm_mean = [0.485, 0.456, 0.406]
            self.norm_std = [0.229, 0.224, 0.225]

        try:
            model = models.convnext_tiny(weights=None)
            num_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(num_features, 6)
        except:
            model = models.resnet50(weights=None)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 6)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)

        if 'val_acc' in checkpoint:
            print(f"  Validation accuracy: {checkpoint['val_acc']:.4f}")

        return model

    def create_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        ])

    def predict(self, image):
        try:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                image = image.convert('RGB')
            else:
                raise ValueError("image must be path or PIL Image")

            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()

            return {
                'success': True,
                'predicted_label': pred_class,
                'predicted_class': self.class_names[pred_class],
                'confidence': confidence
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def generate_heatmap(self, image):
        """Generate heatmap, supports PIL image or path"""
        try:
            if isinstance(image, str):
                original_image = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                original_image = image.convert('RGB')
            else:
                return None

            input_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
            input_tensor.requires_grad = True

            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

            target_layer = self.get_target_layer()

            if target_layer is None:
                self.model.zero_grad()
                outputs[0, pred_class].backward()
                gradients = input_tensor.grad
                if gradients is not None:
                    gradients_avg = torch.mean(gradients.abs(), dim=1)
                    cam = gradients_avg.squeeze().cpu().numpy()
                    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
                    cam = cv2.resize(cam, original_image.size)
                    heatmap = cm.viridis(cam)[:, :, :3]
                    heatmap = (heatmap * 255).astype(np.uint8)
                    original_np = np.array(original_image)
                    overlay = cv2.addWeighted(original_np, 0.5, heatmap, 0.5, 0)
                    return overlay
                return None

            features = []

            def hook_fn(module, input, output):
                features.append(output)

            handle = target_layer.register_forward_hook(hook_fn)

            self.model.zero_grad()
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

            outputs[0, pred_class].backward()

            feature_maps = features[0]
            handle.remove()

            if feature_maps.grad is not None:
                gradients = feature_maps.grad
                weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * feature_maps, dim=1).squeeze()
            else:
                cam = torch.mean(feature_maps, dim=1).squeeze()

            cam = F.relu(cam)
            cam = cam.cpu().detach().numpy()

            if np.max(cam) > np.min(cam):
                cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
            else:
                cam = np.zeros_like(cam)

            cam = cv2.resize(cam, original_image.size)
            heatmap = cm.viridis(cam)[:, :, :3]
            heatmap = (heatmap * 255).astype(np.uint8)
            original_np = np.array(original_image)
            overlay = cv2.addWeighted(original_np, 0.5, heatmap, 0.5, 0)

            return overlay

        except Exception as e:
            print(f"{Colors.RED}Heatmap generation failed: {e}{Colors.END}")
            return None

    def display_heatmap(self, image):
        overlay = self.generate_heatmap(image)
        if overlay is not None:
            plt.figure(figsize=(10, 5))
            plt.imshow(overlay)
            plt.title(f'Heatmap')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        return overlay

    def save_heatmap(self, overlay, save_path):
        try:
            Image.fromarray(overlay).save(save_path)
            return True
        except Exception as e:
            print(f"{Colors.RED}Save failed: {e}{Colors.END}")
            return False


# ==================== 4. Main Diagnosis System ====================
class IntegratedDiagnosisSystem:
    """Integrated diagnosis system"""

    def __init__(self):
        print(f"{Colors.BOLD}{Colors.HEADER}")
        print("=" * 60)
        print("      Neonatal Ear Deformity Intelligent Diagnosis System")
        print("=" * 60)
        print(f"{Colors.END}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n{Colors.BLUE}Using device: {self.device}{Colors.END}")

        # Model path configuration
        self.binary_model_path = os.path.join(MODELS_DIR, "best_7cls_convnext_tiny.pth")
        self.six_model_path = os.path.join(MODELS_DIR, "best_6cls_convnext_tiny.pth")
        self.contrastive_model_path = os.path.join(MODELS_DIR, "contrastive_model_final.pth")
        self.normal_center_path = os.path.join(MODELS_DIR, "normal_center.npy")

        # Binary classification bias toward normal threshold
        self.normal_threshold = 0.05

        # YOLO detector
        self.yolo_detector = None

        # Check model files
        self.check_models()

        print(f"\n{Colors.BLUE}Initializing system...{Colors.END}")

        # Binary classification module
        try:
            self.binary_classifier = BinaryClassifier(self.binary_model_path, self.device, self.normal_threshold)
            print(f"{Colors.GREEN}✓ Binary classification module loaded successfully{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}Binary classification module loading failed: {e}{Colors.END}")
            sys.exit(1)

        # Deformity scoring module
        try:
            contrastive_model = ContrastiveModel().to(self.device)
            checkpoint = torch.load(self.contrastive_model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                contrastive_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                contrastive_model.load_state_dict(checkpoint)
            contrastive_model.eval()

            normal_center = np.load(self.normal_center_path)
            normal_center = normal_center / (np.linalg.norm(normal_center) + 1e-8)

            deformity_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.deformity_scorer = EarSimilarityScorer(
                contrastive_model, normal_center, deformity_transform, self.device
            )
            print(f"{Colors.GREEN}✓ Deformity module loaded successfully{Colors.END}")
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ Deformity module loading failed: {e}{Colors.END}")
            self.deformity_scorer = None

        # Six-class classification module
        try:
            self.six_classifier = SixClassClassifier(self.six_model_path, self.device)
            print(f"{Colors.GREEN}✓ Six-class classification module loaded successfully{Colors.END}")
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ Six-class classification module loading failed: {e}{Colors.END}")
            self.six_classifier = None

        # YOLO module
        try:
            self.yolo_detector = YOLODetector(YOLO_MODEL_PATH, self.device)
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ YOLO module initialization failed: {e}{Colors.END}")
            self.yolo_detector = None

    def check_models(self):
        """Check if model files exist"""
        print(f"\n{Colors.BLUE}Checking model files...{Colors.END}")

        if not os.path.exists(MODELS_DIR):
            print(f"{Colors.YELLOW}⚠ models folder does not exist, creating...{Colors.END}")
            os.makedirs(MODELS_DIR)
            print(f"{Colors.GREEN}✓ Created models folder, please place model files in it{Colors.END}")
            print(f"{Colors.RED}✗ Missing model files, program cannot run properly{Colors.END}")
            sys.exit(1)

        missing_models = []
        model_files = [
            (self.binary_model_path, "Binary classification model"),
            (self.six_model_path, "Six-class classification model"),
            (self.contrastive_model_path, "Contrastive learning model"),
            (self.normal_center_path, "Normal center file")
        ]

        for model_path, model_name in model_files:
            if os.path.exists(model_path):
                print(f"  ✓ {model_name}: {os.path.basename(model_path)}")
            else:
                print(f"  ✗ {model_name}: {os.path.basename(model_path)} not found")
                missing_models.append(model_name)

        # Check YOLO model separately
        if os.path.exists(YOLO_MODEL_PATH):
            print(f"  ✓ YOLO model: {os.path.basename(YOLO_MODEL_PATH)}")
        else:
            print(
                f"  ⚠ YOLO model: {os.path.basename(YOLO_MODEL_PATH)} not found (object detection cropping unavailable)")

        if missing_models:
            print(f"\n{Colors.RED}Missing model files: {', '.join(missing_models)}{Colors.END}")
            print(f"{Colors.YELLOW}Please place model files in: {MODELS_DIR}{Colors.END}")
            sys.exit(1)

        print(f"{Colors.GREEN}✓ All required model files check passed{Colors.END}")

    def ask_crop_option(self, image_path=None):
        """Ask whether to enable object detection cropping (pass image path for preview)"""
        if self.yolo_detector is None or not self.yolo_detector.enabled:
            print(f"{Colors.YELLOW}⚠ YOLO module unavailable, cropping cannot be enabled{Colors.END}")
            return False

        print(f"\n{Colors.BOLD}Object Detection Cropping Option{Colors.END}")
        print("When enabled, the system will automatically detect ear position and crop before classification")
        print("Improves robustness against background interference")

        choice = input(
            f"{Colors.YELLOW}Enable object detection cropping? (y/n, default n): {Colors.END}").strip().lower()
        use_crop = (choice == 'y')

        if use_crop:
            print(f"{Colors.GREEN}✓ Object detection cropping enabled{Colors.END}")
        else:
            print(f"{Colors.BLUE}✗ Object detection cropping disabled, using original image{Colors.END}")

        return use_crop

    def process_with_crop(self, image_path, use_crop):
        """
        Process image: if cropping enabled and detection successful, return cropped image
        Otherwise return original image
        Returns: (processed_image, was_cropped, bbox)
        """
        if not use_crop or self.yolo_detector is None or not self.yolo_detector.enabled:
            # No cropping, return original image
            original_image = Image.open(image_path).convert('RGB')
            return original_image, False, None

        # Try detection and cropping
        cropped_image, bbox, success = self.yolo_detector.detect_and_crop(image_path)

        if success and cropped_image is not None:
            print(f"{Colors.GREEN}✓ Ear detected, cropped{Colors.END}")
            return cropped_image, True, bbox
        else:
            print(f"{Colors.YELLOW}⚠ Ear not detected, using original image{Colors.END}")
            original_image = Image.open(image_path).convert('RGB')
            return original_image, False, None

    def save_text_file(self, default_name, content):
        """Save text file"""
        save_path = os.path.join(SCRIPT_DIR, default_name)

        print(f"\n{Colors.YELLOW}Save to {default_name}? (y/n, default n): {Colors.END}", end="")
        choice = input().strip().lower()

        if choice != 'y':
            return None

        print(f"  Save path: {save_path}")
        custom_path = input(f"  Press Enter to use default, or enter custom path: ").strip()

        if custom_path:
            save_path = custom_path

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"{Colors.GREEN}✓ Saved to: {save_path}{Colors.END}")
            return save_path
        except Exception as e:
            print(f"{Colors.RED}Save failed: {e}{Colors.END}")
            return None

    def save_image_file(self, default_name, image_array):
        """Save image file"""
        save_path = os.path.join(SCRIPT_DIR, default_name)

        print(f"\n{Colors.YELLOW}Save to {default_name}? (y/n, default n): {Colors.END}", end="")
        choice = input().strip().lower()

        if choice != 'y':
            return None

        print(f"  Save path: {save_path}")
        custom_path = input(f"  Press Enter to use default, or enter custom path: ").strip()

        if custom_path:
            save_path = custom_path

        try:
            Image.fromarray(image_array).save(save_path)
            print(f"{Colors.GREEN}✓ Saved to: {save_path}{Colors.END}")
            return save_path
        except Exception as e:
            print(f"{Colors.RED}Save failed: {e}{Colors.END}")
            return None

    def save_all_heatmaps(self, folder_path, results, all_images=True):
        """Batch save all heatmaps"""
        print(f"\n{Colors.YELLOW}Enter folder path to save heatmaps: {Colors.END}")
        save_folder = input(f"  (Press Enter to use default: {SCRIPT_DIR}/heatmaps): ").strip()

        if not save_folder:
            save_folder = os.path.join(SCRIPT_DIR, "heatmaps")

        os.makedirs(save_folder, exist_ok=True)

        print(f"\n{Colors.BLUE}Starting heatmap generation and saving...{Colors.END}")
        success_count = 0

        for r in results:
            if all_images or not r['is_normal']:
                filename = os.path.splitext(r['filename'])[0]
                save_path = os.path.join(save_folder, f"{filename}_heatmap.png")
                print(f"  Generating: {r['filename']}")

                overlay = self.six_classifier.generate_heatmap(r['image'])
                if overlay is not None:
                    try:
                        Image.fromarray(overlay).save(save_path)
                        print(f"    ✓ Saved: {save_path}")
                        success_count += 1
                    except Exception as e:
                        print(f"    ✗ Save failed: {e}")
                else:
                    print(f"    ✗ Heatmap generation failed")

        print(f"\n{Colors.GREEN}✓ Done! Successfully saved {success_count} heatmaps to: {save_folder}{Colors.END}")

    def diagnose_single(self, image_path):
        """Diagnose single image"""
        print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
        print(f"{Colors.BOLD}Diagnosing image: {os.path.basename(image_path)}{Colors.END}")
        print(f"{'=' * 60}")

        # Ask whether to crop
        use_crop = self.ask_crop_option(image_path)

        # Process image (crop or original)
        processed_image, was_cropped, bbox = self.process_with_crop(image_path, use_crop)

        if was_cropped:
            print(f"{Colors.BLUE}Using cropped image for diagnosis{Colors.END}")
        else:
            print(f"{Colors.BLUE}Using original image for diagnosis{Colors.END}")

        # Binary classification
        print(f"\n{Colors.BLUE}[Step 1] Binary classification analysis...{Colors.END}")
        binary_result = self.binary_classifier.predict(processed_image)

        if not binary_result['success']:
            print(f"{Colors.RED}Binary classification failed: {binary_result['error']}{Colors.END}")
            return

        # Deformity scoring
        deformity_score_display = None
        deformity_level = None

        if self.deformity_scorer is not None:
            similarity_score, _ = self.deformity_scorer.score_image(processed_image)
            deformity_score_display = self.deformity_scorer.get_deformity_score(similarity_score)
            deformity_level = self.deformity_scorer.get_level(deformity_score_display)

        # Display binary classification results
        print(f"\n{Colors.BOLD}Binary Classification Result:{Colors.END}")
        if binary_result['is_normal']:
            print(f"  Diagnosis: {Colors.GREEN}Normal Ear{Colors.END}")
            print(
                f"  Normal probability: {Colors.YELLOW}{binary_result['normal_prob']:.4f} ({binary_result['normal_prob'] * 100:.2f}%){Colors.END}")
            print(f"  Threshold: Normal probability ≥ {self.normal_threshold}")

            if deformity_score_display is not None:
                deformity_choice = input(f"\nView deformity score? (y/n): ").strip().lower()
                if deformity_choice == 'y':
                    print(f"\n{Colors.BOLD}Deformity Assessment:{Colors.END}")
                    print(f"  Deformity score: {Colors.YELLOW}{deformity_score_display:.2f}/100{Colors.END}")
                    print(f"  Level: {deformity_level}{Colors.END}")

            # Add heatmap generation option for normal ears
            if self.six_classifier is not None:
                heatmap_choice = input(f"\nGenerate and save heatmap for this normal ear? (y/n): ").strip().lower()
                if heatmap_choice == 'y':
                    print(f"\n{Colors.YELLOW}Generating heatmap...{Colors.END}")
                    overlay = self.six_classifier.generate_heatmap(processed_image)
                    if overlay is not None:
                        plt.figure(figsize=(10, 5))
                        plt.imshow(overlay)
                        plt.title(f'Heatmap - Normal')
                        plt.axis('off')
                        plt.tight_layout()
                        plt.show()
                        default_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_heatmap.png"
                        self.save_image_file(default_name, overlay)
        else:
            print(f"  Diagnosis: {Colors.RED}Abnormal Ear{Colors.END}")
            print(
                f"  Normal probability: {Colors.YELLOW}{binary_result['normal_prob']:.4f} ({binary_result['normal_prob'] * 100:.2f}%){Colors.END}")
            print(f"  Threshold: Normal probability < {self.normal_threshold}")

            if deformity_score_display is not None:
                print(f"\n{Colors.BOLD}Deformity Assessment:{Colors.END}")
                print(f"  Deformity score: {Colors.YELLOW}{deformity_score_display:.2f}/100{Colors.END}")
                print(f"  Level: {deformity_level}{Colors.END}")

        # Six-class classification (only for abnormal ears)
        six_result = None
        if not binary_result['is_normal'] and self.six_classifier is not None:
            print(f"\n{Colors.YELLOW}{'─' * 50}{Colors.END}")
            choice = input(f"\nPerform six-class classification? (y/n): ").strip().lower()

            if choice == 'y':
                print(f"\n{Colors.BLUE}[Step 2] Six-class classification...{Colors.END}")
                six_result = self.six_classifier.predict(processed_image)

                if six_result['success']:
                    print(f"\n{Colors.BOLD}Six-class Classification Result:{Colors.END}")
                    print(f"  Class: {Colors.GREEN}{six_result['predicted_class']}{Colors.END}")
                    print(
                        f"  Confidence: {Colors.YELLOW}{six_result['confidence']:.4f} ({six_result['confidence'] * 100:.2f}%){Colors.END}")

                    heatmap_choice = input(f"\nGenerate and save heatmap? (y/n): ").strip().lower()
                    if heatmap_choice == 'y':
                        print(f"\n{Colors.YELLOW}Generating heatmap...{Colors.END}")
                        overlay = self.six_classifier.generate_heatmap(processed_image)
                        if overlay is not None:
                            plt.figure(figsize=(10, 5))
                            plt.imshow(overlay)
                            plt.title(f'Heatmap - {six_result["predicted_class"]}')
                            plt.axis('off')
                            plt.tight_layout()
                            plt.show()
                            default_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_heatmap.png"
                            self.save_image_file(default_name, overlay)
                else:
                    print(f"{Colors.RED}Six-class classification failed: {six_result['error']}{Colors.END}")

        # Save diagnosis results
        save_choice = input(f"\nSave diagnosis results? (y/n): ").strip().lower()
        if save_choice == 'y':
            default_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_result.txt"
            report = self.generate_report(image_path, binary_result, deformity_score_display, deformity_level,
                                          six_result, was_cropped)
            self.save_text_file(default_name, report)

        print(f"\n{Colors.BOLD}{'─' * 50}{Colors.END}")
        print(f"{Colors.BOLD}Diagnosis Summary:{Colors.END}")
        if binary_result['is_normal']:
            print(f"  ✓ Normal ear")
        else:
            print(f"  ✗ Abnormal ear")
            if deformity_score_display is not None:
                print(f"    Deformity score: {deformity_score_display:.2f}")

    def generate_report(self, image_path, binary_result, deformity_score_display, deformity_level, six_result,
                        was_cropped=False):
        """Generate diagnosis report text"""
        report = "=" * 60 + "\n"
        report += "Neonatal Ear Deformity Diagnosis Results\n"
        report += "=" * 60 + "\n\n"
        report += f"Image: {os.path.basename(image_path)}\n"
        report += f"Image path: {image_path}\n"
        report += f"Was cropped: {'Yes' if was_cropped else 'No'}\n\n"
        report += f"Binary classification result: {'Normal Ear' if binary_result['is_normal'] else 'Abnormal Ear'}\n"
        report += f"Normal probability: {binary_result['normal_prob']:.4f} ({binary_result['normal_prob'] * 100:.2f}%)\n"
        report += f"Judgment threshold: Normal probability ≥ {self.normal_threshold} -> normal\n\n"
        if deformity_score_display is not None:
            report += f"Deformity score: {deformity_score_display:.2f}/100 (higher score = more severe deformity)\n"
            report += f"Deformity level: {deformity_level}\n\n"
        if six_result and six_result['success']:
            report += f"Six-class classification result:\n"
            report += f"  Class: {six_result['predicted_class']}\n"
            report += f"  Confidence: {six_result['confidence']:.4f} ({six_result['confidence'] * 100:.2f}%)\n"

        return report

    def diagnose_folder(self, folder_path):
        """Batch diagnose folder"""
        print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
        print(f"{Colors.BOLD}Batch diagnosing folder: {folder_path}{Colors.END}")
        print(f"{'=' * 60}")

        # Ask whether to crop first
        use_crop = self.ask_crop_option()

        # Collect all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path)
                                if f.lower().endswith(ext)])

        if not image_paths:
            print(f"{Colors.RED}No image files in folder{Colors.END}")
            return

        print(f"\nFound {len(image_paths)} images")

        # Batch diagnosis
        results = []
        for i, path in enumerate(image_paths, 1):
            print(f"\n{Colors.BLUE}[{i}/{len(image_paths)}] Processing: {os.path.basename(path)}{Colors.END}")

            # Process image (crop or original)
            processed_image, was_cropped, bbox = self.process_with_crop(path, use_crop)

            # Binary classification
            binary_result = self.binary_classifier.predict(processed_image)
            if not binary_result['success']:
                print(f"  {Colors.RED}Failed: {binary_result['error']}{Colors.END}")
                continue

            # Deformity scoring (only for abnormal ears)
            deformity_score_display = None
            deformity_level = None

            if not binary_result['is_normal'] and self.deformity_scorer is not None:
                similarity_score, _ = self.deformity_scorer.score_image(processed_image)
                deformity_score_display = self.deformity_scorer.get_deformity_score(similarity_score)
                deformity_level = self.deformity_scorer.get_level(deformity_score_display)

            # Record result
            result = {
                'path': path,
                'filename': os.path.basename(path),
                'image': processed_image,
                'is_normal': binary_result['is_normal'],
                'normal_prob': binary_result['normal_prob'],
                'deformity_score': deformity_score_display,
                'deformity_level': deformity_level,
                'was_cropped': was_cropped
            }

            # Display result
            status = f"{Colors.GREEN}Normal{Colors.END}" if binary_result[
                'is_normal'] else f"{Colors.RED}Abnormal{Colors.END}"
            if deformity_score_display is not None:
                print(
                    f"  Result: {status}, Normal prob: {binary_result['normal_prob']:.4f}, Deformity: {deformity_score_display:.2f}")
            else:
                print(f"  Result: {status}, Normal prob: {binary_result['normal_prob']:.4f}")

            results.append(result)

        # Statistics
        normal_count = sum(1 for r in results if r['is_normal'])
        abnormal_count = len(results) - normal_count

        print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
        print(f"{Colors.BOLD}Batch Diagnosis Statistics:{Colors.END}")
        print(f"  Total images: {len(results)}")
        print(f"  Normal ears: {Colors.GREEN}{normal_count}{Colors.END}")
        print(f"  Abnormal ears: {Colors.RED}{abnormal_count}{Colors.END}")
        print(f"  Judgment threshold: Normal probability < {self.normal_threshold} -> abnormal")

        # Six-class classification for abnormal images
        if abnormal_count > 0 and self.six_classifier is not None:
            choice = input(f"\nPerform six-class classification for abnormal images? (y/n): ").strip().lower()
            if choice == 'y':
                print(f"\n{Colors.BLUE}Six-class classification...{Colors.END}")
                for r in results:
                    if not r['is_normal']:
                        print(f"  Processing: {r['filename']}")
                        six_result = self.six_classifier.predict(r['image'])
                        if six_result['success']:
                            r['six_class'] = six_result['predicted_class']
                            r['six_confidence'] = six_result['confidence']
                            print(
                                f"    Result: {six_result['predicted_class']} (Confidence: {six_result['confidence']:.4f})")

        # Batch generate heatmaps for all images (both normal and abnormal)
        if self.six_classifier is not None:
            heatmap_choice = input(
                f"\nBatch generate and save heatmaps for ALL images (including normal) in this batch? (y/n): ").strip().lower()
            if heatmap_choice == 'y':
                self.save_all_heatmaps(folder_path, results, all_images=True)

        # Save results
        save_choice = input(f"\nSave diagnosis results? (y/n): ").strip().lower()
        if save_choice == 'y':
            default_name = f"diagnosis_results_{os.path.basename(folder_path)}.csv"

            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Filename', 'Binary Result', 'Normal Probability', 'Deformity Score', 'Deformity Level',
                             'Six-class Result', 'Six-class Confidence', 'Was Cropped'])

            for r in results:
                writer.writerow([
                    r['filename'],
                    'Normal' if r['is_normal'] else 'Abnormal',
                    f"{r['normal_prob']:.4f}",
                    f"{r['deformity_score']:.2f}" if r['deformity_score'] else 'N/A',
                    r['deformity_level'] if r['deformity_level'] else 'N/A',
                    r.get('six_class', 'N/A'),
                    f"{r.get('six_confidence', 0):.4f}" if r.get('six_confidence') else 'N/A',
                    'Yes' if r.get('was_cropped') else 'No'
                ])

            self.save_text_file(default_name, output.getvalue())

    def main_menu(self):
        """Main menu"""
        while True:
            print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
            print(f"{Colors.BOLD}Main Menu{Colors.END}")
            print(f"{'=' * 60}")
            print("1. Diagnose single image")
            print("2. Batch diagnose folder")
            print("3. View system information")
            print("4. Exit program")
            print("-" * 40)

            choice = input(f"{Colors.YELLOW}Please choose (1-4): {Colors.END}").strip()

            if choice == '1':
                image_path = input(f"\n{Colors.BLUE}Enter image path: {Colors.END}").strip()
                image_path = image_path.strip('\'"')

                if not os.path.exists(image_path):
                    print(f"{Colors.RED}File not found: {image_path}{Colors.END}")
                    continue

                self.diagnose_single(image_path)

            elif choice == '2':
                folder_path = input(f"\n{Colors.BLUE}Enter folder path: {Colors.END}").strip()
                folder_path = folder_path.strip('\'"')

                if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
                    print(f"{Colors.RED}Folder not found or invalid: {folder_path}{Colors.END}")
                    continue

                self.diagnose_folder(folder_path)

            elif choice == '3':
                print(f"\n{Colors.BOLD}System Information:{Colors.END}")
                print(f"  Script directory: {SCRIPT_DIR}")
                print(f"  Models directory: {MODELS_DIR}")
                print(f"  Device: {self.device}")
                print(
                    f"  Binary classification module: {'Loaded' if hasattr(self, 'binary_classifier') else 'Not loaded'}")
                print(f"  Binary classification threshold: Normal probability < {self.normal_threshold} -> abnormal")
                print(f"  Deformity module: {'Loaded' if self.deformity_scorer else 'Not loaded'}")
                print(f"  Six-class module: {'Loaded' if self.six_classifier else 'Not loaded'}")
                print(
                    f"  YOLO cropping module: {'Loaded' if self.yolo_detector and self.yolo_detector.enabled else 'Not loaded'}")
                print(f"\nModel files:")
                print(f"  Binary: {os.path.basename(self.binary_model_path)}")
                print(f"  Six-class: {os.path.basename(self.six_model_path)}")
                print(f"  Contrastive: {os.path.basename(self.contrastive_model_path)}")
                print(f"  Normal center: {os.path.basename(self.normal_center_path)}")
                print(f"  YOLO model: {os.path.basename(YOLO_MODEL_PATH)}")

            elif choice == '4':
                print(f"\n{Colors.GREEN}Thank you for using, goodbye!{Colors.END}")
                break

            else:
                print(f"{Colors.RED}Invalid option, please choose again{Colors.END}")


def main():
    try:
        system = IntegratedDiagnosisSystem()
        system.main_menu()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Program interrupted by user{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}Program error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()