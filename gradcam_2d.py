
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: shape (1, C, H, W), already on correct device
        class_idx: target class. If None, uses predicted class.
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)  # (1, num_classes)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Zero grads and backward on target class score
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Pool gradients across spatial dims → (C,)
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])  # (C,)

        # Weight activations by pooled grads
        cam = self.activations[0]  # (C, H, W)
        for i in range(cam.shape[0]):
            cam[i] *= pooled_grads[i]

        # Average over channels, ReLU, normalize
        cam = cam.mean(dim=0).numpy()          # (H, W)
        cam = np.maximum(cam, 0)               # ReLU
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize to [0,1]
        return cam, class_idx

    def overlay_on_image(self, original_img_np, cam, alpha=0.4):
        """
        original_img_np: HxWxC numpy array, uint8 [0,255]
        cam: HxW float array [0,1]
        Returns: blended HxWxC image
        """
        h, w = original_img_np.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (1 - alpha) * original_img_np + alpha * heatmap
        return np.uint8(overlay)

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


### Usage

import torchvision.models as models

# Load YOUR trained model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, NUM_CLASSES)  # match your architecture
model.load_state_dict(torch.load("your_model.pth", map_location="cpu"))
model.eval()

# Attach GradCAM to last conv layer
gradcam = GradCAM(model, target_layer=model.layer4[1].conv2)

# Run on a sample
input_tensor = your_preprocessed_image.unsqueeze(0)  # (1, C, H, W)
cam, predicted_class = gradcam.generate(input_tensor)

# Overlay
overlay = gradcam.overlay_on_image(original_img_np, cam)

# Cleanup when done
gradcam.remove_hooks()
