## ConvLoRA for FastViT

This codebase integrates **Low-Rank Adaptation (LoRA)** into **FastViT** (a hybrid CNN-Transformer architecture). Unlike standard LoRA, which targets Linear layers in Transformers, this implementation adapts `nn.Conv2d` layers to handle the spatial characteristics of FastViT's `RepMixer` and `MobileOne` blocks.

I might use this quite conveniently for downstream tasks like image classification, etc.

### Core Features

* **ConvLoRA Wrapper:** A custom `nn.Module` that wraps existing `nn.Conv2d` layers, freezing the original weights and injecting a trainable side path.
* **Spatial Consistency:**
* **Down-Projection ():** Preserves `kernel_size`, `stride`, and `padding` of the target layer to maintain the receptive field.
* **Up-Projection ():** Uses a  convolution to project back to the output channel dimension.


* **Recursive Patching:** Automatically traverses the FastViT hierarchy (Stages 0-3) to identify and adapt dense convolution layers (primarily  projections).

### Implementation Details

The standard LoRA equation  is modified for convolutions:

1. **Freeze:** The pre-trained target convolution is frozen (`requires_grad=False`).
2. **Adapt:**
* `lora_A`: Reduces channels  (preserves spatial dims).
* `lora_B`: Restores channels  (pointwise mixing).


3. **Scale:** The output is scaled by  and added to the original output.

### Quick Start

1. **Install Dependencies:**
```bash
pip install timm torch torchvision

```


2. **Run the Patching Script:**
The provided script loads a `fastvit_t8` model from `timm`, applies LoRA to all 4 stages, and verifies the parameter counts.
```python
# Example Usage
model = timm.create_model('fastvit_t8', pretrained=True)
model = apply_lora_to_fastvit(model, r=16, alpha=16)

# Ready for training (only LoRA params will update)

```



### Results (FastViT-T8 Example)

* **Rank ():** 16
* **Alpha ():** 16
* **Target:** All 4 Stages (Blocks 0-3)
* **Total Params:** ~4.3M
* **Trainable Params:** ~1.3M (31%)*

**Note: The trainable percentage is higher than typical LLM LoRA implementations because the base model (FastViT-T8) is extremely small (4M params). On larger variants (e.g., FastViT-SA24), this percentage would drop significantly.*

### Why this approach?

Standard Linear LoRA fails on FastViT because the `RepMixer` blocks operate on 4D tensors. Flattening these for a Linear layer would destroy spatial locality and require expensive reshaping. **ConvLoRA** operates directly on the 4D feature maps, preserving the efficiency benefits of the FastViT architecture.
