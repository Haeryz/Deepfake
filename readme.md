Base mode : 2D CNN
Model1 : timm/tf_efficientnetv2_s.in21k
Model2 : google/mobilenet_v2_1.0_224
Dataset : ComplexDataLab/OpenFake

# OpenFake Deepfake Detection Training Pipeline

## 📊 Dataset Preparation

### Load the OpenFake Dataset
First, we'll load exactly 12,000 samples from the massive OpenFake dataset - perfectly balanced between real and fake images to avoid any bias in our models.

### Balance and Split the Data
We'll ensure a clean 50/50 split of real vs fake images, then divide into:
- **Training set:** 8,400 images (70%) - where our models learn
- **Validation set:** 1,800 images (15%) - to tune hyperparameters
- **Test set:** 1,800 images (15%) - final performance evaluation

### Preprocess Images
All images get standardized to 224×224 pixels with ImageNet normalization. This ensures consistency across all three models and leverages the power of pretrained weights.

---

## 🧠 Model 1: Custom CNN from Scratch

### Architecture Design
Build a lightweight convolutional neural network with:
- **4 Conv2D blocks** with increasing filters (32→64→128→256)
- **MaxPooling** after each block to reduce spatial dimensions
- **Batch Normalization** for training stability
- **Dropout layers** to prevent overfitting
- **Dense layers** for final classification

Target: ~500k-1M parameters (small enough for Streamlit Community Cloud)

### Training Configuration
- **Optimizer:** Adam with learning rate 1e-3
- **Loss:** Binary Cross-Entropy
- **Epochs:** 20-30 (since we're training from scratch)
- **Batch size:** 64 (L4 can handle this easily)
- **Early stopping** based on validation loss

### Expected Performance
Since we're training from scratch on only 12k images, expect:
- **Accuracy:** 70-80% (decent but not stellar)
- **Training time:** ~30-45 minutes on L4

This model proves you understand the fundamentals, but won't beat pretrained models.

---

## 🚀 Model 2: EfficientNetV2-S (timm/tf_efficientnetv2_s.in21k)

### Load Pretrained Weights
Pull the model pretrained on ImageNet-21k - this gives us a massive head start with features learned from 21,841 classes.

### Freeze and Unfreeze Strategy
- **Phase 1:** Freeze all layers except the classifier head, train for 5 epochs
- **Phase 2:** Unfreeze all layers, fine-tune end-to-end for 10-15 more epochs

This two-phase approach prevents destroying the pretrained features while adapting to deepfake detection.

### Training Configuration
- **Optimizer:** AdamW with learning rate 3e-4 (lower than custom CNN)
- **Scheduler:** Cosine annealing to gradually reduce learning rate
- **Loss:** Binary Cross-Entropy with label smoothing (0.1)
- **Epochs:** 15-20 total
- **Batch size:** 32 (EfficientNet is bigger, reduce batch size)
- **Mixed precision (FP16)** to speed up training on L4

### Expected Performance
With transfer learning magic:
- **Accuracy:** 88-94% (significantly better than custom CNN)
- **Training time:** ~1-1.5 hours on L4

---

## 📱 Model 3: MobileNetV2 (google/mobilenet_v2_1.0_224)

### Load Pretrained Weights
Get MobileNetV2 pretrained on ImageNet-1k - optimized for mobile/edge deployment with depthwise separable convolutions.

### Freeze and Unfreeze Strategy
Same two-phase approach:
- **Phase 1:** Freeze backbone, train classifier (5 epochs)
- **Phase 2:** Unfreeze all, fine-tune (10-15 epochs)

### Training Configuration
- **Optimizer:** AdamW with learning rate 5e-4 (slightly higher than EfficientNet)
- **Scheduler:** Cosine annealing
- **Loss:** Binary Cross-Entropy with label smoothing
- **Epochs:** 15-20 total
- **Batch size:** 64 (MobileNet is lightweight)
- **Mixed precision (FP16)** for faster training

### Expected Performance
Fast and efficient:
- **Accuracy:** 85-91% (good balance of speed and accuracy)
- **Training time:** ~45 minutes - 1 hour on L4

---

## 📈 Training Monitoring

### Track These Metrics
For all three models, log:
- **Training loss** and **validation loss** (watch for overfitting)
- **Training accuracy** and **validation accuracy**
- **Precision, Recall, F1-Score** (more important than raw accuracy for deepfakes)
- **Confusion matrix** (are we biased toward real or fake?)

### Visualizations
Create plots showing:
- Loss curves over epochs
- Accuracy curves over epochs
- Learning rate schedule
- Sample predictions with confidence scores

---

## 💾 Model Export to ONNX

### Why ONNX?
Streamlit Community Cloud has 1GB RAM (same constraint as Azure F1). ONNX Runtime is lightweight (~100MB vs PyTorch's 600MB) and supports quantization - perfect for keeping your app within memory limits.

### Export Process (for each model)
1. **Set model to eval mode** (disable dropout, batchnorm)
2. **Create dummy input** (1×3×224×224 tensor)
3. **Export to ONNX** with opset version 14
4. **Apply dynamic quantization** (INT8 for weights)
5. **Verify the quantized model** works correctly

### Expected Model Sizes
- Custom CNN: 2-4MB (ONNX quantized)
- EfficientNetV2-S: 5-7MB (ONNX quantized)
- MobileNetV2: 3-4MB (ONNX quantized)
- **Total: ~12-15MB for all three** ✅

### Deployment Package
You'll push to GitHub:
- Three ONNX models (12-15MB total)
- Streamlit app script
- requirements.txt (onnxruntime, streamlit, pillow, numpy)
- README with demo instructions

**Streamlit will automatically deploy from your GitHub repo** - no manual server setup needed!

---

## 🧪 Final Evaluation

### Test Set Performance
Run all three models on the held-out test set (1,800 images) and compare:
- **Accuracy**
- **Precision** (of predicted fakes, how many are actually fake?)
- **Recall** (of actual fakes, how many did we catch?)
- **F1-Score** (harmonic mean - best single metric)
- **Inference time** (ONNX on CPU)

### Expected Results Comparison
| Model | Accuracy | F1-Score | ONNX Size | Inference Time |
|-------|----------|----------|-----------|----------------|
| Custom CNN | 72-78% | 0.70-0.76 | ~3MB | ~50ms |
| EfficientNetV2-S | 89-93% | 0.88-0.92 | ~6MB | ~120ms |
| MobileNetV2 | 86-90% | 0.84-0.89 | ~3.5MB | ~80ms |

### Portfolio Story
This comparison shows:
- **Custom CNN:** You understand fundamentals, but limited by training from scratch
- **EfficientNetV2:** Best accuracy, modern architecture, worth the extra compute
- **MobileNetV2:** Sweet spot for edge deployment - fast and accurate enough

---

## 🎯 Ready for Streamlit Deployment

After this notebook, you'll have:
✅ Three trained models with different approaches
✅ Three quantized ONNX models ready for Streamlit Cloud
✅ Complete training logs and visualizations
✅ Performance comparison showing tradeoffs
✅ Everything needed for your portfolio app

**Total training time on L4: ~3-4 hours for all three models**

**Streamlit Community Cloud specs:**
- 1GB RAM (same as Azure F1)
- Free tier
- Auto-deploys from GitHub
- Perfect for portfolio projects
- No cold start issues like Azure F1

**Next step:** Build the Streamlit app with three tabs (one per model) that loads the ONNX models and lets users upload images for real-time deepfake detection.

---

**Ready for the actual code?**