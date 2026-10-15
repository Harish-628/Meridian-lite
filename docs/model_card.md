# Model Card: MERIDIAN-Lite Damage Classifier

Following the framework proposed by Mitchell et al. (2019) (*Model Cards for Model Reporting*), this document details the architecture, intended use, provenance, evaluation, and operational boundaries of the active damage classification model in MERIDIAN-Lite.

---

## 1. Model Details

- **Model Name:** MERIDIAN-Lite Bi-Temporal Damage Classifier (Demonstration CNN)
- **Model File Path:** `models/weights/damage_classifier.onnx`
- **Model Format:** Open Neural Network Exchange (ONNX)
- **IR Version:** 10 | **Opset Version:** 14
- **Model Size:** 23,568 bytes (23.5 KB)
- **SHA-256 Checksum:** `bda9515c3a3c431abf46a8792e20ff131c09aa6d23a1e54c387d344c9a05f753`
- **Truncated Hash:** `bda9515c`
- **Execution Engine:** ONNX Runtime (`CPUExecutionProvider` by default, GPU capable)
- **License:** MIT License (Repository project license; no separate license header in ONNX protobuf)

---

## 2. Model Provenance & Training Status

- **Formal Provenance Classification:** **Model provenance not independently verified**
- **Was the model pretrained?** **No.**
- **Was it fine-tuned from a pretrained checkpoint?** **No.**
- **Was it trained from scratch via backpropagation?** **No.**
- **Was xBD or xView2 used for training?** **No.**
- **Was Hurricane Michael imagery used for training?** **No.**
- **Is it a demonstration model?** **Yes.** It is a hand-parameterized 2-layer convolutional neural network synthesized to demonstrate offline bi-temporal inference, verify multi-stage spatial pipeline integration, and validate the end-to-end GeoJSON generation pipeline without external network or GPU dependency.
- **Data Leakage Risk:** **None.** Because no dataset was used for training or weight optimization, there is zero training/test data leakage between the model weights and disaster evaluation tiles.

---

## 3. Architecture & Tensor Contracts

### Input Tensor Contract
- **Name:** `input`
- **Shape:** `[batch_size, 6, 64, 64]`
- **Data Type:** `FLOAT32`
- **Semantics:** 6-channel bi-temporal tensor representing pre-disaster and post-disaster spatial image patches stacked along the channel dimension:
  $$\text{Channels } 0..2: [R_{\text{pre}}, G_{\text{pre}}, B_{\text{pre}}] \in [0.0, 1.0]$$
  $$\text{Channels } 3..5: [R_{\text{post}}, G_{\text{post}}, B_{\text{post}}] \in [0.0, 1.0]$$

### Output Tensor Contract
- **Name:** `probabilities`
- **Shape:** `[batch_size, 4]`
- **Data Type:** `FLOAT32`
- **Semantics:** Softmax probability distribution over four damage classification tiers adhering to the Joint Damage Scale (JDS / xBD standard):
  - Index 0: `no-damage` (Weight 0.00)
  - Index 1: `minor-damage` (Weight 0.25)
  - Index 2: `major-damage` (Weight 0.75)
  - Index 3: `destroyed` (Weight 1.00)

### Internal Layers
1. **Conv1:** `Conv(input, w_conv1, b_conv1, strides=[1,1], pads=[1,1,1,1])` $\rightarrow [B, 16, 64, 64]$
   - Analytical difference kernels ($+1/27$ post vs $-1/27$ pre) and Laplacian edge filters.
2. **ReLU1:** `Relu(Conv1)`
3. **MaxPool1:** `MaxPool(ReLU1, kernel=[2,2], stride=[2,2])` $\rightarrow [B, 16, 32, 32]$
4. **Conv2:** `Conv(MaxPool1, w_conv2, b_conv2, strides=[1,1], pads=[1,1,1,1])` $\rightarrow [B, 32, 32, 32]$
5. **ReLU2:** `Relu(Conv2)`
6. **GlobalAveragePool:** `GlobalAveragePool(ReLU2)` $\rightarrow [B, 32, 1, 1]$
7. **Flatten:** `Flatten(GlobalAveragePool)` $\rightarrow [B, 32]$
8. **FC (Gemm):** `Gemm(Flatten, w_fc, b_fc)` $\rightarrow [B, 4]$
9. **Softmax:** `Softmax(FC, axis=1)` $\rightarrow [B, 4]$

---

## 4. Performance & Operational Characteristics

- **Inference Latency (CPU SIMD):** ~1.9 ms per patch; ~113.4 ms for full 58-patch building mosaic.
- **Memory Footprint:** < 35 MB working set in ONNX Runtime.
- **Hardware Requirement:** Generic x86_64 or ARM64 CPU with standard SIMD (AVX2/NEON). No GPU or CUDA required.
- **Evaluated Test Sets:**
  - Held-out Hurricane Michael GeoTIFF tile (`data/processed/hurricane_michael_test_pre.tif` & `post.tif`).
  - Candidate building patches: 58 detected spatial features.
  - Mean confidence score: 0.4208.
  - Priority Zone 1 allocation: `Z-005` (Priority Score: 0.6785).

---

## 5. UI Transparency & Compliance Rules

To prevent misleading claims to users, evaluators, or competition judges:
1. **Never** label this model as "pretrained" or "xView2 trained".
2. **Display** as: `"Verified Local ONNX Model (Lightweight Demonstration CNN)"` with truncated hash `bda9515c`.
3. **Clarify** in model info: *"Model Provenance Not Independently Verified (not an official xBD checkpoint)"*.
4. **Distinguish Modes:**
   - **Real Model Inference:** Local neural network inference via ONNX Runtime on real uploaded or preprocessed imagery.
   - **Demo Simulation:** Synthetic deterministic benchmark simulation for offline UI walkthroughs.
