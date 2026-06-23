# facial_recognition

## Introduction

**Face recognition** is one of the most studied problems in computer vision. It is often broken down into two distinct sub-tasks that are easy to confuse but fundamentally different:

- **Detection** — Detector draws a bounding box around each face found in the frame. It does not care about who the person is.
- **Recognition** — Given a cropped face region (typically the output of a detector), the recognizer tries to match it against a set of known identities.

In a real-world pipeline these two stages are always combined. A typical production system works roughly as follows:

1. A **detector** (HAAR cascade, MTCNN, RetinaFace, …) scans each frame and returns one or more face crops.
2. Each crop is fed into a **feature extractor** (a deep CNN) that maps the face to a compact, high-dimensional embedding vector. Similar faces should be close together in this embedding space; different identities should be far apart.
3. The embedding is then compared against a **database** of enrolled identities, either by nearest-neighbour search or by a lightweight classifier trained on top of the embeddings.
4. If the distance to the closest enrolled identity is below a threshold the person is **recognised**, otherwise they are labelled **unknown**.

This project implements exactly that pipeline on an embedded FPGA board (Xilinx PynqZ2), where compute resources are severely constrained, making every design choice critical.

---

## Model — ResNet50 pre-trained on VGGFace2

The feature extractor used here is a **ResNet50** network pre-trained on [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/), a large-scale face dataset containing more than 3 million images of 9 000+ identities. Pre-training on such a dataset gives the model a very strong prior on what facial features are discriminative.

The model (`.prototxt` and `.caffemodel`) come from a similar android existing [project](https://github.com/MeloniZippoProjects/MultimediaProject).

ResNet50 was chosen over more modern alternatives (MobileFaceNet, ArcFace on a MobileNet backbone, etc.) for a very specific reason: **DPU v1.4 compatibility**. The DPU (Deep Processing Unit) IP core synthesised for PynqZ2 is a small B1152 architecture running at 150 MHz with limited on-chip RAM. The DNNDK v3.1 toolchain enforces strict layer and memory constraints, and most lightweight face models rely on layers or memory patterns that cannot be compiled for this target. ResNet50 is one of the very few models that:

- uses only layers supported by the DPU (Conv, BN, ReLU, Average Pool — see `dexplorer -w` output on target board)
- fits within the available on-chip RAM budget
- still produces high-quality 2048-D embedding vectors

The model is quantised with the `decent` tool (8-bit fixed point) and then compiled with `dnnc` into a `.elf` binary that runs natively on the DPU hardware accelerator, without involving the ARM CPU for inference.

---

## Classifier — Linear SVM

Once a 2048-D embedding is extracted for a new face, it needs to be matched to a known identity. A **Support Vector Machine with a linear kernel** is used for this task.

### How it works

The linear SVM is a binary classifier extended to the multi-class setting via the One-vs-One (OvO) strategy: for N enrolled identities, N·(N−1)/2 binary classifiers are trained, one for each pair of classes. At inference time, each binary classifier casts a vote, and the class that wins the most pairings is selected as the prediction. The confidence score is the signed distance of the query embedding from the winning classifier's separating hyperplane — the farther from the boundary, the more certain the prediction. Predictions whose absolute confidence falls below `SVM_UNKNOWN_THRESHOLD` (default `0.15`) are rejected as `unknown`.

During **enrollment**, one or more photos are provided for a new identity. Each photo is run through the detector + ResNet50 pipeline to produce embeddings, which are stored in `face_database.csv`. The SVM is retrained on the fly on these embeddings. At **recognition** time, every 5 frames the current face crop is embedded and classified by the SVM; if no class scores above the rejection threshold, the label `unknown` is printed.

### Why a linear SVM

A linear SVM is particularly well-suited here for several reasons:

- **Embedding space is already metric**: because the ResNet50 was trained with a metric loss (softmax over VGGFace2 classes), embeddings of the same person cluster tightly and are approximately linearly separable from other identities.
- **Speed**: inference is just a dot product followed by a sign check — essentially free on the ARM CPU even at real-time frame rates.
- **Few training samples**: SVMs generalise well with very few examples per class, which is realistic in an embedded enrollment scenario where you may only have a handful of photos per person.
- **Interpretability**: the decision boundary is explicit and the confidence margin gives a reliable rejection score to flag unknown faces.

### Limitations

The linear SVM requires **at least two samples per enrolled identity** to define a meaningful separating hyperplane between any pair of classes. When only one photo has been provided for an identity, the OvO classifier for that pair degenerates — the hyperplane is unconstrained — and the affected identity is systematically pushed below the confidence threshold and labelled `unknown`. This is not a bug in the SVM implementation; it is a fundamental property of margin-based classifiers that cannot be fixed by tuning `C` or the threshold.

---

## Classifier — Nearest Centroid Classifier (NCC)

### How it works

During the build phase, one **centroid** is computed per enrolled identity by averaging all of its stored embeddings and L2-normalising the result:

```
centroid_i = L2_normalize( mean( embeddings of identity i ) )
```

All centroids are stored as rows of a matrix of shape `[N_identities × 2048]`.

At inference time, the query embedding is also L2-normalised and then compared against every centroid in a single matrix multiplication:

```
similarities = query_embedding * centroid_matrix^T     # shape: 1 × N_identities
```

Because both the query and all centroids are L2-normalised, this dot product is equivalent to the **cosine similarity** between the query and each centroid. The identity whose centroid has the highest similarity is selected as the prediction. If that maximum similarity falls below `NCC_UNKNOWN_THRESHOLD` (default `0.65`), the face is labelled `unknown`.

### Why cosine similarity on L2-normalized embeddings

ResNet50 trained on VGGFace2 produces embeddings that lie approximately on a hypersphere after L2 normalisation. In this geometry, angular distance (equivalently, cosine similarity) is the natural and most discriminative metric — it measures the direction of the embedding vector rather than its magnitude, which can vary with image quality or lighting. Computing cosine similarity via a matrix multiply is also a single O(N) operation regardless of embedding dimension, making it fast enough for real-time use on the Cortex-A9.

---

## Common issues

This project doesn't properly work on single-shot sample. 

---

## Classifier selection

User can compile application selecting one of available classifier, `#define SELECT_CLASSIFIER` could be either 0 for SVM or 1 for NCC.

In general, SVM is preferred even for single-shot sample (that should be avoided as use case) for a limited database because its cost is _quadratic_ for number of classes.

NCC is less accurate but preferred for large dataset.

---

## Board — Xilinx PynqZ2

The [PynqZ2](https://www.tulembedded.com/FPGA/ProductsPYNQ-Z2.html) is a low-cost FPGA development board based on the **Xilinx Zynq-7020 SoC**, which integrates a dual-core ARM Cortex-A9 processor and Xilinx 7-series FPGA fabric on the same chip.

Key hardware characteristics relevant to this project:

| Feature | Value |
|---|---|
| SoC | Xilinx Zynq XC7Z020 |
| ARM CPU | Dual-core Cortex-A9 @ 650 MHz |
| FPGA fabric | 85k logic cells |
| On-chip RAM | 4.9 Mb BRAM |
| External RAM | 512 MB DDR3 |
| DPU core | B1152, v1.4.0 @ 150 MHz |
| OS | Petalinux 2019.2 (Yocto-based) |

The **DPU IP** (Deep Processing Unit) is a programmable inference engine synthesised in the FPGA fabric. It offloads the ResNet50 forward pass from the ARM CPU, providing a significant speedup for the feature extraction step. The ARM CPU handles everything else: camera capture via GStreamer, face detection (HAAR or LBP cascade), SVM/NCC classification, HTTP streaming, and the command-line interface.

---

## Repository structure

```
board/
├── facial_recognition/     <-- board-side application
host/
├── docker-dnndk/           <-- Docker container for model compilation
    └── models/resnet50/    <-- quantization and compilation scripts for resnet50
```

Refer to the individual `README` files inside each folder for detailed setup and usage instructions.
