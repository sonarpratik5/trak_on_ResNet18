# TRAK Data Attribution + Grad-CAM Analysis on ResNet-18 (CIFAR-10)

## Overview

This project explores **data attribution and interpretability in deep neural networks** by combining:

- **TRAK (Training Data Attribution)**
- **Grad-CAM (Gradient-weighted Class Activation Mapping)**

The goal of this experiment was to:

1. Identify **which training samples most influenced a model prediction**.
2. Analyze **whether influential samples share visual representations** with the test input.
3. Visualize **which image regions the model uses for its prediction**.

The model used for the experiment is **ResNet-18 trained on the CIFAR-10 dataset**.

---

## Experiment Pipeline

The workflow of the experiment is as follows:

1. Train **ResNet-18** on **CIFAR-10**.
2. Select a **test image** (cat in this case).
3. Apply **TRAK attribution** to retrieve the **top-k influential training samples**.
4. Apply **Grad-CAM** on:
   - the **test image**
   - the **top influential training samples**
5. Compare the **activation regions** to identify **shared representations**.

---

## Visualization

<image1>

**Figure Description**

Top section:
- Test image used for inference.
- Top-5 training samples identified by **TRAK** as most influential.

Bottom section:
- **Grad-CAM heatmaps** for:
  - the test image
  - the top influential samples

These heatmaps highlight **regions of the image that contributed most to the model's prediction**.

---

## Observations

### 1. Influential Samples Are Not Always the Same Class

Although the **test image belongs to the "cat" class**, some of the **top influential samples belong to different classes** such as:

- airplane
- car
- truck

This indicates that **the model does not rely purely on semantic similarity** between objects.

Instead, the attribution suggests the model may rely on **shared visual patterns**.

---

### 2. Shared Low-Level Visual Features

Across the influential samples and the test image, several **common visual patterns** appear:

- **Blue background / sky**
- **Horizontal edges**
- **Outdoor lighting conditions**
- **Large smooth color regions**

This suggests the model may be using **low-level visual cues** such as:

- color gradients
- texture
- edge orientation

rather than strictly recognizing **object identity**.

This behavior is a known effect in **small datasets like CIFAR-10**, where models often learn **shortcut features**.

---

### 3. Grad-CAM Spatial Resolution Limitation

Grad-CAM was applied to the **last convolutional layer of ResNet-18**.

For CIFAR-10 inputs (32×32), the final convolutional feature map is approximately:

2 × 2

This produces a heatmap structure like:


When upsampled back to the original image size, the visualization becomes **very coarse**, resulting in **large blurry activation regions**.

Because of this limitation, Grad-CAM cannot localize fine-grained features such as:

- cat ears
- eyes
- whiskers
- object boundaries

Instead, it highlights **broad regions of the image**.

---

### 4. Representation Similarity

Even with coarse heatmaps, there are visible similarities between the test image and influential samples:

- Activation along **horizontal boundaries**
- Focus on **background regions**
- Emphasis on **large color transitions**

This supports the idea that the model is partially relying on **background structure rather than object-specific features**.

---

## Limitations of CIFAR-10 for Interpretability

CIFAR-10 images are only:

32 × 32 pixels


This causes two major issues:

1. **Low spatial resolution in final feature maps**
2. **Limited visual detail for attribution methods**

As a result, interpretability methods like **Grad-CAM produce coarse explanations**.

---

## Possible Improvements

### 1. Apply Grad-CAM on Earlier Layers

Instead of the final convolutional layer, Grad-CAM could be applied to earlier layers such as:

layer2
layer3

These layers have larger feature maps:

8 × 8
or
16 × 16


which would produce **more detailed activation maps**.

---

### 2. Use Higher Resolution Datasets

Running the same experiment on **ImageNet** would significantly improve interpretability.

For ImageNet (224×224 inputs), the final convolutional layer in ResNet-18 has spatial size:

7 × 7


This allows Grad-CAM to highlight:

- object parts
- textures
- shapes
- semantic features

leading to **more meaningful comparisons between influential samples**.

---

## Key Insight

The experiment suggests that:

> Influential training samples identified by TRAK may share **low-level visual representations** with the test input, even when they belong to **different semantic classes**.

This highlights an important phenomenon in deep learning:

**Models often rely on statistical visual patterns rather than purely semantic understanding.**

---

## Future Work

Potential extensions of this experiment include:

- Running the pipeline on **ImageNet models**
- Comparing **multiple attribution methods**
- Using **higher-resolution interpretability techniques**
- Investigating **dataset biases revealed through TRAK**

---

## Tools Used

- PyTorch
- ResNet-18
- CIFAR-10
- TRAK (Training Data Attribution)
- Grad-CAM

---
