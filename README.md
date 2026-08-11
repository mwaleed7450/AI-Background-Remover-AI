# 🤖 Team 3 — ML / AI

This is where the main ML work happens. This module covers intelligent background removal, edge refinement, image quality analysis, and a set of standalone AI-powered image features.

---

## Background Removal & Refinement Pipeline

### 1. Smart Edge Refinement ⭐

```
Original Image
      +
Initial Mask
      ↓
ML Edge Refinement
      ↓
Improved Mask
```

**Tasks**
- [ ] Create/refine training dataset
- [ ] Train/refine model
- [ ] Hair/fine-edge handling
- [ ] Evaluate IoU
- [ ] Evaluate boundary accuracy
- [ ] Integrate inference

---

### 2. Hair & Fur Detection

**Tasks**
- [ ] Hair/fur dataset
- [ ] Fine-detail segmentation
- [ ] Hair refinement model
- [ ] Fur refinement
- [ ] Edge preservation
- [ ] Testing

---

### 3. Multiple Object Detection

**Detects**
```
Person
Dog
Bag
Car
Product
etc.
```

**Tasks**
- [ ] Object detection
- [ ] Object segmentation
- [ ] Object IDs
- [ ] Generate individual masks
- [ ] Return selectable objects

---

### 4. Image Quality Analysis

**Detects**
- [ ] Blur
- [ ] Noise
- [ ] Low resolution
- [ ] Poor lighting
- [ ] Compression quality

**Sample Output**
```
Resolution: Good
Blur: Low
Noise: Medium
Lighting: Poor
Overall Quality: 78%
```

---

### 5. Background Complexity Detection

**Classifies**
```
Simple
Medium
Complex
```

Then automatically chooses appropriate processing.

---

### 6. Mask Quality Scoring

**Sample Output**
```
Mask Quality: 94%
Edge Quality: 91%
Subject Confidence: 97%
```

---

### 7. Low-Light Enhancement

**Tasks**
- [ ] Detect dark images
- [ ] Image enhancement model
- [ ] Improve visibility
- [ ] Feed enhanced image into segmentation

---

### 8. Blur & Noise Correction

```
Input
 ↓
Detect Blur/Noise
 ↓
Enhancement
 ↓
Background Removal
```

---

### 9. AI Image Upscaling

**Tasks**
- [ ] Super-resolution model
- [ ] 2× upscaling
- [ ] 4× upscaling
- [ ] Quality comparison

---

### 10. Edge Decontamination

Removes leftover background colors around edges.

**Example**
```
Before:
Subject + green halo

After:
Clean subject edge
```

---

### 11. Automatic Shadow Detection

Detects whether a shadow belongs to the subject.

**Tasks (later phase)**
- [ ] Shadow detection
- [ ] Shadow mask
- [ ] Shadow preservation
- [ ] Optional shadow removal

---

## 🔥 Additional ML Features

These are separate from normal background removal.

### 12. AI Image Similarity Search ⭐

```
Upload Image
     ↓
Image Embedding
     ↓
Vector Search
     ↓
Similar Images
```

**Tasks**
- [ ] Image embedding model
- [ ] Generate embeddings
- [ ] Vector database
- [ ] Similarity calculation
- [ ] Ranking
- [ ] Similar-image API

---

### 13. AI Image Categorization

**Classifies**
```
Portrait
Product
Food
Animal
Landscape
Document
Vehicle
etc.
```

---

### 14. Duplicate Image Detection

**Detects**
- [ ] Exact duplicates
- [ ] Resized duplicates
- [ ] Cropped duplicates
- [ ] Slightly modified images

---

### 15. AI Color Palette Extraction

**Extracts**
```
Primary Color
Secondary Color
Accent Color
Dominant Colors
```

**Display**
```
████  ████  ████  ████  ████
```

---

### 16. AI Image Composition Scoring

**Analyzes**
- [ ] Subject positioning
- [ ] Balance
- [ ] Lighting
- [ ] Visual clarity
- [ ] Composition

**Sample Output**
```
Composition Score: 88%
```

---

## 📋 Status Legend

| Symbol | Meaning |
|--------|---------|
| ⭐ | High priority feature |
| [ ] | Task not started |
| [x] | Task completed |