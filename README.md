
# ContactAngle_Interpolation

This project addresses the challenge of reconstructing spatial wettability distributions across porous rock surfaces using sparse in-situ contact angle measurements. It provides a computational framework for assigning contact angles to 3D domains by extrapolating pointwise measurements using geometric and statistical techniques.

---

## 📚 Abstract

Local wettability variations in porous media arise from mineral composition, fluid–solid history, and surface roughness. These variations—commonly referred to as *mixed-wetting*—strongly influence multiphase flow and relative permeability. While micro-CT imaging enables automated contact angle measurements, these are limited to regions with visible triple-phase contact lines. 

This work proposes a **computational heuristic** to extrapolate such pointwise measurements across the solid surface using domain-specific clustering and interpolation techniques. Using Bentheimer sandstone volumes, synthetic wettability distributions are generated and tested through various interpolation strategies, including **nearest neighbor**, **grain-based propagation**, and **kriging**. Accuracy is quantified by voxel-wise errors, and flow impact is evaluated via Lattice Boltzmann simulations.

---

## 🧠 Project Goals

- Generate synthetic 3D porous rock models with ground-truth contact angle distributions.
- Apply different interpolation strategies to propagate sampled contact angle values.
- Validate accuracy and simulate the effects on fluid flow using interpolated fields.
- Visualize and analyze interpolation outcomes.

---

## 📁 Directory Overview

```
.
├── main.py                         # Main experiment loop (synthetic data)
├── main_Validation.py             # Validation with real measurement input
├── main_TestKriging.py            # Testing custom kriging methods
├── main_CreateRockVolumes_fromRAW.py  # Process raw micro-CT volumes
├── utilities.py                   # Core utilities and interpolation algorithms
├── Plotter.py                     # 2D/3D plotting and visualization
├── Rock Volumes/                  # Input rock volume datasets (.raw)
└── Interpolated Volumes/         # Output volume interpolations and plots
```

---

## 🧪 Interpolation Methods

### 🔹 1. Nearest Neighbor (NN)
- Removes internal solid regions to focus on the interface.
- For each solid cell, finds the closest sampled point (Euclidean distance).
- Copies the sampled value to the target voxel.
- Fast and simple but ignores geometric barriers and pore structure.

### 🔹 2. Watershed Grain-Based
- Applies 3D watershed clustering on the solid structure.
- Associates each grain with its closest sample point (by centroid).
- Assigns the entire grain the contact angle of its nearest sample.
- Effective for rock types where wettability varies between grains.

### 🔹 3. Sample Expansion
- Implements a BFS-like propagation from sample points across the solid surface.
- Expands to neighboring solid cells, overwriting them with the sample value.
- Ensures full coverage and geometric continuity, but may be over-aggressive.

### 🔹 4. Universal Kriging (Custom)
- Fits a variogram model to sampled values.
- For each solid cell, selects `N` closest samples and builds a kriging system.
- Solves for weights using covariance and solves for the interpolated value.
- Can handle global trends and spatial correlations more flexibly.

---

## ▶️ How to Use

### Step 1: Install Dependencies
```bash
pip install numpy scipy matplotlib pyvista scikit-image
```

### Step 2: Run an Interpolation Experiment
```bash
python main.py
```

### Step 3: Validate with Real Measurements
```bash
python main_Validation.py
```

### Step 4: Test Kriging Performance
```bash
python main_TestKriging.py
```

### Step 5: Generate Processed Rock Volumes
```bash
python main_CreateRockVolumes_fromRAW.py
```

---

## 🧾 Input & Output

- **Input:**
  - `.raw` files: 3D binary arrays (solid=0, fluid=1).
  - `.npy` files: Measured contact angle datasets, shape (4, N) → `[x, y, z, angle]`.
- **Output:**
  - `.png` renderings of interpolated volumes.
  - `.raw` or `.npy` interpolated fields for simulation input.
  - Histograms and statistical plots of interpolation error.

---

## 📊 Visualization

- `Plot_Domain`: Shows scalar field with PyVista colormap.
- `Plot_Classified_Domain`: Highlights discrete classes (e.g., solid/fluid/grain).
- `plot_hist` & `plot_heatmap`: Statistical distribution analysis.

---

## 👨‍🔬 Authors & Credits

Developed at **Universidade Federal de Santa Catarina**  
By Gabriel César Silveira, Christoph I. Zevenbergen, Ricardo L. M. Bazarin, Diogo S. Nardelli  
Presented at **InterPore 2025**

---
