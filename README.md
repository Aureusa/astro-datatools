# 🌌 astro-datatools

**astro-datatools** is a lightweight, astronomy-focused data processing toolkit designed for working with FITS-based datasets in modern data science and machine learning workflows.

The library provides reusable abstractions for loading, transforming, and managing astronomical data while preserving metadata and enabling reproducible, on-the-fly transformations. It is intentionally **model-agnostic** and **framework-independent**.

---

## ✨ Features

- 📂 Native **FITS file handling**
- 🧱 Object-oriented data abstraction with metadata tracking
- 🔄 Modular, composable data transformations
- ⚡ Lazy / on-the-fly data generation
- 🧪 Reproducible data enhancement pipelines
- 🧩 Designed to integrate with ML frameworks (e.g. PyTorch, Detectron2) without coupling

---

## 🧠 Design Philosophy

- **Astronomy-first**: Built around FITS files and astro metadata
- **Separation of concerns**: Data handling lives independently of models
- **Composable transforms**: Each transformation is a first-class object
- **Reproducibility**: All transformations are traceable and metadata-aware
- **Minimal assumptions**: No hard dependency on a specific task or framework

---
