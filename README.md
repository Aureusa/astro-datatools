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

## 🚀 Getting Started

Create a venv and install the package:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

---

## Projects

- **RaCUN** ([https://github.com/Aureusa/racun](https://github.com/Aureusa/racun)): Radio Component Unification via Neural networks for the LOFAR Two-metre Sky Survey. This project uses `astro-datatools` for data generation, transformation, and augmentation. It is a good example of how to use the library in practice, and the supporting code can be found in the `projects/racun` folder.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details
