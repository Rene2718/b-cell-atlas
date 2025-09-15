# 🧬 B Cell Atlas Dashboard (Beta)

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/Dash-2.0+-blueviolet)](https://dash.plotly.com/)
An interactive web application for exploring single-cell transcriptomic data from in vitro differentiated B cells. Built with [Dash](https://plotly.com/dash/) for rapid visualization and filtering by metadata and gene expression. >>> https://web-production-5e9e.up.railway.app/

---

## 🔍 Features

* UMAP visualization of single-cell clusters
* Metadata-based coloring and filtering (e.g., phenotype, isotype)
* Gene expression UMAP overlays
* Dot plots showing average expression and percent-positive across groups
* Notes and contextual descriptions for selected groups
* Fully responsive layout and custom styling

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 📘 Notes

* Immunoglobulin (IG) genes were removed prior to analysis to prevent their high expression from dominating clustering.
* All cells were derived from PBMC-isolated B cells and differentiated in vitro across multiple time points and donors.

---

## 💬 Feedback

This is a beta release.
Please send comments or questions to: [**rene271828@gmail.com**](mailto:rene271828@gmail.com).

---

## 🧠 Acknowledgments

Developed by **Rene Cheng Gibson**
Seattle Children’s Research Institute — *James & Rawlings Labs*
Data visualizations powered by [Plotly Dash](https://plotly.com/dash/)
