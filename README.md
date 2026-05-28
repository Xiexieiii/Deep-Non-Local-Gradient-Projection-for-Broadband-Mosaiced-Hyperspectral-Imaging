# Deep Non-Local Gradient Projection for Broadband Mosaiced Hyperspectral Imaging

Official PyTorch implementation of the paper **"Deep Non-Local Gradient Projection for Broadband Mosaiced Hyperspectral Imaging"**.

---

## 📝 Overview

This project addresses the challenging task of fusing **broadband mosaiced hyperspectral images (HSI)** and **high-resolution panchromatic (PAN) images**. While Broadband Mosaiced Spectral Filter Arrays (BMSFAs) improve optical throughput, they suffer from highly overlapped, non-uniform spectral coupling and spatial degradation. 

To break these limitations, we propose a novel Deep Unfolding Framework that bridges physical imaging models with advanced learnable priors.

### ✨ Key Contributions
* **Graph-Enhanced Gradient Mapping:** Unlike standard convolutions that fail on sparse/non-uniform residuals, our module dynamically builds similarity graphs to enable **global feature propagation** over long-range structures.
* **Detail-Aware Message Injection:** A dedicated mechanism designed to inject explicit detail information into the graph, successfully mitigating the common **over-smoothing** problem during graph propagation.
* **State-of-the-Art Performance:** Extensive experiments on benchmark datasets demonstrate superior performance over existing SOTA methods in both spectral and spatial fidelity.

---

## 📑 Paper Information

### Authors
- **Yunyu Xie**
- **Renwei Dian** (Corresponding Author), Member, IEEE
- **Nan Wang**
- **Lishan Tan**
- **Shutao Li**, Fellow, IEEE

### Affiliations
* **School of Artificial Intelligence and Robotics**, Hunan University, Changsha 410082, China
* **Innovation Institute of Industrial Design and Machine Intelligence Quanzhou-Hunan University**, Hunan University, Quanzhou 362006, China

### Keywords
* Broadband mosaiced imaging    
* Fusion imaging
* Deep unfolding networks
* Graph neural network

---

## 💻 Code

The official implementation of this work is currently being organized and will be made publicly available after further preparation. 

Please stay tuned! ⭐

---

## ✉️ Contact

For questions about the paper or code, please feel free to contact the authors via email:

- **Yunyu Xie**: [yyxie@hnu.edu.cn](mailto:yyxie@hnu.edu.cn)
- **Renwei Dian**: [drw@hnu.edu.cn](mailto:drw@hnu.edu.cn)
- **Nan Wang**: [wangn@hnu.edu.cn](mailto:wangn@hnu.edu.cn)
- **Lishan Tan**: [LishanTan@hnu.edu.cn](mailto:LishanTan@hnu.edu.cn)
- **Shutao Li**: [shutao_li@hnu.edu.cn](mailto:shutao_li@hnu.edu.cn)

> 💡 *Note: For privacy and security reasons, phone numbers are not recommended to be listed in a public GitHub repository. Please contact the authors via email.*
