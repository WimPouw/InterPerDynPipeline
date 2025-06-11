# An Open-source Standardized Pipeline for Equitable Observations of Interactive Behavioral Dynamics
![Motion tracking example](./images/ts.gif)
**Authors:**  
Arkadiusz Białek¹, Wim Pouw²,³, James Trujillo⁴, Fred Hasselman⁵, Babajide Alamu Owoyele⁶, Natalia Siekiera¹, Joanna Rączaszek-Leonardi⁷ & Travis J. Wiltshire²

¹ Institute of Psychology, Jagiellonian University  
² Department of Cognitive Science & Artificial Intelligence, Tilburg University  
³ Donders Institute for Brain, Cognition and Behavior, Radboud University Nijmegen  
⁴ Institute for Logic, Language and Computation, University of Amsterdam  
⁵ Behavioral Science Institute, Radboud University Nijmegen  
⁶ Artificial Intelligence and Intelligent Systems, Hasso Plattner Institute Potsdam  
⁷ Human Interactivity and Language Lab, Faculty of Psychology, University of Warsaw  

## What This Pipeline Does

This [pipeline](https://wimpouw.github.io/InterPerDynPipeline/) enables researchers to:
- **Track human movement** from top-view video recordings using YOLOv8 pose estimation
- **Extract behavioral time-varying variables** like approach/avoidance patterns  
- **Extract features** like smoothness to understand social interactive qualities
- **Analyze interpersonal dynamics** using bnon-linear time series methods (CRQA)
- **Ensure privacy** through video masking techniques
- **Reproduce results** with your own data

## 🚀🚀🚀 Getting Started (Quarto notebook landing page)

**👉 [Complete Installation & Usage Guide →](https://wimpouw.github.io/InterPerDynPipeline/)**

All installation instructions, dependencies, step-by-step tutorials, and code examples are available in our documentation quarto notebook: https://wimpouw.github.io/InterPerDynPipeline/.

## 📁 Repository Structure

```
InterPerDynPipeline/
├── code_STEP1_posetrackingprocessing/    # Motion tracking scripts (python)
├── code_STEP2_smoothness_features/       # Feature extraction (python)
├── code_STEP3_nonlinear_analysis/        # CRQA analysis (R)
├── code_STEP4_statisticalanalysis/       # Statistical modeling (R)
├── dataoutput_STEP*/                     # Output directories
├── meta/                                 # Metadata and documentation
└── docs/                                 # Generated documentation
```

## 🎓 Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{bialek2024interperdyn,
  title={An Open-source Standardized Pipeline for Equitable Observations of Interactive Behavioral Dynamics},
  author={Białek, Arkadiusz and Pouw, Wim and Trujillo, James and Hasselman, Fred and Owoyele, Babajide Alamu and Siekiera, Natalia and Rączaszek-Leonardi, Joanna and Wiltshire, Travis J.},
  journal={[PsyArxiv]},
  year={2025}
}
```
## 🤝 Contributing
We welcome contributions to this pipeline! Please see reach out.

## Current code contributors
https://github.com/WimPouw/InterPerDynPipeline/graphs/contributors

## Support
- **📚 Documentation**: https://wimpouw.github.io/InterPerDynPipeline/
- **✉️ Main contact for this repo**: w.pouw@tilburguniversity.edu

## 📄 License
This project is licensed under the MIT License.
