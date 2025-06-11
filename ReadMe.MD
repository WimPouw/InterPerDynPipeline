# An Open-source Standardized Pipeline for Equitable Observations of Interactive Behavioral Dynamics
![Motion tracking example](./images/ts.gif)

## What This Pipeline Does

This pipeline enables researchers to:
- **Track human movement** from top-view video recordings using YOLOv8 pose estimation
- **Extract behavioral time-varying variables** like approach/avoidance patterns  
- **Extract features** like smoothness to understand social interactive qualities
- **Analyze interpersonal dynamics** using bnon-linear time series methods (CRQA)
- **Ensure privacy** through video masking techniques
- **Reproduce results** with your own data

## 🚀 Getting Started

**👉 [Complete Installation & Usage Guide →](https://wimpouw.github.io/InterPerDynPipeline/)**

All installation instructions, dependencies, step-by-step tutorials, and code examples are available in our comprehensive documentation.

### Behavioral Analysis
- **Smoothness metrics**: Quantify coordination quality using spectral arc length and dimensionless jerk
- **Approach dynamics**: Measure proximity changes and mutual positioning
- **Cross-recurrence analysis**: Assess coupling strength between interaction partners

## 📁 Repository Structure

```
InterPerDynPipeline/
├── code_STEP1_posetrackingprocessing/    # Motion tracking scripts
├── code_STEP2_smoothness_features/       # Feature extraction
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
  journal={[Journal Name]},
  year={2024}
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
