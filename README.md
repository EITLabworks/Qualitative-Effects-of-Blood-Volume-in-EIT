# Normalization of Electrical Impedance Tomography Data featuring Blood Volume Changes
Normalization of Electrical Impedance Tomography (EIT) Data featuring Blood Volume Changes characterized by a water tank measurement.

This repository contains algorithms, training routines and results for an analysis of preproccessing EIT data featuring emulated stroke volume changes from a water tank experiment. 


----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------
__Abstract:__   
The assessment, targeting and monitoring of central hemodynamic measures in clinical situations has been shown to improve patient safety and lower mortality. Aortic pressure
 represents a key parameter that can indicate various clinical conditions [1]. Since the gold standard method is highly invasive and non-invasive techniques show no long-term
 capabilities, inaccurate measurements, and a need for invasive calibrations, the demand for non-invasive alternatives remains [1]. Electrical impedance tomography (EIT) has the potential
 to capture hemodynamic changes like aortic pressure in the body, even with ventilation and noise superimposing on heart-based activity. The aim is to create a neural network that maps
 the EIT voltages to the concurrent aortic pressures employing a channel charting forward function. An integral part of neural networks is the creation of a low-dimensional latent space.
 Channel charting is a semi-supervised learning algorithm usually applied in geolocation. It maps a high-dimensional feature vector onto a low-dimensional hyperspace such that
 the distance relations between data points are preserved [2]. The design of distance measures and feature extraction are key challenges for this projection. The channel charting
 forward function is implemented using a neural network which has to be adequately configured [2]. This approach was conducted with data from an in-vivo animal model study,
 in which EIT voltages with a thorax belt and invasive aortic pressures were recorded simultaneously. To ensure good data quality, the recordings were synchronized, segmented, and
 calibrated and an anomaly detection was performed, which removed disturbed data sections. To eliminate regularization effects, the raw EIT voltages are used as high-dimensional
 input data. An implemented pipeline with a dissimilarity based network shows the feasibility of estimating aortic pressures from channel charted EIT voltages.
 
 REFERENCES
 

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------




## Overview:
- [Structure](#structure)
- [Installation](#installation)
- [Evaluation Results](#evaluation)
- [Author](#author)



## Structure
This repository includes:



## Installation
Clone the repository:
```bash
 git clone https://github.com/EITLabworks/Qualitative-Effects-Blood-Volume-EIT.git
```



## Evaluation Results
### 4 Exemplary Validation Curves
![Exemplary CAP Curves.](nn/models/20250612-124551/ValidationReconstructedSignalPart24Plot.png)

### 4 Exemplary Testing Curves
![Exemplary CAP Curves.](nn/models/20250612-124551/TestingReconstructedSignal4Plot.png)

### Testing Min Mean Max 
![Exemplary CAP Curves.](nn/models/20250612-124551/TestCurvePaperPredictionErrorMinMaxMean.png)




## Author
This repository is created by Patricia Fuchs, Institute of Communications Engineering, University of Rostock, Germany.   
The research is based on the paper "Normalization of Electrical Impedance Tomography Data featuring Blood Volume Changes" for the "Annual International Conference of the IEEE Engineering in Medicine and Biology Society" (EMBC) 2026.    
For questions, please contact: pat.fuchs@uni-rostock.de
