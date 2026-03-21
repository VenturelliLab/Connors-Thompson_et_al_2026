# Machine learning-guided design of human gut microbiome dynamics in response to dietary fibers

**Bryce M. Connors<sup>1,2,+</sup>, Jaron Thompson<sup>1,2,+</sup>, and Ophelia S. Venturelli<sup>1,2,3,4*</sup>**

<sup>1</sup>Department of Biochemistry, University of Wisconsin-Madison, Madison, WI  
<sup>2</sup>Department of Chemical & Biological Engineering, University of Wisconsin-Madison, Madison, WI  
<sup>3</sup>Department of Bacteriology, University of Wisconsin-Madison, Madison, WI  
<sup>4</sup>Department of Biomedical Engineering, Duke University, Durham, NC  
<sup>+</sup>These authors contributed equally.  
<sup>\*</sup>Correspondence: [ophelia.venturelli@duke.edu](mailto:ophelia.venturelli@duke.edu)

---

## Overview

This repository contains the source code used in the manuscript:  
**"Machine learning-guided design of human gut microbiome dynamics in response to dietary fibers"**  

The project implements a recurrent neural network (miRNN) to model, predict, and optimize the dynamics of microbial species and metabolite concentrations in response to dietary fiber perturbations.

---

## Features

- A variational Bayesian RNN-based model (`miRNN`) to simulate microbial-metabolite dynamics.
- Alternative linear baselines (`LR`, `LR2`) for comparison.
- Evidence-based model selection and convergence checks.
- Integrated Bayesian experimental design (UCB and exploration-based strategies).
- Functions for prediction, uncertainty quantification, and design of experiments.

---

## Key Classes and Functions

miRNN

Implements a Bayesian RNN with leaky ReLU activations. Key methods:

    fit(data_scaled, ...): Fits the model using evidence optimization.

    predict(data_scaled): Returns point predictions and uncertainties.

    search_UCB(...): Selects informative experiments using upper confidence bounds.

    explore(...), exploit(...): Exploration/exploitation for experiment design.

---

## Example notebooks

**MiRNN Examples.ipynb:** A tutorial notebook that goes through the steps of preparing data, fitting a model, and plotting model predictions. 

**Design (Exploration).ipynb:** Selects a set of experiments that collectively maximize the expected information gain (pure exploration strategy does not try to optimize functions of microbial communities).

**Design (Batch UCB).ipynb:** Selects a set of experiments that explore and exploit the design space, where exploitation refers to selection of conditions that maximize a user defined objective function and exploration is measured as the expected information gain of the set of experiments. 

**Design (Thompson sampling).ipynb:** Selects a set of experiments that maximize a user defined objective function. Exploration is achieved by sampling parameters from the posterior distribution and then using the resulting model to evaluate the objective. 

**DTL 0.ipynb:** Analyzes a DTL objective and related response variables using fitted model outputs and summary statistics. 

**Identify LoComm-IPX.ipynb:** Screens the design space to identify communities or conditions associated with the LoComm-IPX target. 

**LinearRegression.ipynb:** Fits and evaluates a linear regression baseline for comparison against the miRNN model. 

**LinearRegression (2nd order).ipynb:** Extends the linear baseline with second-order terms to capture nonlinear effects and interactions.

**MCMM/MCMM_analysis.ipynb:** Analyzes community-scale metabolic model outputs using `micom` and visualization tools. 

**MiRNN KFold Improvement 0, 1, 2, 3 (32h).ipynb:** Evaluates miRNN performance across cross-validation folds and compares improvement across configurations. 

**MiRNN Predict Design Space.ipynb:** Uses the trained miRNN to generate predictions across the experimental design space. 

**MiRNN Predict Full Design Space.ipynb:** Expands design-space prediction to the full set of candidate conditions. 

**MiRNN Sensitivity (AC-BU-PC SHAP).ipynb:** Uses SHAP-style sensitivity analysis to interpret how features influence acetate, butyrate, and propionate predictions. 

**MiRNN Sensitivity (Butyrate SHAP).ipynb:** Focuses SHAP-based interpretation specifically on butyrate predictions. 

**MiRNN Sensitivity (Community SHAP).ipynb:** Applies sensitivity analysis to community-level predictions from the miRNN. 

**MiRNN Sensitivity (Diversity SHAP).ipynb:** Examines how input features influence predicted diversity-related outcomes using SHAP-style analysis. 

**MiRNN Sensitivity (Stability SHAP).ipynb:** Interprets miRNN predictions for stability-related outcomes with SHAP-style feature attributions. 

**Plot Design Space.ipynb:** Visualizes the predicted or measured design space to reveal trends, tradeoffs, and promising regions. 

**Plot Design Space (Kmeans).ipynb:** Visualizes the design space with k-means clustering to group similar regions or candidate conditions. 

**Plot Design Space (Kmeans-reverse).ipynb:** Provides an alternate clustered design-space visualization using a reverse k-means-based ordering or grouping. 

**Plot DTL Progress.ipynb:** Plots progress of the DTL objective over time or across design rounds. 

**Plot Kfold.ipynb:** Summarizes and visualizes cross-validation performance across folds. 

**Plot objective components.ipynb:** Breaks a composite objective into its individual components for inspection. 

**Plot objective correlations.ipynb:** Examines correlations among objective terms or related outputs in the design space. 