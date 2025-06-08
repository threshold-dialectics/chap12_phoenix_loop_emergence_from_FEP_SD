# Threshold Dialectics: Experiment 12A - Phoenix Loop Emergence

This repository contains the simulation code for **Experiment 12A**, as described in the book *Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness*. The experiment demonstrates the emergence of the "Phoenix Loop"—a stereotyped, four-phase recovery cycle—from a top-down System Dynamics model. It shows that characteristic diagnostic signatures appear consistently across different collapse scenarios, validating the Phoenix Loop as a generic recovery archetype within TD-structured systems.

This repository focuses on Experiment 12A. The corresponding Experiment 12B, which investigates the emergence of the Phoenix Loop in an Agent-Based Model, is located in a separate repository.

## About the Book: Threshold Dialectics

> *Threshold Dialectics* offers a fundamentally different perspective on systemic failure. It aims not merely to predict collapse, but to enable what we term **Active Robustness**—a system's capacity to proactively and dynamically maintain its viability in the face of uncertainty and change. This book argues that the path towards fragility, collapse, and subsequent recovery is governed not by isolated states, but by the intricate *dialectic*---the dynamic interplay---among core adaptive capacities. Stability erodes most rapidly, and collapse becomes imminent, not when one resource dial simply hits red, but when the *coupled drift velocities* of fundamental adaptive levers become significantly large and detrimentally correlated.
>
> --- *From the Preface*

## Key Concepts

To understand the code in this repository, it is helpful to be familiar with these core concepts from Threshold Dialectics:

*   **The Three Levers:** Core adaptive capacities inherent in any complex adaptive system.
    *   **Perception Gain ("gLever"):** The system's sensitivity to incoming information or prediction errors.
    *   **Policy Precision ("betaLever"):** The rigidity or confidence with which the system adheres to its current operational rules or strategies.
    *   **Energetic Slack ("FEcrit"):** The system's buffer or reserve capacity available to absorb shocks and fuel adaptation.
*   **The Tolerance Sheet ("ThetaT"):** A dynamic viability boundary defined by the interplay of the three levers ("ThetaT = C * g^w1 * beta^w2 * Fcrit^w3"). A system collapses when its systemic strain exceeds this tolerance.
*   **The Phoenix Loop:** A four-phase model of post-collapse recovery and reorganization.
    1.  **Phase I: Disintegration:** The initial, rapid breakdown of order following a "ThetaT" breach.
    2.  **Phase II: Flaring:** A period of high exploration and creative chaos, marked by a surge in "rhoE".
    3.  **Phase III: Pruning:** Consolidation of successful innovations and abandonment of failed ones.
    4.  **Phase IV: Restabilization:** The establishment of a new, stable operating regime.
*   **TD Diagnostics:** Key metrics used to monitor the system's dynamic state.
    *   **Speed Index ("SpeedIndex"):** The magnitude of the joint rate of change of "betaLever" and "FEcrit". High speed indicates rapid structural drift.
    *   **Couple Index ("CoupleIndex"):** The rolling correlation between the rates of change of "betaLever" and "FEcrit". Measures the synchrony of their drift.
    *   **Exploration Entropy Excess ("rhoE"):** The ratio of the system's current exploratory activity to its stable baseline. A key signature for the "Flaring" phase of the Phoenix Loop.

## Focus of this Experiment (12A)

As detailed in Chapter 12 of the book, this experiment focuses on the emergence of the Phoenix Loop from a top-down **System Dynamics model**. The primary script, "phoenix_loop_scenarios_sim.py", simulates five different collapse scenarios (e.g., Strain Shock, Resource Exhaustion). The key finding is that while the *pathway to collapse* differs in each scenario, the *post-collapse recovery sequence* consistently follows the four phases of the Phoenix Loop. The experiment validates that the diagnostic signatures, particularly the surge in "rhoE", are a robust and generic feature of recovery in a TD-structured system.

## Repository Structure

This repository contains three main Python scripts:

*   **"phoenix_loop_scenarios_sim.py"**
    *   **Purpose:** The primary script for **Experiment 12A**. It defines the "PhoenixLoopSimulator" (a System Dynamics model) and runs five distinct collapse scenarios.
    *   **Functionality:** For each scenario, it simulates the system's dynamics, calculates TD diagnostics (including the "rhoE" proxy based on "1/betaLever"), and generates detailed time-series and diagnostic trajectory plots.
    *   **Output:** Generates and displays plots for each scenario, saved to a "results/" directory.

*   **"phoenix_loop_classifier_accuracy_ML.py"**
    *   **Purpose:** A downstream application that demonstrates how the phases of the Phoenix Loop can be accurately identified using machine learning. This script corresponds to the work detailed in **Chapter 11**.
    *   **Functionality:** It simulates a "TDSystem" to generate data with ground-truth phase labels, engineers a rich set of features (including windowed means and standard deviations), and trains a "RandomForestClassifier" to identify the four recovery phases based on the TD diagnostics.
    *   **Output:** Trains and saves a model ("phoenix_rf_classifier.joblib") and a feature scaler ("phoenix_feature_scaler.joblib"). It also displays a confusion matrix and an example time-series plot showing the model's high predictive accuracy.

*   **"phoenix_loop_robustness_sim.py"**
    *   **Purpose:** A related experiment exploring a micro-foundational perspective on the "rhoE" (Exploration Entropy Excess) diagnostic. It uses a simple Agent-Based Model (ABM) to investigate how different methods of calculating exploration entropy (Shannon, variance, range) behave within a Phoenix Loop simulation.
    *   **Functionality:** It links a micro-level ABM to the macro-level "PhoenixLoopSimulatorWithABM", running simulations to compare the diagnostic signals produced by different entropy proxies.
    *   **Output:** Generates and displays plots comparing the system dynamics under different "rhoE" calculation methods.

## Installation

To run the simulations, you will need Python 3 and several scientific computing libraries. It is recommended to use a virtual environment.

1.  **Clone the repository:**
    """bash
    git clone https://github.com/your-username/threshold-dialectics-exp12a.git
    cd threshold-dialectics-exp12a
    """

2.  **Create and activate a virtual environment (optional but recommended):**
    """bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use "venv\Scripts\activate"
    """

3.  **Install the required libraries:**
    """bash
    pip install -r requirements.txt
    """

    **"requirements.txt":**
    """
    numpy
    scipy
    pandas
    scikit-learn
    matplotlib
    joblib
    """

## Usage: Running the Experiments

You can run each experiment by executing its corresponding Python script from your terminal.

### 1. Experiment 12A: Phoenix Loop Emergence in Scenarios

This script will run the five collapse scenarios and generate time-series and diagnostic trajectory plots for each, saving them to the "results/" directory.

"""bash
python phoenix_loop_scenarios_sim.py
"""
**Expected Output:** A series of Matplotlib windows will appear, displaying detailed plots for each scenario (e.g., "Strain\_Shock\_Collapse", "Beta\_Runaway\_Collapse"). The plots will also be saved as PNG files in a newly created "results" folder.

### 2. Machine Learning Classifier Training and Evaluation

This script will generate simulation data, train the Random Forest classifier, and evaluate its performance.

"""bash
python phoenix_loop_classifier_accuracy_ML.py
"""
**Expected Output:**
*   Console output detailing the simulation progress, model training, and the final classification report.
*   Two files saved to the repository root: "phoenix_rf_classifier.joblib" and "phoenix_feature_scaler.joblib".
*   A confusion matrix plot showing the classifier's high accuracy on the test set.
*   A detailed time-series plot for an example test run, comparing the ground truth phases with the ML predictions.

### 3. Entropy Robustness Simulation

This script runs the ABM-linked simulation to compare different methods for calculating "rhoE".

"""bash
python phoenix_loop_robustness_sim.py
"""
**Expected Output:** A series of Matplotlib plots will be displayed, showing the system dynamics under different entropy calculation methods (e.g., "Shannon", "Variance") and baseline choices.

## Citation

If you use or refer to this code or the concepts from Threshold Dialectics, please cite the accompanying book:

@book{pond2025threshold,
  author    = {Axel Pond},
  title     = {Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness},
  year      = {2025},
  isbn      = {978-82-693862-2-6},
  publisher = {Amazon Kindle Direct Publishing},
  url       = {https://www.thresholddialectics.com},
  note      = {Code repository: \url{https://github.com/threshold-dialectics}}
}

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

