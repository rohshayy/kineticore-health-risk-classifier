# KinetiCore: Multi-Modal Health Risk Classifier

## **Project Overview**
KinetiCore is a non-linear anomaly detection system designed for wearable health technology. It addresses a critical challenge in physiological monitoring: distinguishing between high heart rate caused by physical exertion (Safe) and high heart rate occurring during stasis (Medical Risk). 

By utilizing a **Multi-Layer Perceptron (MLP)**, the system learns complex decision boundaries that linear models cannot capture, reducing false positives in emergency alert systems.

## **The Intelligence Architecture (MLP)**
The engine implements a Deep Learning approach to solve a multi-modal classification problem.
* **Hidden Layers:** A dual-layer architecture (10, 10) allows the model to perform high-dimensional coordinate transformations to identify "Risk Zones."
* **ReLU Activation:** Utilizes the **Rectified Linear Unit** function to maintain a constant gradient of $1.0$ for positive inputs, effectively mitigating the **Vanishing Gradient** problem common in deep networks.
* **Non-Linear Decision Boundaries:** Unlike Logistic Regression, which draws a straight line, KinetiCore "bends" its mathematical space to isolate specific clusters of "High HR + Low Acceleration."



## **Multi-Modal Data Engineering**
To simulate a sophisticated smartwatch environment, the model processes three input dimensions:
1.  **Heart Rate (BPM):** The primary physiological signal.
2.  **Wrist Movement:** Rotational data to filter out localized "noise" (e.g., household chores).
3.  **Full-Body Acceleration:** Linear force to confirm the user's metabolic demand.

## **Technical Implementation Highlights**
* **Contextual Anomaly Detection:** The model identifies that features are interdependent; a high heart rate is only classified as a risk if body acceleration remains below a specific threshold ($x_1 > 120 \cap x_3 < 0.3$).
* **Z-Score Normalization:** Implemented `StandardScaler` to ensure the disparate scales of BPM (60-160) and Acceleration (0-1) contribute equally to the weight updates.
* **Latent Space Visualization:** Utilizes a synthetic meshgrid to project a 2D "slice" of a 3D mathematical world, creating a visual map of the AI's decision surface.



## **Technical Stack**
* **Language:** Python
* **Libraries:** Scikit-Learn (Neural Network), NumPy, Matplotlib
* **Mathematical Concepts:** Backpropagation, Stochastic Gradient Descent, Non-linear Activation.
