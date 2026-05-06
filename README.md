# KinetiCore: Multi-Modal Health Risk Classifier

## **Project Overview**
KinetiCore is a non-linear classification engine designed to solve a complex "Contextual Anomaly" problem in wearable health technology. The goal is to distinguish between physiological heart rate spikes caused by physical exertion (Safe) versus spikes occurring during stasis (Medical Risk).

## **The Intelligence Architecture (Neural Network)**
Traditional linear models fail when features are interdependent in a non-linear fashion. KinetiCore utilizes a **Multi-Layer Perceptron (MLP)** to "bend" the mathematical decision boundary.

### **1. Hidden Layer Topography**
* **Architecture:** Two hidden layers of 10 neurons each.
* **Function:** This depth allows the model to learn feature interactions (e.g., Heart Rate is only an anomaly if Body Acceleration is low).

### **2. ReLU Activation Function**
* **The Math:** Rectified Linear Unit ($f(x) = max(0, x)$).
* **Benefit:** By maintaining a constant gradient of $1.0$ for positive inputs, ReLU prevents the **Vanishing Gradient** problem during backpropagation, ensuring efficient training.

## **Data Architecture: Multi-Modal Sensing**
The model processes a three-dimensional feature space to reduce "False Positives" (e.g., elevated heart rate from household chores):
1.  **Heart Rate (BPM):** Primary physiological signal.
2.  **Wrist Movement:** Rotational data to identify local extremity noise.
3.  **Full-Body Acceleration:** Linear force used to determine actual physical mass movement.

http://googleusercontent.com/image_content/208



## **Technical Implementation Highlights**
* **Stochastic Feature Interaction:** The AI was trained to understand that high output (Heart Rate) requires high input (Acceleration), treating high output *without* input as a danger signal.
* **Decision Boundary Mapping:** Using a synthetic meshgrid "Surveyor," the project visualizes the 2D "slice" of a 3D mathematical world, showing where the AI draws the risk threshold.
* **Standardized Optimization:** Implemented `StandardScaler` to ensure gradients converge by normalizing Heart Rate (60–160) and Acceleration (0–1).

## **Technical Stack**
* **Language:** Python
* **Libraries:** Scikit-Learn (MLPClassifier), NumPy, Matplotlib
* **Methodology:** Deep Learning, Feature Engineering, Non-Linear Classification
