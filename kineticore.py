import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# 1. SIMULATING REAL HEALTH DATA
# Features: [Heart Rate, Wrist Movement, Body Acceleration]
# We want to detect: 0 = Normal/Activity, 1 = Potential Health Risk (e.g., Tachycardia without movement)

# Normal: High Body Accel + High HR (Exercise) or Low Accel + Low HR (Rest)
# Risk: High HR + NO Body Accel (Panic/Arrhythmia)
def generate_health_data(n=1000):
    np.random.seed(42)
    hr = np.random.uniform(60, 160, n)
    wrist_move = np.random.uniform(0, 1, n)
    body_accel = np.random.uniform(0, 1, n)

    y = []
    for i in range(n):
        # Logic: If HR is high (>120) but body isn't moving (<0.3), it's a risk
        if hr[i] > 120 and body_accel[i] < 0.3:
            y.append(1)
        else:
            y.append(0)
    return np.c_[hr, wrist_move, body_accel], np.array(y)


X, y = generate_health_data()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. THE MLP ARCHITECTURE
# We use (10, 10) to capture the non-linear "Risk Zone"
mlp = MLPClassifier(hidden_layer_sizes=(10, 10),
                    activation='relu',
                    max_iter=2000,
                    random_state=42)
mlp.fit(X_scaled, y)

# 3. VISUALIZATION (Simplifying to HR vs Body Accel for the plot)
h = .5
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 2].min() - 0.1, X[:, 2].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, 0.01))

# We keep wrist movement constant at 0.5 for the visualization
constant_wrist = np.full(xx.ravel().shape, 0.5)
grid_points = np.c_[xx.ravel(), constant_wrist, yy.ravel()]
grid_scaled = scaler.transform(grid_points)

Z = mlp.predict(grid_scaled)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn_r')
plt.scatter(X[:, 0], X[:, 2], c=y, cmap='RdYlGn_r', edgecolors='k', s=20)
plt.title("KinetiCore: Detecting Health Risk (Heart Rate vs. Body Movement)")
plt.xlabel("Heart Rate (BPM)")
plt.ylabel("Full-Body Acceleration")
plt.show()