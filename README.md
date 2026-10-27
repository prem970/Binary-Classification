## 1. Project Overview
This project demonstrates how to build, train, and evaluate a fully connected **Artificial Neural Network (ANN)** using TensorFlow and Keras to solve a non-linear binary classification problem. We leverage the classic `make_moons` toy dataset to illustrate how a multi-layer perceptron constructs a highly accurate curved decision boundary to separate interleaved classes, a task that linear classifiers (like standard logistic regression) cannot solve without manual feature engineering.

---

## 2. Dataset & Preprocessing
- **Source**: `scikit-learn`'s `make_moons` generator.
- **Characteristics**:
  - **Samples**: 1,000 synthetic data points.
  - **Features**: 2 continuous numerical features ($X_1, X_2$).
  - **Noise**: Set to `0.2` to introduce realistic overlap between the clusters, making the classification boundary more challenging to learn.
  - **Classes**: Binary (Class 0 and Class 1).
- **Preprocessing Steps**:
  1. **Data Splitting**: Divided the dataset into **70% Training** (700 samples) and **30% Testing** (300 samples) to ensure robust validation of model generalization.
  2. **Feature Scaling**: Scaled features using **Standardization** (`StandardScaler`), centering the features to have a mean of 0 and standard deviation of 1. Scaling prevents features with larger magnitudes from dominating the gradient updates and speeds up neural network convergence.

---

## 3. Deep Learning Approach & Model Architecture
We construct a feedforward Artificial Neural Network (ANN) with two hidden layers, adding capacity for non-linear mapping using the rectified linear activation function.

### Model Architecture Summary
| Layer | Type | Output Shape | Activation Function | Neurons / Units | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | Input | `(None, 2)` | N/A | 2 | Accepts scaled features $X_1$ and $X_2$ |
| **Hidden 1** | Dense | `(None, 16)` | `ReLU` | 16 | Extracts higher-dimensional spatial features |
| **Hidden 2** | Dense | `(None, 8)` | `ReLU` | 8 | Refines features into lower-dimensional space |
| **Output** | Dense | `(None, 1)` | `Sigmoid` | 1 | Outputs probability of class membership $P(Y=1|X)$ |

### Optimization & Compilation Configuration
- **Loss Function**: **Binary Cross-Entropy** (`binary_crossentropy`), standard for binary classification tasks.
- **Optimizer**: **Adam** (Adaptive Moment Estimation) with default learning rate for fast, stable backpropagation updates.
- **Metrics**: Evaluated using **Accuracy**.
- **Training Regime**: Trained for **50 epochs** with a **batch size of 32** and a **20% validation split** from the training set to monitor validation loss/accuracy trends and detect overfitting early.

---

## 4. Key Performance Results
- **Training Stability**: Loss smoothly decreased while training and validation accuracy steadily increased, demonstrating stable learning without severe overfitting.
- **Test Accuracy**: **96.67%** — indicates excellent generalization to unseen test data under noisy conditions.
- **Decision Boundary Visualization**: 
  - The model effectively generated a smooth, curved S-shaped decision boundary.
  - It successfully minimized classification errors even in the overlapping regions created by the `0.2` noise factor.
