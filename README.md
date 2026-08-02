# Classical Machine Learning & Reinforcement Learning Implementations

## 📖 In-Depth Project Overview
This repository contains a structured, hands-on study of classical **Machine Learning algorithms** and **Reinforcement Learning** agents. Built using **scikit-learn**, **TensorFlow/Keras**, and **OpenAI Gym**, the project covers foundational models such as multiple linear regression, supervised classifiers (K-Nearest Neighbors and Logistic Regression), ensemble models (Random Forests), and Deep Q-Network (DQN) agents trained to play Atari arcade video games. 

---

## 💡 Problem It Solves
Many introductory machine learning resources are either entirely theoretical or hide the core model training mechanics inside complex, production-oriented wrapper frameworks. This makes it difficult for students to:
1. **Understand Model Baselines**: Seeing how a basic regression or classifier initializes, fits, and evaluates on standard datasets.
2. **Bridge Ensembles and Simple Classifiers**: Transitioning from single classifiers (like KNN) to complex ensembles (like Random Forests) on high-dimensional data (MNIST).
3. **Bridge Supervised and Reinforcement Learning**: Transitioning from passive classification (predicting digits) to active agents (DQN) interacting with a dynamic gaming environment.

This project solves this by presenting standalone, self-documenting Jupyter Notebooks that guide you through data loading, feature scaling, model training, evaluation metrics, and reinforcement learning policy loops.

---

## 🚀 How It Is Useful
* **Algorithm Blueprints**: Serves as a reference sheet for implementing and tuning scikit-learn models (linear regression, KNN, logistic regression, random forests).
* **Atari Agent Training Sandbox**: Provides a complete template for configuring a Deep Q-Network (DQN) policy, memory replay buffer, and Atari simulation environment (`SpaceInvaders-v0`).
* **Educational Codebase**: Ideal for machine learning students wanting to review standard datasets (Iris, handwritten digits, MNIST) and analyze performance metrics (confusion matrices, classification reports).

---

## 🗺️ Chronological Notebook Map

### [01_MultipleLinearRegression.ipynb](file:///Users/krishayg/Projects/Machine-Learning/notebooks/01_MultipleLinearRegression.ipynb)
* **Goal**: Implementing and evaluating a Multiple Linear Regression model.
* **Key Concepts**: Feature weighting, coefficient analysis, mean squared error (MSE), and coefficient of determination ($R^2$).

### [02_KNearestNeighbors.ipynb](file:///Users/krishayg/Projects/Machine-Learning/notebooks/02_KNearestNeighbors.ipynb)
* **Goal**: Classifying species on the classical Iris dataset using K-Nearest Neighbors (KNN).
* **Key Concepts**: Distance metrics, choosing the optimal $k$ hyperparameter, and decision boundary visualization.

### [03_LogisticRegression.ipynb](file:///Users/krishayg/Projects/Machine-Learning/notebooks/03_LogisticRegression.ipynb)
* **Goal**: Classifying handwritten digit images using Logistic Regression.
* **Key Concepts**: Multi-class classification, feature scaling, confusion matrices, precision, recall, and F1-score evaluation.

### [04_RandomForests.ipynb](file:///Users/krishayg/Projects/Machine-Learning/notebooks/04_RandomForests.ipynb)
* **Goal**: Training a Random Forest Ensemble Classifier on high-dimensional MNIST digit datasets.
* **Key Concepts**: Decision Trees, bootstrap aggregating (bagging), feature importance estimation, and ensemble generalization.

### [05_ReinforcementLearning.ipynb](file:///Users/krishayg/Projects/Machine-Learning/notebooks/05_ReinforcementLearning.ipynb)
* **Goal**: Training a Deep Q-Network (DQN) agent to play Atari Space Invaders.
* **Key Concepts**: Markov Decision Processes (MDP), Q-value estimation, target networks, experience replay memory, and epsilon-greedy exploration policies.

---

## 🛠️ Installation & Setup
Install the dependencies to execute the notebooks:
```bash
pip install -r requirements.txt
```
*(Key dependencies include PyTorch/TensorFlow, scikit-learn, OpenAI Gym, OpenCV, and stable-baselines/keras-rl2)*
