```md
# 🏥 Diabetes Prediction Using Single-Layer Perceptron

## 📌 Overview
This project explores the **Single-Layer Perceptron (SLP) algorithm** for **predicting diabetes** using the **Pima Indians Diabetes dataset**. The study focuses on how hyperparameters, gradient descent methods, weight initialization, and activation functions affect model performance.

## 🔬 Key Features
✅ **Implemented a Single-Layer Perceptron for binary classification**  
✅ **Hyperparameter tuning: Learning rate, epochs, batch size**  
✅ **Compared Mini-Batch Gradient Descent vs. Stochastic Gradient Descent (SGD)**  
✅ **Experimented with random weight initialization**  
✅ **Tested Sigmoid vs. Step Activation Functions**  
✅ **Stratified K-Fold Cross-Validation for evaluation**  

## 🛠 Tech Stack
- **Programming:** Python  
- **Frameworks:** PyTorch  
- **Libraries:** NumPy, Pandas, Scikit-learn, Matplotlib  
- **Techniques:** Perceptron Algorithm, Mini-batch Gradient Descent  

## 📊 Results
| Experiment | Learning Rate | Epochs | F1 Score |
|------------|--------------|--------|----------|
| Baseline | 0.01 | 10 | 46.31% |
| **Best Model (Mini-Batch Gradient Descent)** | 0.001 | 100 | **61.54%** |

## 🚀 How to Run
```bash
git clone https://github.com/Pamalrojitha/Diabetes_Prediction_SLP.git
cd Diabetes_Prediction_SLP
pip install -r requirements.txt
jupyter notebook
