Breast Cancer Binary Classifier using Logistic Regression

Tools & Libraries

Python
Pandas, NumPy
Scikit-learn
Matplotlib, Seaborn
Steps Performed

Load & Clean Data:

Removed id and empty column.
Encoded target (M = 1, B = 0).
Train/Test Split:

80% training, 20% testing split.
Standardization:

Scaled features using StandardScaler.
Model Training:

Trained LogisticRegression() from scikit-learn.
Evaluation:

Accuracy: 97%
Confusion Matrix
Precision, Recall, F1-Score
ROC-AUC Score: ~0.997 (Excellent!)
Visualization:

ROC Curve
Sigmoid Function Plot
Confusion Matrix Heatmap
How Logistic Regression Works

It uses a sigmoid function to turn any number into a probability between 0 and 1.
Predictions above 0.5 → Class 1 (Malignant), below → Class 0 (Benign).
Threshold can be adjusted based on precision/recall needs.
