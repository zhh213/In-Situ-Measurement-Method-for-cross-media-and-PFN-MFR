from tabpfn import TabPFNClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data = pd.read_csv('zhh_b.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = TabPFNClassifier()
clf.fit(X_train, y_train)
prediction_probabilities = clf.predict_proba(X_test)
auroc = roc_auc_score(y_test, prediction_probabilities[:, 1])
precision, recall, _ = precision_recall_curve(y_test, prediction_probabilities[:, 1])
auprc = auc(recall, precision)
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
recall = recall_score(y_test, predictions)
sensitivity = recall

tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
specificity = tn / (tn + fp)
conf_matrix = confusion_matrix(y_test, predictions)

print("AUROC:", auroc)
print("AUPRC:", auprc)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Sensitivity/Recall:", sensitivity)
print("Specificity:", specificity)
print("conf_matrix:", conf_matrix)