import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import precision_recall_curve, auc

# 读取数据
data = pd.read_csv('zhh_b.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)

# 标准化特征值
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建KNN分类器
knn = KNeighborsClassifier(n_neighbors=1)  # 默认k=3
knn.fit(X_train, y_train)

# 预测
y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)[:, 1]  # 用于计算AUROC和AUPRC

# 计算评价指标
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)  # 敏感度（召回率）
specificity = tn / (tn + fp)
f1 = f1_score(y_test, y_pred)
auroc = roc_auc_score(y_test, y_proba)
precision, recall, _ = precision_recall_curve(y_test, y_proba)
auprc = auc(recall, precision)

# 输出结果
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"AUROC: {auroc:.4f}")
print(f"AUPRC: {auprc:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)