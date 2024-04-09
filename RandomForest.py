import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


# 正确读取数据到 Pandas DataFrame
df = pd.read_csv('dataset/car_1000.txt')

# 假设数据文件已经是正确的格式，列名也正确
# 如果列名不对，可以在 read_csv 中使用 names 参数指定列名
df = pd.read_csv('dataset/car_1000.txt', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'])
# 将特征值转换为数值
features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
for feature in features:
    df[feature] = df[feature].astype('category').cat.codes


# 将数据集分为训练集和测试集
X = df[features]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用随机森林进行决策分类
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

# 在测试集上预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 将标签转换为二进制形式（unacc为0，其他为1）
y_test_binary = np.where(y_test == 'unacc', 0, 1)
y_pred_binary = np.where(np.array(y_pred) == 'unacc', 0, 1)

b_accuracy = metrics.accuracy_score(y_test_binary, y_pred_binary)
print(f"正反例的Accuracy: {accuracy}")
# 计算召回率、精确率和F1-Score
b_recall = metrics.recall_score(y_test_binary, y_pred_binary)
b_precision = metrics.precision_score(y_test_binary, y_pred_binary)
b_f1 = metrics.f1_score(y_test_binary, y_pred_binary)
print(f"正反例:召回率(Recall): {b_recall}")
print(f"正反例:精确率(Precision): {b_precision}")
print(f"正反例:F1-Score: {b_f1}")

# 计算ROC曲线所需的假阳性率和真阳性率
fpr, tpr, _ = metrics.roc_curve(y_test_binary, y_pred_binary)

# 计算AUC（ROC曲线下的面积）
roc_auc = metrics.auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
