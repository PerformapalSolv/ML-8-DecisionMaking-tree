from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

class CARTDecisionTree:
    def __init__(self, min_samples_split=2, min_impurity_decrease=0.0):
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None
        self.feature_names = None

    def fit(self, X, y):
        self.feature_names = X.columns
        X = X.values
        y = y.values
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return y[0]

        if X.shape[0] < self.min_samples_split:
            return Counter(y).most_common(1)[0][0]

        best_feature, best_threshold = self._choose_best_split(X, y)
        if best_feature is None:
            return Counter(y).most_common(1)[0][0]

        feature_name = self.feature_names[best_feature]
        tree = {feature_name: {}}
        left_mask = X[:, best_feature] < best_threshold
        right_mask = X[:, best_feature] >= best_threshold
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        tree[feature_name]['< ' + str(best_threshold)] = self._build_tree(X_left, y_left)
        tree[feature_name]['>= ' + str(best_threshold)] = self._build_tree(X_right, y_right)
        return tree

    def _choose_best_split(self, X, y):
        best_gini = 1.0
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gini = self._gini_impurity(X, y, feature, threshold)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        if 1 - best_gini < self.min_impurity_decrease:
            return None, None

        return best_feature, best_threshold

    def _gini_impurity(self, X, y, feature, threshold):
        left_mask = X[:, feature] < threshold
        right_mask = X[:, feature] >= threshold
        n_left = left_mask.sum()
        n_right = right_mask.sum()
        if n_left == 0 or n_right == 0:
            return 1.0

        _, left_counts = np.unique(y[left_mask], return_counts=True)
        _, right_counts = np.unique(y[right_mask], return_counts=True)
        gini_left = 1 - np.sum((left_counts / n_left) ** 2)
        gini_right = 1 - np.sum((right_counts / n_right) ** 2)
        gini = (n_left * gini_left + n_right * gini_right) / (n_left + n_right)
        return gini

    def predict(self, X):
        X = X.values
        return [self._traverse_tree(x, self.tree) for x in X]

    def _traverse_tree(self, x, node):
        if not isinstance(node, dict):
            return node

        feature_name = list(node.keys())[0]
        feature = self.feature_names.get_loc(feature_name)
        thresholds = list(node[feature_name].keys())
        threshold_left = [t for t in thresholds if t.startswith('<')][0]
        threshold_right = [t for t in thresholds if t.startswith('>=')][0]

        if x[feature] < float(threshold_left.split(' ')[1]):
            return self._traverse_tree(x, node[feature_name][threshold_left])
        else:
            return self._traverse_tree(x, node[feature_name][threshold_right])


df = pd.read_csv('dataset/car_1000.txt', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'])
# 将特征值转换为数值
features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
for feature in features:
    df[feature] = df[feature].astype('category').cat.codes

# 将数据集分为训练集和测试集
X = df[features]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = CARTDecisionTree(min_samples_split=2, min_impurity_decrease=0.0)
clf.fit(X_train, y_train)

# 在测试集上预测
y_pred = clf.predict(X_test)

print('【---------测试集----------】')
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
plt.plot(fpr, tpr, color='darkorange', lw=2, label='test set:ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('test set:Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()



print('【---------验证集----------】')
# 泛化误差计算
val_df = pd.read_csv('dataset/val.csv', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'])
# 将特征值转换为数值
features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
for feature in features:
    val_df[feature] = df[feature].astype('category').cat.codes

X_val = df[features]
y_val = df['label']
# 在验证集上预测
y_pred = clf.predict(X_val)
# 计算准确率
accuracy = metrics.accuracy_score(y_val, y_pred)
print(f"验证集Accuracy: {accuracy}")

# 将标签转换为二进制形式（unacc为0，其他为1）
y_val_binary = np.where(y_val == 'unacc', 0, 1)
y_pred_binary = np.where(np.array(y_pred) == 'unacc', 0, 1)

b_accuracy = metrics.accuracy_score(y_val_binary, y_pred_binary)
print(f"正反例的Accuracy: {accuracy}")
# 计算召回率、精确率和F1-Score
b_recall = metrics.recall_score(y_val_binary, y_pred_binary)
b_precision = metrics.precision_score(y_val_binary, y_pred_binary)
b_f1 = metrics.f1_score(y_val_binary, y_pred_binary)
print(f"正反例:召回率(Recall): {b_recall}")
print(f"正反例:精确率(Precision): {b_precision}")
print(f"正反例:F1-Score: {b_f1}")

# 计算ROC曲线所需的假阳性率和真阳性率
fpr_val, tpr_val, _ = metrics.roc_curve(y_val_binary, y_pred_binary)

# 计算AUC（ROC曲线下的面积）
roc_auc_val = metrics.auc(fpr_val, tpr_val)

# 绘制测试集和验证集的ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Test set: ROC curve (area = %0.2f)' % roc_auc)
plt.plot(fpr_val, tpr_val, color='blue', lw=2, label='Val set: ROC curve (area = %0.2f)' % roc_auc_val)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

