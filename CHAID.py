from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from collections import Counter
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt


class CHAIDDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, alpha=0.05):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.tree = None
        self.feature_names = None

    def fit(self, X, y):
        self.feature_names = X.columns
        X = X.values
        y = y.values
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if self.max_depth is not None and depth >= self.max_depth:
            return Counter(y).most_common(1)[0][0]

        if len(np.unique(y)) == 1:
            return y[0]

        if X.shape[0] < self.min_samples_split or X.shape[0] < 2 * self.min_samples_leaf:
            return Counter(y).most_common(1)[0][0]

        best_feature, best_splits = self._choose_best_split(X, y)
        if best_feature is None:
            return Counter(y).most_common(1)[0][0]

        feature_name = self.feature_names[best_feature]
        tree = {feature_name: {}}
        for split_value, mask in best_splits.items():
            X_subset, y_subset = X[mask], y[mask]
            if X_subset.shape[0] < self.min_samples_leaf:
                tree[feature_name][split_value] = Counter(y_subset).most_common(1)[0][0]
            else:
                tree[feature_name][split_value] = self._build_tree(X_subset, y_subset, depth + 1)
        return tree

    def _choose_best_split(self, X, y):
        best_feature = None
        best_splits = None
        best_p_value = 1.0

        for feature in range(X.shape[1]):
            unique_values = np.unique(X[:, feature])
            if len(unique_values) <= 1:
                continue

            contingency_table = pd.crosstab(X[:, feature], y)
            chi2, p_value, _, _ = chi2_contingency(contingency_table)

            if p_value < best_p_value and p_value <= self.alpha:
                best_p_value = p_value
                best_feature = feature
                best_splits = {value: (X[:, feature] == value) for value in unique_values}

        return best_feature, best_splits

    def predict(self, X):
        X = X.values
        return [self._traverse_tree(x, self.tree) for x in X]

    def _traverse_tree(self, x, node):
        if not isinstance(node, dict):
            return node

        feature_name = list(node.keys())[0]
        feature = self.feature_names.get_loc(feature_name)
        split_values = list(node[feature_name].keys())

        for split_value in split_values:
            if x[feature] == split_value:
                return self._traverse_tree(x, node[feature_name][split_value])



df = pd.read_csv('dataset/car_1000.txt', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'])
# 将特征值转换为数值
features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
for feature in features:
    df[feature] = df[feature].astype('category').cat.codes

# 将数据集分为训练集和测试集
X = df[features]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = CHAIDDecisionTree(max_depth=None, min_samples_split=2, min_samples_leaf=1, alpha=0.05)
clf.fit(X_train, y_train)

# 在测试集上预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 将标签转换为二进制形式（unacc为0，其他为1）
y_test_binary = np.where(y_test == 'unacc', 0, 1)
y_pred_binary = np.where(np.array(y_pred) == 'unacc', 0, 1)
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