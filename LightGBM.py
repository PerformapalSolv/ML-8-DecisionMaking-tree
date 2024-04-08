#  LightGBM需要对标签也进行编码
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np



# 正确读取数据到 Pandas DataFrame
df = pd.read_csv('dataset/car_1000.txt', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'])


# 将特征值转换为数值
features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
for feature in features:
    df[feature] = df[feature].astype('category').cat.codes
# 将标签列进行编码
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# 将数据集分为训练集和测试集
X = df[features]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 转换数据集为LightGBM的Dataset格式
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# 设置参数
params = {
    'objective': 'multiclass',
    'num_class': 4,
    'boosting_type': 'gbdt',
    'metric': 'multi_logloss',
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 0.9
}


# 训练模型
num_round = 700
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])

# 预测
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
y_pred = [list(x).index(max(x)) for x in y_pred]

# 评估模型
print("Accuracy:", accuracy_score(y_test, y_pred))

# 将标签转换为二进制形式（unacc为0，其他为1）
y_test_binary = np.where(y_test == le.transform(['unacc'])[0], 0, 1)
y_pred_binary = np.where(y_pred == le.transform(['unacc'])[0], 0, 1)
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
