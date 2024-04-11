# XGBoost需要对标签也进行编码
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

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

# 转换数据集为XGBoost的DMatrix格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'multi:softmax',  # 多分类问题
    'num_class': len(le.classes_),  # 类别数
    'max_depth': 3,  # 减小树的最大深度
    'min_child_weight': 3,  # 增大叶子节点的最小权重和
    'subsample': 0.8,  # 控制每棵树随机采样的比例
    'colsample_bytree': 0.8,  # 控制每棵树随机采样的列数的占比
    'eta': 0.1,  # 减小步长
    'eval_metric': 'mlogloss'  # 使用mlogloss评估指标
}
# 使用交叉验证选择最佳参数
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=100,
    nfold=5,
    metrics='mlogloss',
    early_stopping_rounds=10
)

# 使用最佳参数训练模型
best_num_round = cv_results.shape[0] - 1
bst = xgb.train(params, dtrain, num_boost_round=best_num_round)

# 预测
y_pred = bst.predict(dtest)
print('【---------测试集----------】')
# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# 将标签转换为二进制形式（unacc为0，其他为1）
y_test_binary = np.where(y_test == le.transform(['unacc'])[0], 0, 1)
y_pred_binary = np.where(y_pred == le.transform(['unacc'])[0], 0, 1)

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
    val_df[feature] = val_df[feature].astype('category').cat.codes
# 使用与训练集相同的编码器对验证集的标签进行编码
val_df['label'] = le.transform(val_df['label'])
X_val = val_df[features]
y_val = val_df['label']
dval = xgb.DMatrix(X_val, label=y_val)
# 在验证集上预测
y_pred_val = bst.predict(dval)
# 计算准确率
accuracy_val = metrics.accuracy_score(y_val, y_pred_val)
print(f"验证集Accuracy: {accuracy_val}")

# 将标签转换为二进制形式（unacc为0，其他为1）
y_val_binary = np.where(y_val == le.transform(['unacc'])[0], 0, 1)
y_pred_val_binary = np.where(y_pred_val == le.transform(['unacc'])[0], 0, 1)

b_accuracy_val = metrics.accuracy_score(y_val_binary, y_pred_val_binary)
print(f"正反例的Accuracy: {b_accuracy_val}")
# 计算召回率、精确率和F1-Score
b_recall_val = metrics.recall_score(y_val_binary, y_pred_val_binary)
b_precision_val = metrics.precision_score(y_val_binary, y_pred_val_binary)
b_f1_val = metrics.f1_score(y_val_binary, y_pred_val_binary)
print(f"正反例:召回率(Recall): {b_recall_val}")
print(f"正反例:精确率(Precision): {b_precision_val}")
print(f"正反例:F1-Score: {b_f1_val}")

# 计算验证集的ROC曲线所需的假阳性率和真阳性率
fpr_val, tpr_val, _ = metrics.roc_curve(y_val_binary, y_pred_val_binary)

# 计算验证集的AUC
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


