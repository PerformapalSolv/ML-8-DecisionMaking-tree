# README

[TOC]

## ID3决策树

ID3 使用的分类标准是信息增益，它表示得知特征 A 的信息而使得样本集合不确定性减少的程度。

数据集的信息熵：

![image-20240408235918617](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240408235918617.png)

其中 $C_{k}$ 表示集合 D 中属于第 k 类样本的样本子集。

针对某个特征 A，对于数据集 D 的条件熵 $H(D|A)$

 为：![image-20240408235846700](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240408235846700.png)

其中 $D_{i}$  表示 D 中特征 A 取第 i 个值的样本子集， $D_{ik}$表示$D_{i}$中属于第 k 类的样本子集。

信息增益 = 信息熵 - 条件熵：

![image-20240408235832461](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240408235832461.png)

信息增益越大表示使用特征 A 来划分所获得的“纯度提升越大”。

**手写代码解释:**

这里实现了ID3算法的基本思想,通过递归构建决策树,并使用信息增益作为划分标准。在预测阶段,对每个样本遍历决策树,根据特征值选择相应的分支,直到达到叶子节点,返回预测的类别。

```python
class ID3DecisionTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon  # 信息增益阈值,如果小于该值,则停止划分
        self.tree = None  # 决策树
        self.feature_names = None  # 特征名称

    def fit(self, X, y):
        self.feature_names = X.columns  # 存储特征名称
        X = X.values  # 将 DataFrame 转换为 NumPy 数组
        y = y.values  # 将 Series 转换为 NumPy 数组
        self.tree = self._build_tree(X, y)  # 构建决策树

    def _build_tree(self, X, y):
        if len(np.unique(y)) == 1:  # 如果所有样本属于同一类别,返回该类别
            return y[0]

        if X.shape[1] == 0:  # 如果没有更多特征可用于划分,返回出现次数最多的类别
            return Counter(y).most_common(1)[0][0]

        best_feature, best_threshold = self._choose_best_feature(X, y)  # 选择最佳划分特征和阈值
        if best_feature is None:  # 如果无法找到合适的划分特征,返回出现次数最多的类别
            return Counter(y).most_common(1)[0][0]

        feature_name = self.feature_names[best_feature]  # 获取最佳划分特征的名称
        tree = {feature_name: {}}  # 创建字典表示当前节点
        left_mask = X[:, best_feature] < best_threshold  # 左子树的样本掩码
        right_mask = X[:, best_feature] >= best_threshold  # 右子树的样本掩码
        X_left, y_left = X[left_mask], y[left_mask]  # 左子树的样本和标签
        X_right, y_right = X[right_mask], y[right_mask]  # 右子树的样本和标签

        tree[feature_name]['< ' + str(best_threshold)] = self._build_tree(X_left, y_left)  # 递归构建左子树
        tree[feature_name]['>= ' + str(best_threshold)] = self._build_tree(X_right, y_right)  # 递归构建右子树
        return tree

    def _choose_best_feature(self, X, y):
        best_gain = -1  # 最佳信息增益
        best_feature = None  # 最佳划分特征
        best_threshold = None  # 最佳划分阈值

        for feature in range(X.shape[1]):  # 遍历所有特征
            thresholds = np.unique(X[:, feature])  # 获取当前特征的所有取值作为候选阈值
            for threshold in thresholds:  # 遍历所有候选阈值
                gain = self._information_gain(X, y, feature, threshold)  # 计算当前特征和阈值的信息增益

                if gain > best_gain:  # 如果当前信息增益更大,更新最佳划分特征、阈值和信息增益
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_gain < self.epsilon:  # 如果最佳信息增益小于阈值,则停止划分
            return None, None

        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        parent_entropy = self._entropy(y)  # 计算父节点的熵
        left_mask = X[:, feature] < threshold  # 左子树的样本掩码
        right_mask = X[:, feature] >= threshold  # 右子树的样本掩码
        n_left = left_mask.sum()  # 左子树的样本数
        n_right = right_mask.sum()  # 右子树的样本数
        if n_left == 0 or n_right == 0:  # 如果左子树或右子树没有样本,信息增益为0
            return 0
        child_entropy = (n_left / len(y)) * self._entropy(y[left_mask]) + (n_right / len(y)) * self._entropy(y[right_mask])  # 计算子节点的熵
        return parent_entropy - child_entropy  # 返回信息增益

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)  # 获取每个类别的样本数
        probabilities = counts / len(y)  # 计算每个类别的概率
        return -(probabilities * np.log2(probabilities)).sum()  # 计算熵

    def predict(self, X):
        X = X.values  # 将 DataFrame 转换为 NumPy 数组
        return [self._traverse_tree(x, self.tree) for x in X]  # 对每个样本遍历决策树进行预测

    def _traverse_tree(self, x, node):
        if not isinstance(node, dict):  # 如果当前节点是叶子节点,直接返回类别
            return node

        feature_name = list(node.keys())[0]  # 获取当前节点的特征名称
        feature = self.feature_names.get_loc(feature_name)  # 获取当前特征的索引
        thresholds = list(node[feature_name].keys())  # 获取当前节点的所有阈值
        threshold_left = [t for t in thresholds if t.startswith('<')][0]  # 获取左子树的阈值
        threshold_right = [t for t in thresholds if t.startswith('>=')][0]  # 获取右子树的阈值

        if x[feature] < float(threshold_left.split(' ')[1]):  # 如果样本的特征值小于左子树阈值,递归遍历左子树
            return self._traverse_tree(x, node[feature_name][threshold_left])
        else:  # 否则,递归遍历右子树
            return self._traverse_tree(x, node[feature_name][threshold_right])
```

> 1. 初始化决策树对象,设置信息增益阈值、决策树和特征名称。
> 2. 在 `fit` 方法中,将数据集转换为 NumPy 数组,并调用 `_build_tree` 方法构建决策树。
> 3. 在 `_build_tree` 方法中,递归构建决策树。如果所有样本属于同一类别或没有更多特征可用于划分,则返回相应的类别。否则,调用 `_choose_best_feature` 方法选择最佳划分特征和阈值,并根据阈值将数据集划分为左右子树,递归构建子树。
> 4. 在 `_choose_best_feature` 方法中,遍历所有特征和候选阈值,计算每个特征和阈值的信息增益,选择信息增益最大的特征和阈值作为最佳划分。如果最佳信息增益小于阈值,则停止划分。
> 5. 在 `_information_gain` 方法中,计算特定特征和阈值的信息增益。首先计算父节点的熵,然后根据阈值将数据集划分为左右子树,计算子节点的熵,最后返回信息增益(父节点熵减去子节点熵)。
> 6. 在 `_entropy` 方法中,计算给定数据集的熵。首先获取每个类别的样本数,计算每个类别的概率,然后计算熵。
> 7. 在 `predict` 方法中,对每个样本遍历决策树进行预测。在 `_traverse_tree` 方法中,根据当前节点的特征和阈值,递归遍历左子树或右子树,直到达到叶子节点,返回相应的类别。

 **缺点**

- <font color='dd0000'>**ID3 没有剪枝策略，容易过拟合**</font>；
- 信息增益准则对可取值数目较多的特征有所偏好，类似“编号”的特征其信息增益接近于 1；
- 只能用于处理离散分布的特征；
- 没有考虑缺失值。

**结果:**

【---------测试集----------】
Accuracy: 0.97
正反例的Accuracy: 0.97
正反例:召回率(Recall): 0.9821428571428571
正反例:精确率(Precision): 0.9482758620689655
正反例:F1-Score: 0.9649122807017544

![image-20240409093734533](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240409093734533.png)

【---------验证集----------】
验证集Accuracy: 0.993
正反例的Accuracy: 0.993
正反例:召回率(Recall): 0.9966666666666667
正反例:精确率(Precision): 0.9867986798679867
正反例:F1-Score: 0.9917081260364843

![image-20240409093752717](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240409093752717.png)

---

## C4.5决策树

利用信息增益率可以克服信息增益的缺点，其公式为![image-20240408235747490](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240408235747490.png)

$H_A(D)$!称为特征 A 的固有值。

这里需要注意，信息增益率对可取值较少的特征有所偏好（分母越小，整体越大），因此 C4.5 并不是直接用增益率最大的特征进行划分，而是使用一个启发式方法：先从候选划分特征中找到信息增益高于平均值的特征，再从中选择增益率最高的。

**手写代码解释**

C4.5决策树在ID3决策树的基础上进行了改进,主要有以下几点:

1. 使用增益率(Gain Ratio)代替信息增益(Information Gain)作为划分标准。增益率考虑了特征的固有值(Intrinsic Value),可以减少偏向选择取值较多的特征的问题。
2. 引入了预剪枝(Pre-pruning)策略。通过设置最小分裂样本数(`min_samples_split`)和最小信息增益(`min_gain`)两个参数,在决策树生长过程中进行剪枝。如果样本数小于最小分裂样本数或最佳增益率小于最小信息增益,则停止划分,返回出现次数最多的类别作为叶子节点。这种策略可以防止决策树过拟合。
3. C4.5算法还支持处理连续值特征。对于连续值特征,通过选择最佳阈值将特征二值化,然后递归构建子树。

总的来说,这个C4.5决策树的实现通过引入**增益率和预剪枝策略**,在一定程度上改进了ID3算法,提高了决策树的泛化能力和可解释性

```python
class C45DecisionTree:
    def __init__(self, min_samples_split=2, min_gain=0.0):
        self.min_samples_split = min_samples_split  # 最小分裂样本数,用于预剪枝
        self.min_gain = min_gain  # 最小信息增益,用于预剪枝
        self.tree = None  # 决策树
        self.feature_names = None  # 特征名称

    def fit(self, X, y):
        self.feature_names = X.columns  # 存储特征名称
        X = X.values  # 将 DataFrame 转换为 NumPy 数组
        y = y.values  # 将 Series 转换为 NumPy 数组
        self.tree = self._build_tree(X, y)  # 构建决策树

    def _build_tree(self, X, y):
        if len(np.unique(y)) == 1:  # 如果所有样本属于同一类别,返回该类别
            return y[0]

        if X.shape[0] < self.min_samples_split:  # 如果样本数小于最小分裂样本数,返回出现次数最多的类别(预剪枝)
            return Counter(y).most_common(1)[0][0]

        best_feature, best_threshold = self._choose_best_feature(X, y)  # 选择最佳划分特征和阈值
        if best_feature is None:  # 如果无法找到合适的划分特征,返回出现次数最多的类别
            return Counter(y).most_common(1)[0][0]

        feature_name = self.feature_names[best_feature]  # 获取最佳划分特征的名称
        tree = {feature_name: {}}  # 创建字典表示当前节点
        left_mask = X[:, best_feature] < best_threshold  # 左子树的样本掩码
        right_mask = X[:, best_feature] >= best_threshold  # 右子树的样本掩码
        X_left, y_left = X[left_mask], y[left_mask]  # 左子树的样本和标签
        X_right, y_right = X[right_mask], y[right_mask]  # 右子树的样本和标签

        tree[feature_name]['< ' + str(best_threshold)] = self._build_tree(X_left, y_left)  # 递归构建左子树
        tree[feature_name]['>= ' + str(best_threshold)] = self._build_tree(X_right, y_right)  # 递归构建右子树
        return tree

    def _choose_best_feature(self, X, y):
        best_gain_ratio = -1  # 最佳增益率
        best_feature = None  # 最佳划分特征
        best_threshold = None  # 最佳划分阈值

        for feature in range(X.shape[1]):  # 遍历所有特征
            thresholds = np.unique(X[:, feature])  # 获取当前特征的所有取值作为候选阈值
            for threshold in thresholds:  # 遍历所有候选阈值
                gain_ratio = self._gain_ratio(X, y, feature, threshold)  # 计算当前特征和阈值的增益率

                if gain_ratio > best_gain_ratio:  # 如果当前增益率更大,更新最佳划分特征、阈值和增益率
                    best_gain_ratio = gain_ratio
                    best_feature = feature
                    best_threshold = threshold

        if best_gain_ratio < self.min_gain:  # 如果最佳增益率小于阈值,则停止划分(预剪枝)
            return None, None

        return best_feature, best_threshold

    def _gain_ratio(self, X, y, feature, threshold):
        gain = self._information_gain(X, y, feature, threshold)  # 计算信息增益
        intrinsic_value = self._intrinsic_value(X, feature, threshold)  # 计算固有值
        if intrinsic_value == 0:  # 如果固有值为0,返回0
            return 0
        return gain / intrinsic_value  # 返回增益率

    def _intrinsic_value(self, X, feature, threshold):
        left_mask = X[:, feature] < threshold  # 左子树的样本掩码
        right_mask = X[:, feature] >= threshold  # 右子树的样本掩码
        n_left = left_mask.sum()  # 左子树的样本数
        n_right = right_mask.sum()  # 右子树的样本数
        if n_left == 0 or n_right == 0:  # 如果左子树或右子树没有样本,返回0
            return 0
        n_total = len(X)  # 总样本数
        intrinsic_value = -(n_left / n_total) * np.log2(n_left / n_total) - (n_right / n_total) * np.log2(n_right / n_total)  # 计算固有值
        return intrinsic_value

    def _information_gain(self, X, y, feature, threshold):
        parent_entropy = self._entropy(y)  # 计算父节点的熵
        left_mask = X[:, feature] < threshold  # 左子树的样本掩码
        right_mask = X[:, feature] >= threshold  # 右子树的样本掩码
        n_left = left_mask.sum()  # 左子树的样本数
        n_right = right_mask.sum()  # 右子树的样本数
        if n_left == 0 or n_right == 0:  # 如果左子树或右子树没有样本,信息增益为0
            return 0
        child_entropy = (n_left / len(y)) * self._entropy(y[left_mask]) + (n_right / len(y)) * self._entropy(y[right_mask])  # 计算子节点的熵
        return parent_entropy - child_entropy  # 返回信息增益

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)  # 获取每个类别的样本数
        probabilities = counts / len(y)  # 计算每个类别的概率
        return -(probabilities * np.log2(probabilities)).sum()  # 计算熵

    def predict(self, X):
        X = X.values  # 将 DataFrame 转换为 NumPy 数组
        return [self._traverse_tree(x, self.tree) for x in X]  # 对每个样本遍历决策树进行预测

    def _traverse_tree(self, x, node):
        if not isinstance(node, dict):  # 如果当前节点是叶子节点,直接返回类别
            return node

        feature_name = list(node.keys())[0]  # 获取当前节点的特征名称
        feature = self.feature_names.get_loc(feature_name)  # 获取当前特征的索引
        thresholds = list(node[feature_name].keys())  # 获取当前节点的所有阈值
        threshold_left = [t for t in thresholds if t.startswith('<')][0]  # 获取左子树的阈值
        threshold_right = [t for t in thresholds if t.startswith('>=')][0]  # 获取右子树的阈值

        if x[feature] < float(threshold_left.split(' ')[1]):  # 如果样本的特征值小于左子树阈值,递归遍历左子树
            return self._traverse_tree(x, node[feature_name][threshold_left])
        else:  # 否则,递归遍历右子树
            return self._traverse_tree(x, node[feature_name][threshold_right])
```

**缺点**

- 剪枝策略可以再优化；
- C4.5 用的是多叉树，用二叉树效率更高；
- C4.5 只能用于分类；
- C4.5 使用的熵模型拥有大量耗时的对数运算，连续值还有排序运算；
- C4.5 在构造树的过程中，对数值属性值需要按照其大小进行排序，从中选择一个分割点，所以只适合于能够驻留于内存的数据集，当训练集大得无法在内存容纳时，程序无法运行。

**结果：**

【---------测试集----------】
Accuracy: 0.975
正反例的Accuracy: 0.975
正反例:召回率(Recall): 0.9821428571428571
正反例:精确率(Precision): 0.9649122807017544
正反例:F1-Score: 0.9734513274336283

![image-20240409093629527](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240409093629527.png)

【---------验证集----------】
验证集Accuracy: 0.995
正反例的Accuracy: 0.995
正反例:召回率(Recall): 0.9966666666666667
正反例:精确率(Precision): 0.9933554817275747
正反例:F1-Score: 0.995008319467554

![image-20240409093649025](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240409093649025.png)

---

## CHAID决策树

CHAID（Chi-squared Automatic Interaction Detection）是一种基于卡方检验的决策树算法，主要用于分类问题。它与其他决策树算法的不同之处在于，CHAID在选择最佳划分特征时使用了卡方检验来评估特征的统计显著性。

对于每一个节点，首先对所有特征与标签做卡方检验（离散变量用卡方检验，连续变量用F检验），取卡方值最大的那个特征作为要分裂的特征，这个特征与标签相关性最大。

接着，寻找这个特征的切分点，对于离散变量，我们选择两个可取值，对这两个可取值和标签之间进行卡方检验，得到卡方值，并且我们要确定一个临界值α，当卡方值<α时，说明这两个可取值对标签的相关性比较小，那么就将这两个可取值合并。一直重复这个步骤，直到无法合并，就找到了切分点。

<img src="https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240408231024879.png" alt="image-20240408231024879" style="zoom: 67%;" />

**手写代码分析**：

```python
class CHAIDDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, alpha=0.05):
        self.max_depth = max_depth  # 最大树深度,用于预剪枝
        self.min_samples_split = min_samples_split  # 最小分裂样本数,用于预剪枝
        self.min_samples_leaf = min_samples_leaf  # 叶子节点最小样本数,用于预剪枝
        self.alpha = alpha  # 显著性水平,用于卡方检验
        self.tree = None  # 决策树
        self.feature_names = None  # 特征名称

    def fit(self, X, y):
        self.feature_names = X.columns  # 存储特征名称
        X = X.values  # 将 DataFrame 转换为 NumPy 数组
        y = y.values  # 将 Series 转换为 NumPy 数组
        self.tree = self._build_tree(X, y, depth=0)  # 构建决策树

    def _build_tree(self, X, y, depth):
        if self.max_depth is not None and depth >= self.max_depth:  # 如果达到最大树深度,返回出现次数最多的类别(预剪枝)
            return Counter(y).most_common(1)[0][0]

        if len(np.unique(y)) == 1:  # 如果所有样本属于同一类别,返回该类别
            return y[0]

        if X.shape[0] < self.min_samples_split or X.shape[0] < 2 * self.min_samples_leaf:  # 如果样本数小于最小分裂样本数或不足以分裂为两个叶子节点,返回出现次数最多的类别(预剪枝)
            return Counter(y).most_common(1)[0][0]

        best_feature, best_splits = self._choose_best_split(X, y)  # 选择最佳分裂特征和分裂点
        if best_feature is None:  # 如果无法找到合适的分裂特征,返回出现次数最多的类别
            return Counter(y).most_common(1)[0][0]

        feature_name = self.feature_names[best_feature]  # 获取最佳分裂特征的名称
        tree = {feature_name: {}}  # 创建字典表示当前节点
        for split_value, mask in best_splits.items():  # 遍历每个分裂点
            X_subset, y_subset = X[mask], y[mask]  # 获取分裂后的子集
            if X_subset.shape[0] < self.min_samples_leaf:  # 如果子集样本数小于叶子节点最小样本数,将其设为叶子节点,返回出现次数最多的类别(预剪枝)
                tree[feature_name][split_value] = Counter(y_subset).most_common(1)[0][0]
            else:  # 否则,递归构建子树
                tree[feature_name][split_value] = self._build_tree(X_subset, y_subset, depth + 1)
        return tree

    def _choose_best_split(self, X, y):
        best_feature = None  # 最佳分裂特征
        best_splits = None  # 最佳分裂点
        best_p_value = 1.0  # 最佳p值

        for feature in range(X.shape[1]):  # 遍历所有特征
            unique_values = np.unique(X[:, feature])  # 获取当前特征的唯一值
            if len(unique_values) <= 1:  # 如果唯一值少于等于1,跳过该特征
                continue

            contingency_table = pd.crosstab(X[:, feature], y)  # 创建列联表
            chi2, p_value, _, _ = chi2_contingency(contingency_table)  # 进行卡方检验

            if p_value < best_p_value and p_value <= self.alpha:  # 如果当前p值小于最佳p值且小于显著性水平,更新最佳分裂特征、分裂点和p值
                best_p_value = p_value
                best_feature = feature
                best_splits = {value: (X[:, feature] == value) for value in unique_values}

        return best_feature, best_splits

    def predict(self, X):
        X = X.values  # 将 DataFrame 转换为 NumPy 数组
        return [self._traverse_tree(x, self.tree) for x in X]  # 对每个样本遍历决策树进行预测

    def _traverse_tree(self, x, node):
        if not isinstance(node, dict):  # 如果当前节点是叶子节点,直接返回类别
            return node

        feature_name = list(node.keys())[0]  # 获取当前节点的特征名称
        feature = self.feature_names.get_loc(feature_name)  # 获取当前特征的索引
        split_values = list(node[feature_name].keys())  # 获取当前节点的所有分裂点

        for split_value in split_values:  # 遍历分裂点
            if x[feature] == split_value:  # 如果样本的特征值等于分裂点,递归遍历对应的子树
                return self._traverse_tree(x, node[feature_name][split_value])
```

CHAID(Chi-squared Automatic Interaction Detection)决策树是一种基于卡方检验的决策树算法。它的主要特点是:

1. 使用卡方检验来选择最佳分裂特征和分裂点。通过计算每个特征的列联表和对应的卡方统计量,选择卡方统计量对应的p值最小且小于显著性水平的特征作为分裂特征,并根据唯一值进行分裂。
2. 支持多路分裂。与二叉决策树不同,CHAID决策树可以在每个节点上产生多个分支,每个分支对应一个唯一值。
3. 使用预剪枝策略来控制树的生长。通过设置最大树深度(`max_depth`)、最小分裂样本数(`min_samples_split`)、叶子节点最小样本数(`min_samples_leaf`)等参数,在树的生长过程中进行剪枝。具体来说:
   - 如果当前深度达到最大树深度,停止分裂,返回出现次数最多的类别。
   - **如果当前节点的样本数小于最小分裂样本数或不足以分裂为两个满足叶子节点最小样本数要求的子节点,停止分裂,返回出现次数最多的类别。**
   - **如果分裂后的子节点样本数小于叶子节点最小样本数,将其设为叶子节点,返回出现次数最多的类别。****

总的来说,这个CHAID决策树的实现利用卡方检验来选择最佳分裂特征和分裂点,支持多路分裂,并使用预剪枝策略来控制树的生长。

**结果：**

【---------测试集----------】
Accuracy: 0.86
正反例的Accuracy: 0.86
正反例:召回率(Recall): 0.8214285714285714
正反例:精确率(Precision): 0.8679245283018868
正反例:F1-Score: 0.8440366972477065

![image-20240409105147412](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240409105147412.png)

【---------验证集----------】
验证集Accuracy: 0.935
正反例的Accuracy: 0.935
正反例:召回率(Recall): 0.9233333333333333
正反例:精确率(Precision): 0.9518900343642611
正反例:F1-Score: 0.9373942470389172

![image-20240409105203310](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240409105203310.png)

## CART决策树

ID3 、 C4.5、CHAID 虽然在对训练样本集的学习中可以尽可能多地挖掘信息，但是其生成的决策树分支、规模都比较大，CART 算法的二分法可以简化决策树的规模，提高生成决策树的效率。

CART（Classification and Regression Trees）是一种常用的决策树算法，既可以用于分类问题，也可以用于回归问题。它的核心思想是通过递归地将数据集分割成越来越小的子集，直到子集足够纯净（纯净度的度量方式根据分类或回归问题不同而不同）或者达到了预先设定的停止条件

在每个节点处，CART算法选择一个特征和一个阈值，将数据集划分成两个子集，使得子集的纯度增加（或者不纯度减少）。对于分类问题，通常使用基尼不纯度（Gini Impurity）或者信息增益（Entropy）来衡量纯度的增加；对于回归问题，通常使用平方误差来衡量预测值与真实值之间的差异。

> 熵模型拥有大量耗时的对数运算，基尼指数在简化模型的同时还保留了熵模型的优点。基尼指数代表了模型的不纯度，基尼系数越小，不纯度越低，特征越好。这和信息增益（率）正好相反。
>
> ![image-20240409000138003](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240409000138003.png)

**手写代码分析**

```python
class CARTDecisionTree:
    def __init__(self, min_samples_split=2, min_impurity_decrease=0.0):
        self.min_samples_split = min_samples_split  # 最小分裂样本数,用于预剪枝
        self.min_impurity_decrease = min_impurity_decrease  # 最小不纯度减少量,用于预剪枝
        self.tree = None  # 决策树
        self.feature_names = None  # 特征名称

    def fit(self, X, y):
        self.feature_names = X.columns  # 存储特征名称
        X = X.values  # 将 DataFrame 转换为 NumPy 数组
        y = y.values  # 将 Series 转换为 NumPy 数组
        self.tree = self._build_tree(X, y)  # 构建决策树

    def _build_tree(self, X, y):
        if len(np.unique(y)) == 1:  # 如果所有样本属于同一类别,返回该类别
            return y[0]

        if X.shape[0] < self.min_samples_split:  # 如果样本数小于最小分裂样本数,返回出现次数最多的类别(预剪枝)
            return Counter(y).most_common(1)[0][0]

        best_feature, best_threshold = self._choose_best_split(X, y)  # 选择最佳分裂特征和阈值
        if best_feature is None:  # 如果无法找到合适的分裂特征,返回出现次数最多的类别
            return Counter(y).most_common(1)[0][0]

        feature_name = self.feature_names[best_feature]  # 获取最佳分裂特征的名称
        tree = {feature_name: {}}  # 创建字典表示当前节点
        left_mask = X[:, best_feature] < best_threshold  # 左子树的样本掩码
        right_mask = X[:, best_feature] >= best_threshold  # 右子树的样本掩码
        X_left, y_left = X[left_mask], y[left_mask]  # 左子树的样本和标签
        X_right, y_right = X[right_mask], y[right_mask]  # 右子树的样本和标签

        tree[feature_name]['< ' + str(best_threshold)] = self._build_tree(X_left, y_left)  # 递归构建左子树
        tree[feature_name]['>= ' + str(best_threshold)] = self._build_tree(X_right, y_right)  # 递归构建右子树
        return tree

    def _choose_best_split(self, X, y):
        best_gini = 1.0  # 最佳基尼不纯度
        best_feature = None  # 最佳分裂特征
        best_threshold = None  # 最佳分裂阈值

        for feature in range(X.shape[1]):  # 遍历所有特征
            thresholds = np.unique(X[:, feature])  # 获取当前特征的所有取值作为候选阈值
            for threshold in thresholds:  # 遍历所有候选阈值
                gini = self._gini_impurity(X, y, feature, threshold)  # 计算当前特征和阈值的基尼不纯度

                if gini < best_gini:  # 如果当前基尼不纯度更小,更新最佳分裂特征、阈值和基尼不纯度
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        if 1 - best_gini < self.min_impurity_decrease:  # 如果最佳分裂的不纯度减少量小于阈值,停止分裂(预剪枝)
            return None, None

        return best_feature, best_threshold

    def _gini_impurity(self, X, y, feature, threshold):
        left_mask = X[:, feature] < threshold  # 左子树的样本掩码
        right_mask = X[:, feature] >= threshold  # 右子树的样本掩码
        n_left = left_mask.sum()  # 左子树的样本数
        n_right = right_mask.sum()  # 右子树的样本数
        if n_left == 0 or n_right == 0:  # 如果左子树或右子树没有样本,基尼不纯度为1
            return 1.0

        _, left_counts = np.unique(y[left_mask], return_counts=True)  # 获取左子树每个类别的样本数
        _, right_counts = np.unique(y[right_mask], return_counts=True)  # 获取右子树每个类别的样本数
        gini_left = 1 - np.sum((left_counts / n_left) ** 2)  # 计算左子树的基尼不纯度
        gini_right = 1 - np.sum((right_counts / n_right) ** 2)  # 计算右子树的基尼不纯度
        gini = (n_left * gini_left + n_right * gini_right) / (n_left + n_right)  # 计算加权平均基尼不纯度
        return gini

    def predict(self, X):
        X = X.values  # 将 DataFrame 转换为 NumPy 数组
        return [self._traverse_tree(x, self.tree) for x in X]  # 对每个样本遍历决策树进行预测

    def _traverse_tree(self, x, node):
        if not isinstance(node, dict):  # 如果当前节点是叶子节点,直接返回类别
            return node

        feature_name = list(node.keys())[0]  # 获取当前节点的特征名称
        feature = self.feature_names.get_loc(feature_name)  # 获取当前特征的索引
        thresholds = list(node[feature_name].keys())  # 获取当前节点的所有阈值
        threshold_left = [t for t in thresholds if t.startswith('<')][0]  # 获取左子树的阈值
        threshold_right = [t for t in thresholds if t.startswith('>=')][0]  # 获取右子树的阈值

        if x[feature] < float(threshold_left.split(' ')[1]):  # 如果样本的特征值小于左子树阈值,递归遍历左子树
            return self._traverse_tree(x, node[feature_name][threshold_left])
        else:  # 否则,递归遍历右子树
            return self._traverse_tree(x, node[feature_name][threshold_right])
```

这里的CART(Classification and Regression Trees)决策树使用基于基尼不纯度(Gini Impurity)，它的主要特点是:

1. 使用基尼不纯度作为分裂标准。基尼不纯度衡量了数据集中类别的混乱程度,值越小表示数据集的纯度越高。在每个节点上,选择基尼不纯度最小的特征和阈值进行分裂。
2. 生成二叉决策树。与多路分裂的决策树不同,CART决策树在每个节点上只进行二分裂,生成左右两个子树。
3. 支持回归和分类任务。CART算法可以用于处理连续值目标变量(回归)和离散值目标变量(分类)。对于回归任务,使用均方差作为分裂标准,对于分类任务,使用基尼不纯度作为分裂标准。
4. 使用预剪枝策略来控制树的生长。通过设置最小分裂样本数(`min_samples_split`)和最小不纯度减少量(`min_impurity_decrease`)等参数,在树的生长过程中进行剪枝。具体来说:
   - 如果当前节点的样本数小于最小分裂样本数,停止分裂,返回出现次数最多的类别。
   - 如果当前节点的最佳分裂的不纯度减少量小于最小不纯度减少量,停止分裂,返回出现次数最多的类别。

**结果**

【---------测试集----------】
Accuracy: 0.97
正反例的Accuracy: 0.97
正反例:召回率(Recall): 0.9821428571428571
正反例:精确率(Precision): 0.9482758620689655
正反例:F1-Score: 0.9649122807017544

![image-20240409105332294](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240409105332294.png)

【---------验证集----------】
验证集Accuracy: 0.994
正反例的Accuracy: 0.994
正反例:召回率(Recall): 0.9966666666666667
正反例:精确率(Precision): 0.9900662251655629
正反例:F1-Score: 0.9933554817275748

![image-20240409105346612](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240409105346612.png)

### 关于四种基本决策树的小总结

根据给出的ID3、C4.5、CHAID和CART决策树的代码实现,对它们进行详细的比较和分析:

1. 分裂标准:
   - ID3: 使用信息增益(Information Gain)作为分裂标准。信息增益衡量了特征对减少数据集不确定性的贡献度,值越大表示特征对分类的重要性越高。
   - C4.5: 使用增益率(Gain Ratio)作为分裂标准。增益率是信息增益与固有值(Intrinsic Value)的比值,用于解决信息增益偏向选择取值较多的特征的问题。
   - CHAID: 使用卡方检验(Chi-squared Test)作为分裂标准。通过计算每个特征的列联表和对应的卡方统计量,选择卡方统计量对应的p值最小且小于显著性水平的特征作为分裂特征。
   - CART: 使用基尼不纯度(Gini Impurity)作为分裂标准。基尼不纯度衡量了数据集中类别的混乱程度,值越小表示数据集的纯度越高。
2. 树的分支:
   - ID3算法生成的是多叉决策树。ID3在每个节点选择一个最优划分属性,该属性可能有多个取值,因此决策树的分支数由属性值的个数决定,是一棵多叉树。
   - C4.5算法是ID3的改进,也生成多叉决策树。与ID3一样,C4.5选择最优划分属性,属性可以有多个取值,所以生成的是多叉树。不过C4.5引入了增益率来选择属性,并且能处理连续属性。
   - CART算法生成的是二叉决策树。CART每次将当前样本集一分为二,所以不管在中间过程还是最终的决策树,都是一棵二叉树。
   - CHAID算法生成的是多叉决策树。它基于卡方检验来选择最佳分割变量和分割点,一个父节点可以有两个或多个子节点,因此得到的是多叉树。
3. 连续值处理:
   - ID3: 没有明确处理连续值特征的方法。通常需要对连续值特征进行离散化预处理。
   - C4.5、CART: 通过选择最佳阈值将连续值特征二值化,然后递归构建子树。
   - CHAID: 没有明确处理连续值特征的方法。通常需要对连续值特征进行离散化预处理。
4. 缺失值处理:
   - ID3、C4.5、CHAID、CART: 在给出的代码实现中,都没有明确处理缺失值的方法。通常需要在预处理阶段对缺失值进行填充或删除。由于实验数据没有缺失值
5. 剪枝策略:
   - ID3: 没有明确的剪枝策略。容易生成过拟合的决策树。
   - C4.5: 使用预剪枝策略。通过设置最小分裂样本数、最小信息增益等参数,在树的生长过程中限制树的深度和复杂度。
   - CHAID: 使用预剪枝策略。通过设置最大树深度、最小分裂样本数、叶子节点最小样本数等参数,在树的生长过程中限制树的深度和复杂度。
   - CART: 使用预剪枝策略。通过设置最小分裂样本数、最小不纯度减少量等参数,在树的生长过程中限制树的深度和复杂度。
6. 特点和适用场景:
   - ID3: 简单易懂,适用于离散值特征的分类任务。但对连续值特征和缺失值的处理有局限性,且容易过拟合。
   - C4.5: 相比ID3,使用增益率来选择分裂特征,减少了偏向选择取值较多的特征的问题。支持连续值特征的处理,并引入了预剪枝策略。适用于离散值和连续值特征的分类任务。
   - CHAID: 使用卡方检验来选择分裂特征,支持多路分裂,生成的决策树更加直观和易于解释。适用于离散值特征的分类任务,特别是在特征取值较多的情况下。
   - CART: 使用基尼不纯度来选择分裂特征,支持连续值特征的处理,并引入了预剪枝策略。适用于离散值和连续值特征的分类和回归任务。

## RandomForest随机森林方法

RandomForestClassifier是scikit-learn提供的随机森林分类器的实现。它基于CART决策树分类器,通过集成多棵决策树的方式来提高分类的准确性和鲁棒性。下面是RandomForestClassifier的简单介绍:

1. 随机采样:RandomForestClassifier从原始训练集中随机有放回地抽取若干个样本子集,每个子集用来训练一棵决策树。这种有放回的抽样方式被称为bootstrap采样。
2. 特征随机选择:在构建每棵决策树时,RandomForestClassifier从所有特征中随机选择一个特征子集,只在这个子集中选择最优分割特征。这种特征抽样方式能够降低决策树之间的相关性。
3. 决策树生成:对于每个样本子集,RandomForestClassifier使用选定的特征子集来构建一棵决策树。决策树的生成过程与一般的决策树分类器类似,通过递归地选择最优分割特征来生长树,直到达到停止条件。
4. 多数投票:生成指定数量的决策树后,RandomForestClassifier将它们组合成为一个森林。在进行预测时,每棵树独立地对新样本进行分类,然后通过多数投票的方式得到最终的分类结果。

RandomForestClassifier的主要参数包括:

- n_estimators:生成的决策树的数量,默认为100。
- max_depth:每棵决策树的最大深度,默认为None,即不限制深度。
- min_samples_split:内部节点再划分所需最小样本数,默认为2。
- max_features:每棵决策树随机选择的特征数量,默认为"auto",即特征数的平方根。

*示例*

```python
from sklearn.ensemble import RandomForestClassifier

# 创建随机森林分类器
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# 训练模型
rf.fit(X_train, y_train)

# 预测新样本
y_pred = rf.predict(X_test)
```

**结果：**

【---------测试集----------】
Accuracy: 0.94
正反例的Accuracy: 0.94
正反例:召回率(Recall): 0.8928571428571429
正反例:精确率(Precision): 0.9803921568627451
正反例:F1-Score: 0.9345794392523364

![image-20240409105606027](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240409105606027.png)

【---------验证集----------】
验证集Accuracy: 0.988
正反例的Accuracy: 0.988
正反例:召回率(Recall): 0.98
正反例:精确率(Precision): 0.9966101694915255
正反例:F1-Score: 0.9882352941176471

![image-20240409105622501](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240409105622501.png)

## XGBoost方法

XGBoost (Extreme Gradient Boosting) 是一种基于梯度提升决策树 (GBDT) 算法的高效机器学习库。以上代码展示了如何使用 XGBoost 进行多分类任务。下面我将详细介绍代码中使用的 XGBoost 方法的实现过程：

1. 数据准备：
   - 从 CSV 文件中读取数据集，并指定列名。
   - 将特征值转换为数值表示，使用 `astype('category').cat.codes` 将字符串特征转换为整数编码。
   - 使用 `LabelEncoder` 对标签列进行编码，将字符串标签转换为整数标签。
   - 将数据集分为训练集和测试集。
2. 转换数据格式：
   - XGBoost 需要使用特定的数据格式 `DMatrix`。
   - 使用 `xgb.DMatrix()` 将训练集和测试集转换为 `DMatrix` 格式，其中 `label` 参数指定了标签列。
3. 设置参数：
   - `objective`: 指定问题类型，这里是多分类问题，使用 `'multi:softmax'`。
   - `num_class`: 指定类别数，即标签的唯一值的数量。
   - `max_depth`: 决策树的最大深度，用于控制模型复杂度和防止过拟合。
   - `eta`: 学习率或步长，控制每次迭代的权重缩减，可以防止过拟合。
   - `eval_metric`: 评估指标，这里使用分类错误率 `'merror'`。
4. 训练模型：
   - 使用 `xgb.train()` 函数训练模型。
   - 传入设置的参数 `params`、转换后的训练数据 `dtrain` 和迭代次数 `num_round`。
   - 训练过程中，XGBoost 会构建一系列的决策树，每棵树都试图去拟合前面树的残差。
   - 通过不断地迭代和优化，模型会逐步提高性能。
5. 预测：
   - 使用训练好的模型 `bst` 对测试集 `dtest` 进行预测，得到预测结果 `y_pred`。
   - XGBoost 会将测试样本传递给每棵决策树，并将所有树的预测结果进行加权平均，得到最终的预测结果。
6. 评估模型：
   - 使用 `accuracy_score()` 计算模型在测试集上的准确率。
   - 将预测结果 `y_pred` 与真实标签 `y_test` 进行比较，计算分类的准确性。

XGBoost 的核心在于梯度提升决策树算法。它通过迭代地构建一系列的决策树，每棵树都试图去拟合前面树的残差，不断地优化模型。XGBoost 在每次迭代时，使用梯度下降来最小化损失函数，并更新树的权重。通过多轮迭代和优化，XGBoost 可以生成一个强大的集成模型，在分类和回归任务上都有良好的表现。

XGBoost 还引入了一些优化技巧，如二阶泰勒展开、正则化、列抽样等，以提高模型的泛化能力和训练效率。同时，XGBoost 还支持并行计算，可以充分利用多核 CPU 加速训练过程。

总的来说，XGBoost 通过梯度提升决策树算法，结合各种优化技巧，实现了高效准确的机器学习模型。它在结构化数据的分类和回归任务上表现出色，广泛应用于各个领域。

**结果：**

【---------测试集----------】
Accuracy: 0.98
正反例的Accuracy: 0.98
正反例:召回率(Recall): 0.9821428571428571
正反例:精确率(Precision): 0.9821428571428571
正反例:F1-Score: 0.9821428571428571

![image-20240409110811203](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240409110811203.png)

【---------验证集----------】
验证集Accuracy: 0.6442307692307693
正反例的Accuracy: 0.6442307692307693
正反例:召回率(Recall): 0.0
正反例:精确率(Precision): 0.0
正反例:F1-Score: 0.0

![image-20240409110827976](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240409110827976.png)

## LightGBM方法

LightGBM (Light Gradient Boosting Machine) 是一种基于决策树的梯度提升框架,它是XGBoost的一个优化版本。以上代码展示了如何使用 LightGBM 进行多分类任务。下面我将详细介绍 LightGBM 分类器的原理：

1. 决策树构建：
   - LightGBM 使用基于直方图的决策树算法,将连续特征值划分为离散的bin,大大降低了内存占用和计算复杂度。
   - 在每个叶子节点上,LightGBM 使用基尼系数（Gini Impurity）来衡量节点的纯度,并选择最佳分割点来生成决策树。
   - LightGBM 采用叶子节点的最大数量作为决策树的停止条件,而不是树的最大深度。
2. 梯度提升：
   - LightGBM 使用梯度提升的思想来训练一系列的决策树。
   - 在每一轮迭代中,LightGBM 计算当前模型的预测值和真实值之间的残差（负梯度）,然后拟合一棵新的决策树来预测这些残差。
   - 新的决策树与之前的树进行加权求和,得到更新后的模型。
   - 通过多轮迭代,LightGBM 不断地优化模型,使其在训练数据上的预测结果尽可能接近真实值。
3. 特征重要性：
   - LightGBM 通过统计每个特征在所有决策树中的分割次数来衡量特征的重要性。
   - 特征在决策树中被用于分割的次数越多,说明该特征对于预测结果的影响越大,重要性也就越高。
   - 特征重要性可以帮助我们理解模型的决策过程,并进行特征选择和特征工程。
4. 参数设置：
   - `objective`: 指定问题类型,这里是多分类问题,使用 `'multiclass'`。
   - `num_class`: 指定类别数,即标签的唯一值的数量。
   - `boosting_type`: 指定提升类型,这里使用梯度提升决策树（GBDT）。
   - `metric`: 评估指标,这里使用多分类对数损失 `'multi_logloss'`。
   - `num_leaves`: 决策树的最大叶子节点数,控制树的复杂度。
   - `learning_rate`: 学习率,控制每棵树的权重缩减,防止过拟合。
   - `feature_fraction`: 特征采样比例,每棵树随机选择的特征比例。
5. 模型训练和预测：
   - 使用 `lgb.Dataset()` 将训练集和测试集转换为 LightGBM 的数据格式。
   - 调用 `lgb.train()` 函数训练模型,传入参数、训练数据、迭代次数和验证数据集。
   - 使用训练好的模型 `bst` 对测试集进行预测,得到预测的概率值。
   - 将预测的概率值转换为类别标签,选择概率最大的类别作为最终预测结果。
6. 模型评估：
   - 使用 `accuracy_score()` 计算模型在测试集上的准确率。
   - 将预测结果 `y_pred` 与真实标签 `y_test` 进行比较,计算分类的准确性。

LightGBM 的优势在于其高效的内存使用和并行计算能力。它使用直方图算法和基于深度的决策树生长策略,大大加快了训练速度。同时,LightGBM 还支持分布式计算和GPU加速,可以处理大规模数据集。

**结果：**

[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
【---------测试集----------】
Accuracy: 0.97
正反例的Accuracy: 0.97
正反例:召回率(Recall): 0.9642857142857143
正反例:精确率(Precision): 0.9818181818181818
正反例:F1-Score: 0.972972972972973

![image-20240409111148908](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240409111148908.png)

【---------验证集----------】
验证集Accuracy: 0.646978021978022
正反例的Accuracy: 0.646978021978022
正反例:召回率(Recall): 0.0
正反例:精确率(Precision): 0.0
正反例:F1-Score: 0.0

![image-20240409111208127](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240409111208127.png)

## GBDT方法

在本代码中,通过创建GradientBoostingClassifier对象,设置决策树的数量(n_estimators)和随机种子(random_state),然后使用fit方法在训练数据上训练GBDT模型。接着,使用trained model在测试数据上进行预测,得到预测结果y_pred。最后,通过accuracy_score函数计算模型在测试集上的准确率。

GBDT通过迭代地构建一系列的决策树,并使用梯度提升的方式来优化模型。每一轮迭代都在前一轮模型的残差上训练一棵新的决策树,不断地减少模型的误差。通过多轮迭代,GBDT可以生成一个强大的集成模型,在分类和回归任务上都有出色的表现。

GBDT的优点包括:

1. 非线性建模能力强:通过使用决策树作为基础学习器,GBDT可以捕捉数据中的非线性关系和复杂模式。
2. 鲁棒性好:GBDT对异常值和缺失值有较好的鲁棒性,可以自动处理缺失值。
3. 特征重要性评估:GBDT可以计算每个特征对模型的重要性,帮助进行特征选择和解释。
4. 灵活性高:GBDT允许自定义损失函数,可以适应不同类型的问题。

然而,GBDT也有一些限制:

1. 训练时间较长:由于需要迭代地构建多棵决策树,GBDT的训练时间通常比单个决策树或线性模型长。
2. 参数调优:GBDT有多个超参数需要调整,如学习率、决策树数量、最大深度等,寻找最优参数组合可能需要一定的调优过程。
3. 内存消耗:GBDT在训练过程中需要存储多棵决策树,因此内存消耗较大,尤其是在处理大型数据集时。

总的来说,GBDT是一种强大而灵活的集成学习算法,通过迭代地构建决策树和梯度提升的方式,可以有效地解决分类和回归问题。

**结果**

【---------测试集----------】
Accuracy: 0.965
正反例的Accuracy: 0.965
正反例:召回率(Recall): 0.9285714285714286
正反例:精确率(Precision): 0.9811320754716981
正反例:F1-Score: 0.9541284403669724

![image-20240409111327130](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240409111327130.png)

【---------验证集----------】
验证集Accuracy: 0.988
正反例的Accuracy: 0.988
正反例:召回率(Recall): 0.9866666666666667
正反例:精确率(Precision): 0.9866666666666667
正反例:F1-Score: 0.9866666666666668

<![image-20240409111339232](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240409111339232.png)



#### 参考链接

https://zhuanlan.zhihu.com/p/687418021

https://cloud.tencent.com/developer/article/1657763