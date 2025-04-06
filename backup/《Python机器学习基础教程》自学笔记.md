## 第二章 监督学习

 参考书籍: 《Python机器学习基础教程》


### 2.1 分类与回归

监督学习的主要研究问题分为 **分类** 和 **回归** 两种。**分类任务**是将数据分成不同的类别（比如判断邮件是垃圾邮件还是正常邮件），**回归任务**是预测一个连续的数值（比如预测房价）。



### 2.2 泛化、过拟合和欠拟合

**泛化**：模型在新数据上的表现能力，越好说明模型越有**泛化能力**。

**过拟合**：模型在训练数据上表现非常好，但在新数据上表现差，说明模型过度复杂，**记住了噪声**。

**欠拟合**：模型过于简单，无法捕捉数据中的模式，导致在训练和测试数据上都表现不佳。

![Image](https://github.com/user-attachments/assets/f22fc2b6-f062-49f1-aa55-887eddefbfab)



### 2.3 监督学习

#### 2.3.1 一些样本数据

《Python机器学习基础教程》在2.3中介绍了几个在第二章节中会用到的数据：

mglearn.datasets.make_forge() `机器生成的分类数据`

mglearn.datasets.make_wave() `机器生成的回归数据`

![Image](https://github.com/user-attachments/assets/4050d4b8-40ce-4e3c-a71f-6ef671140fb0)

除此以外还有一些现实生活中的高维度数据集，它们内置在了sklearn.datasets中:

- load_breast_cancer()  `威斯康星州乳腺癌数据集`
- load_boston()  `波士顿房价数据集`

> `波士顿房价数据集`在新版本的`sklearn`中因为**人道问题**被移除，无法再使用《Python机器学习基础教程》中介绍的 from sklearn.datasets import load_boston()方式调用了。



#### 2.3.2 K邻近

KNN算法既可以用于分类，也可以用于分类，也可以用于回归，KNN 算法的本质是通过查看与预测点距离最近的数据的类别或数值，来决定预测结果，下图是K值为3的KNN预测算法示例图。

![Image](https://github.com/user-attachments/assets/73d29c2c-7f9f-4d90-a7ac-ca4a3ead5ba7)

##### 分类Forge模型

```python
import mglearn
from mglearn.datasets import make_forge
from sklearn.model_selection import train_test_split

# step.1 导入数据并处理
X, y = mglearn.datasets.make_forge()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

# step.2 训练模型
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3).fit(X_train,y_train)

# step.3 输出模型结果
# [输出结果] Test set Accuracy: 0.8571428571428571
print('Test set Accuracy: {}'.format(clf.score(X_test,y_test)))

# step.4 绘制决策边界
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3)

for n_neighbours,ax in zip([1,3,9],axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbours).fit(X,y)
    mglearn.plots.plot_2d_separator(clf,X,fill=True,ax=ax,alpha=.225)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbours))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()
```

![Image](https://github.com/user-attachments/assets/47a62142-6b2a-4ae0-a903-ac237d2de5e6)

上图展示了数据在不同K值选择下的**决策边界**，使用单一邻居绘制的决策边界紧跟着训练数据。随着邻居个数越来 多，决策边界也越来越平滑。

##### 分类Cancer任务

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# step.1 导入数据并处理
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
 cancer.data, cancer.target, stratify=cancer.target, random_state=66)
training_accuracy = []
test_accuracy = []
# n_neighbors取值从1到10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # 构建模型
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # 记录训练集精度
    training_accuracy.append(clf.score(X_train, y_train))
    # 记录泛化精度
    test_accuracy.append(clf.score(X_test, y_test))
plt.grid(alpha=.225)
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()
```

![Image](https://github.com/user-attachments/assets/88a9ad06-6f4d-4d9d-a5e5-389f87143040)

上图展示了邻居的数量与正确率的关系，调大n_neighbors则导致训练集精度下降，测试集精度上升，折衷点在n_neighbors=6，此时模型既不会过拟合也不会欠拟合，这就是调参。



##### KNN回归

k 近邻算法还可以用于回归，预测的值选择与其最近的n个点的平均值。用于回归的 k 近邻算法在 scikit-learn 的 KNeighborsRegressor 类中实现。其用法与 KNeighborsClassifier 类似。

<img src="C:\Users\flyeg\Desktop\Python机器学习\knn_reg.png" alt="knn_reg" style="zoom: 67%;" />

```python
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np

X, y = mglearn.datasets.make_wave(n_samples=40)
# 将wave数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# 创建1000个数据点，在-3和3之间均匀分布
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # 利用1个、3个或9个邻居分别进行预测
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train),
            reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target",
                "Test data/target"], loc="best")
plt.show()
```

![Image](https://github.com/user-attachments/assets/11c4ffcf-e1c3-4365-b7ae-d88e96e7aa95)

从图中可以看出，仅使用单一邻居，训练集中的每个点都对预测结果有显著影响，预测结 果的图像经过所有数据点。这导致预测结果非常不稳定。考虑更多的邻居之后，预测结果 变得更加平滑，但对训练数据的拟合也不好。



##### 总结

**KNN算法优点:**

- 简单直观，容易理解
- 不用复杂调参，效果通常不错，是个很好的**起步模型**。
- 建模速度快（不训练，只存数据）

**KNN算法缺点:**

- 预测慢：每次都要计算所有点的距离喵。
- 不适合大数据集（样本多 or 特征多）。
- 对高维数据和稀疏数据（大多是 0）效果不好。
- 需要预处理数据，比如归一化等。


#### 2.3.3 线性回归