## 第二章 监督学习

 参考书籍: 《Python机器学习基础教程》


### 2.1 分类与回归

监督学习的主要研究问题分为 **分类** 和 **回归** 两种。**分类任务**是将数据分成不同的类别（比如判断邮件是垃圾邮件还是正常邮件），**回归任务**是预测一个连续的数值（比如预测房价）。



### 2.2 泛化、过拟合和欠拟合

**泛化**：模型在新数据上的表现能力，越好说明模型越有**泛化能力**。

**过拟合**：模型在训练数据上表现非常好，但在新数据上表现差，说明模型过度复杂，**记住了噪声**。

**欠拟合**：模型过于简单，无法捕捉数据中的模式，导致在训练和测试数据上都表现不佳。

![Image](https://github.com/user-attachments/assets/f22fc2b6-f062-49f1-aa55-887eddefbfab)



#### 2.3 监督学习

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