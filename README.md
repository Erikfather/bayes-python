# bayes-python
### 具体代码见：bayes_iris.py
### 我直接用了iris_data数据集，每种花我选取前45条数据当做训练集，剩下5条数据另外存入测试集iris_test_data
#### 测试集如下：
![image](https://github.com/Erikfather/bayes-python/blob/master/iris_test_data.jpg)
#### 因为这个数据集是连续性属性，所以需要利用概率密度函数。
#### 具体实验步骤为：
#### （1）先读取数据集
#### （2）计算训练数据集上每个类别的各个特征属性上的均值和方差
#### （3）开始对测试数据集进行分类
#### （4）首先估计先验概率，这里我每个类别所占整体数据集的比例是一样的
#### （5）利用概率密度函数，计算测试数据集上各个属性在每个类别上的条件概率
#### （6）计算后验概率=先验概率*条件概率
#### （7）比较在各个类别上的后验概率，取最大值，则分为这个类别

#### 结果如下：
![image](https://github.com/Erikfather/bayes-python/blob/master/result.jpg)
#### 我们将结果与测试集比较发现结果完全正确！
