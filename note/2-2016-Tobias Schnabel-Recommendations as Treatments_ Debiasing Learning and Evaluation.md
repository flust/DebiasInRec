## 2016-Tobias Schnabel-Recommendations as Treatments: Debiasing Learning and Evaluation

#### 1. 解决问题

​	大部分评估和训练推荐系统的数据存在选择偏差——用户自己的选择 / 推荐系统的行为。

​	本文从因果推论的角度提供了解决选择偏差的方法，使用有偏数据生成无偏的表现估计器（*evaluate阶段吗？*），使用MF，在现实数据集上得到了提升，且本文的方法具有高度实用性。

>  任务1: 评估打分预测的准确性

​		也就是评估一个预测矩阵 $\hat{Y}$ 多大程度上反应真实打分 $Y$ ，标准的评估方式比如 MAE / MSE ，对 MAE / MSE 的 naive estimator 是仅使用观测到的条目计算，但是这个估测值会因为选择偏差而产生错误——出现频率越高的条目对结果的影响越大（**出现频率低的条目即使预测偏差非常大也对最终结果影响小**）。  —— $\hat{R}_{naive}(\hat{Y})$

> 任务2: 评估推荐的质量

​		重新定义$\hat{Y}$: 二值矩阵，$i$ 是否被推荐给 $u$ , 一些评估推荐的质量合理的指标 CG / DCG / PREC，这里也使用了上述的 naive estimator，所以也是有偏的。

​		在观测缺失的情况下，为了得到推荐质量的无偏估计，（考虑以下联系去估计平均治疗影响对于给定的策略在因果推论中？？？将推荐视作干预，只是说明推荐与药物treatment的相似性？？）。



#### 2. 主要方法

> 倾向-评分性能估计器

​	**IPS Estimator**: $\hat{R}_{IPS}(\hat{Y}|P) = \frac{1}{U\times I}\sum_{O_{u,i} = 1}\frac{l(Y, \hat{Y})}{P_{u,i}}$ 为无偏的

​	**SNIPS Estimator**: 对 IPS Estimator 做修正，$\hat{R}_{SNIPS}(\hat{Y}|P) = \frac{\hat{R}_{IPS}(\hat{Y}|P)}{\sum_{O_{u,i} = 1}\frac{1}{P_{u,i}}}$



> 倾向-得分推荐学习

​	本文在一个ERM（Empirical Risk Minimization）框架下使用上述无偏估计器（IPS / SNIPS）进行学习，使用一个MF进行打分预测。

​	使用倾向-得分ERM得到一个MF方法进行评分预测，训练目标为$$argmin[\sum_{O_{u,i} = 1}\frac{l_{u,i}(Y,V^TW+A)}{P_{u,i}} + \lambda(||V||_{F}^{2} + ||W||_{F}^2)]$$

​	（损失函数中加入倾向性评分）



#### 3. 技术细节

> 对估计器的实验性验证 - ML100K

​	使用半合成的 ML100K 数据集 —— $Y$ 全部已知 —— 可以计算真实性能，选取 $P_{u,i}$ 去模仿可观测到原始ML100K数据集上的边际打分分布，ML100K数据集处理过程如下：

​		（1）首先使用标准 MF 完成 rating 矩阵

​		（2）修正 rating 矩阵， 具体方法为 按不同打分[1,5]的比例修改评分（先拿出得分高的 $p_5 $% ，将其得分设置为5，同理4、3、2、1)

​		（3）对于每个 $\alpha$ 设置 $k$ （当 rating 为 4/5 时，propensity(for observing) 为 $k$，否则 propensity 为 $k \alpha^{4 - r}$），使得观测到的 rating 为总的 5%，这里发现当 $\alpha$ 为 0.25 的时候相对最符合真实数据

​	构造五种预测矩阵分别对 **打分预测准确性** 和 **推荐质量** 进行评估：

​		（1）**REC_ONES**: 把一定数量（5的数量）的1变成5

​		（2）**REC_FOURS**: 把一定数量（5的数量）的4变成5

​		（3）**ROTATE**: 所有预测值减一，1变成5

​		（4）**SKEWED**: 所有预测值以真实值为中心进行正态采样

​		（5）**COARSENED**: 如果小于等于3，设置为3，否则设置为4

​	！！！此部分没理解透，一个是(2)中的比例怎么得到的，另一个是(3)到底啥意思。。再一个是构造过程里rating1比rating5少，怎么得到REC_ONES



> IPS Estimator 偏移量上限及相关内容理论部分

​	略过，暂时不看



> 倾向性估计模型

​	Propensity Estimation via Naive Bayes

​	Propensity Estimation via Logistic Regression





#### 4. 代码细节

​	python代码，





