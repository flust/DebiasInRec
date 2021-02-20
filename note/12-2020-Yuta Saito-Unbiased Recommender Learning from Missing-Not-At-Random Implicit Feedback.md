## 2020-Yuta Saito-Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback

#### 1. 解决问题

​	推荐系统广泛使用隐性反馈比如 click 来进行训练学习，但未点击的数据并不意味着用户的负面反馈，反而更可能是产品并没有暴露给用户（positive-unlabeled problem），这给预测用户偏好带来了困难，过去的研究通过 **修改正面反馈的权重** 或 **EM算法评估置信度** 来解决这个问题。但是，这些方法没办法解决 missing-not-at-random 问题（流行度高的产品更可能被点击即使用户对他们没有特别有兴趣）。

​	为了解决这个问题，本文首先提出了理想的损失函数去实现最大化相关度的推荐，并对该理想损失提出了无偏的估计，随后，分析了所提出的无偏估计器的方差并进一步提出了一个新的估计器（clipped estimator）。

​	偏差权衡？bias-variance trade-off

​	本文在半合成数据集和真实数据集上实验，所提出的方法比 baseline 有大幅度提高，特别的，对于出现概率小的产品效果更好。这个发现表明本文提出的方法更好地达成了 **推荐最高相关度产品** 这一目标。

​	

#### 2. 主要方法

> 提出一个 理想的损失函数 及其 无偏估计



> 解决方差过大问题 a variance reduction estimator





#### 3. 技术细节

> 符号说明

​	**Y** : click matrix （1表示正例 / 0表示负例或无反馈）

​	**R ** : relevance matrix （表示用户$u$和产品$i$的相关度）

​	**O** : exposure matrix （表示产品$i$是否暴露给用户$u$）

​	$\theta$ : exposure parameters

​	$\gamma$ : relevance parameters

​		$Y_{u,i} = O_{u,i} \cdot R_{u,i}$

​		$P(Y_{u,i} = 1) = \theta_{u,i} \cdot \gamma_{u,i}$



$L_{ideal}(\hat{R}) = \frac{1}{|D|}\sum_{(u,i)\in D}[{Y_{u,i}}\delta^{(1)}(\hat{R}_{u,i}) + (1 - {Y_{u,i}})\delta^{(0)}(\hat{R}_{u,i})]$

$\hat{L}_{unbiased}(\hat{R}) = \frac{1}{|D|}\sum_{(u,i)\in D}[\frac{Y_{u,i}}{\theta_{u,i}}\delta_{u,i}^{(1)} + (1 -\frac{Y_{u,i}}{\theta_{u,i}})\delta_{u,i}^{(0)}]$



$\hat{\theta}_{*,i} = (\frac{\sum_{u \in \mathrm{U}}Y_{u,i}}{max_{i \in I}\sum_{u \in U} Y_{u, i}}) ^ \eta$ （item popularity）





#### 4. 代码细节

​	数据处理过后

columns |     0     |     1     |     2     |      3       |       4       |           5            |            6            |

factors |  user_id  |  item_id  | click (Y) | exposure (O) | relevance (R) | exposure param (theta) | relevance param (gamma) |

