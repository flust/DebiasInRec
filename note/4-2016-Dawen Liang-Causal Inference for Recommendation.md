## 2016-Dawen Liang-Causal Inference for Recommendation

#### 1. 解决问题

​	传统的推荐系统只考虑了 click 数据，没有考虑 exposure 信息，这导致了偏差，使得用户不是完全随机的考虑产品，本文使用因果推论方法去纠正这个偏差。

​	

#### 2. 主要方法

> 建立曝光模型 exposure model 计算倾向性得分 propensity score（两种方法）

 1. Bernoulli 伯努利分布 （基于流行度）

    倾向性得分为 $p_{ui} = \hat\rho_i $ （直接为item的出现概率、流行度？）

 2. Poisson factorization 泊松因子分解 （基于个性化偏好）

    倾向性得分为 $p_{ui} = 1 - e^{-E_{q}[\pi_u^T\gamma_i]} $ （用向量进行计算？）



> 建立点击模型 click model 预测点击概率，其中损失函数中用到上述倾向性评分 （两种方法）

​	使用标准 MF 方法，在被曝光的条件下，点击概率服从正态分布

​	具体算法：inverse propensity weighted matrix factorization（IPW-MF）

​	损失函数为 $L = \sum_{(u,i)\in O}\frac{1}{p_{ui}}(y_{ui} - \theta_u^T\beta_i)^2 + \lambda_\theta\sum_u||\theta_u||^2_2 + \lambda_\beta\sum_i||\beta_i||^2$

 1. 直接使用曝光后点击期望

    $E[y_{ui}|a_{ui} = 1, \mathrm{D}]$

 2. 使用曝光后点击概率乘以曝光期望（曝光概率 乘以 (1) ）

    $E[y_{ui}|\mathrm{D}] = P(a_{ui} = 1|\pi_{ui},\mathrm{D}) \times E[y_{ui}|a_{ui} = 1, \mathrm{D}]$



> 两类 baseline

	1. 使用逆倾向性权重
 	2. 传统方法



#### 3. 技术细节

> 数据处理部分 Data pre-processing （MovieLens / Yahoo-R3 / ArXiv）

​	对于每个数据集，建立两种划分方式 regular(REG) / skewed(SKEW)

 1. regular(REG) 

    对 **每个用户按照 70/10/20 的比例随机划分exposed items **

    这使得测试集与训练集和验证集同分布，这也是一般的研究者评估推荐模型的方式

 2. skewed(SKEW)

    重新平衡了划分方式以更好的近似**干预**

    首先 **随机采样 20% **作为测试集合，使每种产品具有 **相同的出现概率**

    接下来 **按照REG的方式，按照 70/10 的比例划分训练集和验证集**

    这使得测试集具有与训练集完全不同的曝光分布（使用此种划分方式来说明因果推理方法具有更好的性能）

（产品流行度直接使用它的出现次数）



> 评估指标 Metrics

 1. 评估 exposure model:

    **predictive log-likelihood**

 2. 评估 click model: 

    **Predictive log tail probability（PLP）**

    **Mean Average Rank**



#### 4. 代码细节

​	还没找到代码