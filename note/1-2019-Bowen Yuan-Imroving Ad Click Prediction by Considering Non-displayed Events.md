## 2019-Bowen Yuan-Imroving Ad Click Prediction by Considering Non-displayed Events

#### 1. 解决问题

​	目前大部分CTR预测方法将CTR预测问题视作二分类问题，**点击** / **未点击** 分别视作 **正** / **负** 例，但是在 **展示** / **未展示** 项目之间存在选择偏差（未展示的项目不会有 feedback），忽略掉偏差会导致预测不准确，所以需要找到方法减轻**选择偏差**——反事实学习方法。

​	关键点在于如何处理 non-displayed 的数据，如何为这些数据设置标签

​	目前现有的反事实学习方法直接应用在现实的广告系统中也存在困难（计算角度），也提出了一个新的框架去解决这个问题。



#### 2. 主要方法

​	利用无偏的集合$S_{t}$计算相应的 propensity score，去进行去偏



#### 3. 技术细节

	1. Direct Method
 	2. IPS Method
 	3. Doubly



#### 4. 代码细节

​	c++代码