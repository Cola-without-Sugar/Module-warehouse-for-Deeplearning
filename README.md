# Module-warehouse-for-Deeplearning
一个存放深度学习中学习到的算法的仓库

## 深度学习中的模块信息

**说明：**此文档针对深度学习中的一些论文阅读得到的一些有趣的思路与模块想法总结而成，旨在能够对模型组成部分有一些了解，并分析各个模块之间对于性能之间提升的原因和思路的探索，文章来源会标注在附录部分，主要的模块思想会整理成文档的形式。

根据一些论文的思想判断，深度学习的基础框架如图所示：

![深度学习中的模块化改进](https://cdn.jsdelivr.net/gh/Cola-without-Sugar/markdown_img/202302221729277.png)

如图片中所述，图像的模块化处理主要分为对图像的预处理部分、数据集处理部分、主干特征提取网络、权重共享与更新模块、特征融合模块与输出编码分类器模块。

* **图像的预处理**  包括图像增强，图像降噪去噪，图像的去雾去雨算法等对图像做一些基础的分析部分。
* **数据集的划分**  数据集的划分主要涉及，数据集的划分方式，标注方式与精度等方式。
* **主干特征提取网络** 一个基本深度训练网络的架构组成。方向主要有：全局信息与局部信息的感知、针对特定领域的特定模块
* **权重共享与更新模块** 主要涉及对提取信息的共享能力，损失函数决定了网络权重的优化方向。优化器则是寻找模型最优点的能力。
* **特征融合模块** 主要涉及到图像对于低层语义的感知能力
* **输出编码分类器** 主要涉及到对于具体分类任务的优化
* **其他** 



## 文件夹目录信息

|-main

​	|-Image Process												 --图像预处理

​	|-Dataset Partition											 --数据集划分

​	|-Backbone Feature Extraction Network	   --主干特征提取网络

​	|-Weight Sharing And Updating Module       --权重共享与更新模块

​	|-Feature Fusion Module								 --特征融合模块

​	|-Output Encoding Classifier						   --输出编码分类器

​	|-Remaining Items											--其余项

​	-directions.md                                                   --说明文档

|-README.md

## 更新日志

2023.3.21	 更新文件夹目录及存储格式，管理更清晰，希望能督促自己！

2023.3.22 	更新了卷积层的相关信息及实现方式。

2023.3.23	 更新了基础卷积块的pytoch实现与细节原理解释。
