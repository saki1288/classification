# 中文文本分类项目

本项目使用scikit-learn和jieba分词对THUCNews数据集进行中文文本分类，并通过网格搜索进行参数优化。

## 项目特点

- 使用jieba进行中文分词
- 使用TF-IDF进行特征提取
- 使用朴素贝叶斯模型进行分类
- 通过网格搜索进行参数优化
- 提供详细的评估指标和可视化结果

## 项目结构

```
.
├── README.md                 # 项目说明文档
├── requirements.txt          # 项目依赖
├── .gitignore               # Git忽略文件
├── chinese_text_classification.ipynb  # 主要代码文件
└── filtered_cnews.train.txt  # 训练数据集
```

## 环境要求

- Python 3.8+
- 依赖包：
  - pandas
  - numpy
  - scikit-learn
  - jieba
  - matplotlib
  - seaborn


```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

1. 使用VSCode打开项目
2. 安装Jupyter扩展
3. 打开 `chinese_text_classification.ipynb`
4. 运行所有单元格即可看到完整的分析结果


## 项目功能

1. 数据加载和预处理
2. 文本分词
3. 特征提取（TF-IDF）
4. 参数优化（网格搜索）
5. 模型训练和预测
6. 模型评估
   - 宏平均和微平均
   - 混淆矩阵可视化
   - 详细分类报告

## 评估指标

- Precision（精确率）
- Recall（召回率）
- F1-score（F1分数）
- 混淆矩阵可视化

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目。
