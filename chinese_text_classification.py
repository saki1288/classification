import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
import time

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    """加载数据"""
    data = []      # 存储文本内容
    labels = []    # 存储标签
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            label, text = line.strip().split('\t')
            data.append(text)
            labels.append(label)
    return data, labels

def cut_words(text):
    """分词函数"""
    return ' '.join(jieba.cut(text))

def main():
    # 读取数据
    print("正在加载数据...")
    texts, labels = load_data('filtered_cnews.train.txt')
    print(f'数据集大小: {len(texts)}')

    # 分词
    print("\n正在进行分词...")
    texts_cut = [cut_words(text) for text in texts]
    print('分词示例：')
    print(texts_cut[0])

    # 划分训练集和测试集
    print("\n正在划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts_cut, labels, 
        test_size=0.2,    # 测试集占20%
        random_state=42   # 随机种子，保证结果可复现
    )
    print(f'训练集大小: {len(X_train)}')
    print(f'测试集大小: {len(X_test)}')

    # 特征提取
    print("\n正在进行特征提取...")
    tfidf = TfidfVectorizer(max_features=5000)  # 创建TF-IDF转换器，最多5000个特征
    X_train_tfidf = tfidf.fit_transform(X_train)  # 训练并转换训练集
    X_test_tfidf = tfidf.transform(X_test)        # 转换测试集
    print(f'特征维度: {X_train_tfidf.shape}')

    # 参数优化
    print("\n开始参数优化...")
    # 定义参数网格
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],  # 平滑参数
        'fit_prior': [True, False]  # 是否学习类别先验概率
    }
    
    # 创建评分器
    scorer = make_scorer(f1_score, average='macro')
    
    # 创建网格搜索对象
    grid_search = GridSearchCV(
        MultinomialNB(),
        param_grid,
        cv=5,  # 5折交叉验证
        scoring=scorer,
        n_jobs=-1,  # 使用所有CPU核心
        verbose=1
    )
    
    # 开始网格搜索
    start_time = time.time()
    grid_search.fit(X_train_tfidf, y_train)
    end_time = time.time()
    
    print(f"\n参数优化完成，耗时: {end_time - start_time:.2f}秒")
    print("\n最佳参数:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    print(f"\n最佳得分: {grid_search.best_score_:.4f}")

    # 使用最佳参数的模型进行预测
    print("\n使用最佳参数进行预测...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_tfidf)

    # 计算评估指标
    print("\n计算评估指标...")
    # 计算宏平均
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, 
        average='macro'  # 使用宏平均
    )
    print('宏平均：')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

    # 计算微平均
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, 
        average='micro'  # 使用微平均
    )
    print('\n微平均：')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

    # 绘制混淆矩阵
    print("\n正在绘制混淆矩阵...")
    cm = confusion_matrix(y_test, y_pred)  # 计算混淆矩阵
    plt.figure(figsize=(10, 8))  # 创建图形
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # 绘制热力图
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig('confusion_matrix.png')  # 保存图片
    plt.close()

    # 打印详细的分类报告
    print("\n生成分类报告...")
    print(classification_report(y_test, y_pred))  # 打印分类报告

if __name__ == "__main__":
    main()