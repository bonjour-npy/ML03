import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# 导入数据，地址根据文件地址更改
df = pd.read_csv('data/HR-Employee-Attrition.csv')
df.head()

# 数据预处理
df.nunique().nsmallest(10)  # 无效列值检查
# EmployeeCount, Over18 and StandardHours 这些属性值完全相同
# 删除无效列值
df.drop(['StandardHours', 'EmployeeCount', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)
df.isnull().sum()  # 缺失值检查
df[df.duplicated()]  # 重复值检查
df.head()

# 中文编码为数字
# LabelEncoder对非数字格式列编码
dtypes_list = df.dtypes.values  # 取出各列的数据类型
columns__list = df.columns  # 取出各列列名
# 循环遍历每一列找出非数字格式进行编码
for i in range(len(columns__list)):
    if dtypes_list[i] == 'object':  # 判断类型
        lb = LabelEncoder()  # 导入LabelEncoder模型
        lb.fit(df[columns__list[i]])  # 训练
        df[columns__list[i]] = lb.transform(df[columns__list[i]])  # 编码\
df.head()

# 查看各列相关性
corr = df.corr()
corr.head()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
plt.show()

# 黑色负相关,白色正相关
'''
通过热力图下，可以看到 
MonthlyIncome 与 JobLevel 相关性较强；
TotalWorkingYears 与 JobLevel 相关性较强；
TotalWorkingYears 与 MonthlyIncome 相关性较强；
PercentSalaryHike 与 PerformanceRating 相关性较强；
YearsInCurrentRole 与 YearsAtCompany 相关性较强；
YearsWithCurrManager 与 YearsAtCompany 相关性较强；
StockOptionLevel与MaritalStatus成负相关，删除其中一列
'''
df.drop(['JobLevel', 'TotalWorkingYears', 'YearsInCurrentRole', 'YearsWithCurrManager',
         'PercentSalaryHike', 'StockOptionLevel'], axis=1, inplace=True)
df.head()

# 特征提取
X = df.drop(['Attrition'], axis=1)
y = df['Attrition']

# 标准化数据
sc = StandardScaler()
X = sc.fit_transform(X)
mean = np.mean(X, axis=0)
print('均值')
print(mean)
standard_deviation = np.std(X, axis=0)
print('标准差')
print(standard_deviation)

# 划分数据集
#  此处填入你的代码。(1)
print(X_train)
'''
array([[-0.97717366, -2.41643713,  1.66971077, ...,  0.33809616,
         0.32522752,  1.49386709],
       [-0.64866811,  0.59004834,  1.17389032, ...,  0.33809616,
        -0.98101416, -0.67914568],
       [ 1.32236521,  0.59004834,  1.7044182 , ...,  0.33809616,
         0.16194731, -0.67914568],
       ...,
       [ 0.66535411,  0.59004834,  0.93341741, ...,  0.33809616,
         0.48850773,  1.80429749],
       [ 0.11784485,  0.59004834, -1.32504473, ..., -2.49382042,
        -0.98101416, -0.67914568],
       [ 0.33684855,  0.59004834, -0.35819486, ..., -1.07786213,
        -0.98101416, -0.67914568]])
'''

# 默认
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print('Accuracy Score:')
#  输出Accuracy Score。此处填入你的代码。(2)
'''
Accuracy Score:
0.8641304347826086
'''

# 输出支持向量
print('支持向量:', np.matrix(svc.fit(X_train, y_train).support_vectors_))
'''
支持向量: [[-0.97717366 -2.41643713  1.66971077 ...  0.33809616  0.32522752
   1.49386709]
 [-0.64866811  0.59004834  1.17389032 ...  0.33809616 -0.98101416
  -0.67914568]
 [-0.10115885 -0.91319439  1.23834698 ...  0.33809616 -0.65445374
  -0.67914568]
 ...
 [-1.74368662  0.59004834  1.31767825 ... -1.07786213 -0.98101416
  -0.36871529]
 [ 0.66535411  0.59004834  1.41188414 ... -1.07786213 -0.49117353
  -0.36871529]
 [-0.64866811 -0.91319439  1.5928586  ...  0.33809616  0.48850773
   0.5625759 ]]
'''

# 混淆矩阵
y_pred = svc.predict(X_test)
#  获取混淆矩阵。此处填入你的代码。(3)

class_names = [0, 1]  # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")

# 准确度、精确度和召回率
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
#  输出F1_score。此处填入你的代码。(4)
'''
Accuracy: 0.8641304347826086
Precision: 0.7857142857142857
Recall: 0.1896551724137931
F1_score: 0.3055555555555555
'''

# PR曲线
from sklearn.metrics import precision_recall_curve

y_scores = svc.decision_function(X_test)
plt.figure("P-R Curve")
plt.title('Precision/Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
# y_test为样本实际的类别，y_scores为样本为正例的概率
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
#  绘制recall,precision曲线。此处填入你的代码。(5)
plt.show()

# 绘制ROC曲线
# 模型预测
# sklearn_predict = sklearn_logistic.predict(X_test)
# y得分为模型预测正例的概率
# y_score = sklearn_logistic.predict_proba(X_test)[:,1]

y_scores = svc.decision_function(X_test)

# 计算不同阈值下，fpr和tpr的组合值，其中fpr表示1-Specificity，tpr表示Sensitivity
fpr, tpr, threshold = metrics.roc_curve(y_test, y_scores)

# 计算AUC的值
roc_auc = metrics.auc(fpr, tpr)
# 绘制面积图
plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
# 添加边际线
plt.plot(fpr, tpr, color='black', lw=1)
# 添加对角线
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
# 添加文本信息
plt.text(0.5, 0.3, 'ROC curve (area = %0.2f)' % roc_auc)
# 添加x轴与y轴标签
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
# 显示图形
plt.show()
