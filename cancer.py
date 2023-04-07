from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 导入肺癌数据集
data = load_breast_cancer()
X = data['data']
y = data['target']
print(X.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 建立支持向量机模型
clf1 = SVC(kernel='linear')  # 线性核函数
clf2 = SVC(kernel='rbf', C=10, gamma=0.0001)  # 采用高斯核函数
clf1.fit(x_train, y_train)
clf2.fit(x_train, y_train)

# 输出两种支持向量机模型的精度。此处填入你的代码。（1）
