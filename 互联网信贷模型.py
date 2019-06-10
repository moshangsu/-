# 导入各种包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, Imputer
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 导入数据及预览
data = pd.read_csv(open("../dataSet/LoanStats3d.csv", 'rb'), low_memory=False, skiprows=[0])
# 后几行是备注信息，删掉
data = data.drop(data.index[-4:], axis=0)
data.head()

# 筛选获得每列至少有2个分类特征的数组集，只有一个的无用，删去
data = data.loc[:, data.apply(pd.Series.nunique) != 1]

#统计每列的缺失值情况
check_null = data.isnull().sum(axis=0).sort_values(ascending=False)/float(len(data)) #查看缺失值比例
# check_null[check_null > 0.2] # 查看缺失比例大于20%的属性。

# 漫长的数据清洗过程
# 去除无用的字符
data['int_rate'] = data.apply(lambda x: x['int_rate'].replace('%', ''), axis=1)
data['revol_util'] = data.apply(lambda x: str(x['revol_util']).replace('%', ''), axis=1)
data['term'] = data.apply(lambda x: str(x['term']).replace(' months', ''), axis=1)

# 删除缺失值超过20%的列
thresh_count = len(data) * 0.2 # 设定阀值
data = data.drop(check_null[check_null > 0.2].index, axis=1) #若某一列数据缺失的数量超过阀值就会被删除
data.info()

# 初步了解“Object”变量概况
pd.set_option('display.max_rows',None)
# data.select_dtypes(include=['object']).describe().T

# 删除无效特征，几乎没有重复的，删掉
data = data.drop(['emp_title', 'title', 'revol_util'], axis=1)

# 漫长的数据清洗过程
# 构建mapping，对有序变量'emp_length”、“grade”、"sub_grade"进行转换
mapping_dict = {
    'emp_length': {
        '10+ years': 10,
        '9 years': 9,
        '8 years': 8,
        '7 years': 7,
        '6 years': 6,
        '5 years': 5,
        '4 years': 4,
        '3 years': 3,
        '2 years': 2,
        '1 year': 1,
        '< 1 year': 0,
        'n/a': 0
    },
    'grade':{
        'A': 1,
        'B': 2,
        'C': 3,
        'D': 4,
        'E': 5,
        'F': 6,
        'G': 7
    },
    'sub_grade':{
        'A1': 1,
        'A2': 2,
        'A3': 3,
        'A4': 4,
        'A5': 5,
        'B1': 6,
        'B2': 7,
        'B3': 8,
        'B4': 9,
        'B5': 10,
        'C1': 11,
        'C2': 12,
        'C3': 13,
        'C4': 14,
        'C5': 15,
        'E1': 16,
        'E2': 17,
        'E3': 18,
        'E4': 19,
        'E5': 20,
        'F1': 21,
        'F2': 22,
        'F3': 23,
        'F4': 24,
        'F5': 25,
        'G1': 26,
        'G2': 27,
        'G3': 28,
        'G4': 29,
        'G5': 30,
    }
}

data = data.replace(mapping_dict) #变量映射

#使用Pandas replace函数定义新函数：
def coding(col, codeDict):
    colCoded = pd.Series(col, copy=True)
    for key, value in codeDict.items():
        colCoded.replace(key, value, inplace=True)
    return colCoded

#把贷款状态LoanStatus编码为违约=1, 正常=0:
data['loan_status'] = coding(data['loan_status'], {'Current':0,'Fully Paid': 0\
                                                     , 'In Grace Period': 1\
                                                     , 'Late (31-120 days)': 1\
                                                     , 'Late (16-30 days)': 1\
                                                     , 'Charged Off': 1\
                                                     , "Issued": 1\
                                                     , "Default": 1\
                                                    , "Does not meet the credit policy. Status:Fully Paid": 1\
                                         , "Does not meet the credit policy. Status:Charged Off": 1})
#统计"loan_status"数据的分布
data['loan_status'].value_counts()


#按缺失值比例从大到小排列
data.isnull().sum(axis=0).sort_values(ascending=False)/float(len(data))

# 空值选择填充
data['emp_length'].fillna(data['emp_length'].mode()[0]) # 工作年限使用众数填充

objectColumns = data.select_dtypes(include=['object']).columns # 筛选数据类型为object的数据
data[objectColumns] = data[objectColumns].fillna('None') # 以分类“None”填充缺失值

# 利用sklearn模块中的Imputer模块填充缺失值
numColumns = data.select_dtypes(include=[np.number]).columns
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)  # 针对axis=0 列来处理
imr = imr.fit(data[numColumns])
data[numColumns] = imr.transform(data[numColumns])

data = pd.get_dummies(data).reset_index(drop=True)

# 特征缩放
# 采用标准化的方法进行去量纲操作，加快算法收敛速度，采用scikit-learn模块preprocessing的子模块StandardScaler进行操作。
col = data.select_dtypes(include=['int64','float64']).columns
col = col.drop('loan_status') #剔除目标变量
dataStd = data # 复制数据至变量loans_ml_df

sc =StandardScaler() # 初始化缩放器
dataStd[col] =sc.fit_transform(dataStd[col]) #对数据进行标准化
dataStd.head() #查看经标准化后的数据

#构建X特征变量和Y目标变量
y = dataStd['loan_status']
X = dataStd.drop(['loan_status'], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 将原始数据划分为测试集和训练集
print(len(X)) # 查看初始特征集合的数量


# 特征的选择优先选取与预测目标相关性较高的特征，不相关特征可能会降低分类的准确率，因此为了增强模型的泛化能力，
# 我们需要从原有特征集合中挑选出最佳的部分特征，并且降低学习的难度，能够简化分类器的计算，同时帮助了解分类问题的因果关系。
names = dataStd.columns
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=10,random_state=123) # 构建分类随机森林分类器
clf.fit(x_train, y_train) # 对自变量和因变量进行拟合
print(names, clf.feature_importances_)


# 筛选特征
res = []
for feature in zip(clf.feature_importances_, names):
    res.append(feature)
res = sorted(res)
res = res[-13:]
res = [row[1] for row in res]

# 构建逻辑回归分类器
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression()
clf1.fit(x_train[res], y_train)

# 评估模型效果
predicted1 = clf1.predict(x_test[res]) # 通过分类器产生预测结果
from sklearn.metrics import accuracy_score
print("Test set accuracy score: %.2f%%" % (accuracy_score(predicted1, y_test) * 100))



