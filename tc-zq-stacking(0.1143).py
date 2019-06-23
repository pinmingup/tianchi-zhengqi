# 导入包
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPRegressor

# 加载数据集
data_train = pd.read_csv('C:/Users/dell/Desktop/tianchidasai/zhengqi/data/train.csv')
data_test = pd.read_csv('C:/Users/dell/Desktop/tianchidasai/zhengqi/data/test.csv')
# 特征集
feature = ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
               'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29',
               'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37']

# # 计算特征方差筛选
re = []
for i in range(38):
    tmp = np.array(data_train[feature[i]])
    if np.var(tmp) < 0.75:
        re.append('V'+str(i))
    # print('V'+str(i))
# 移除方差较小(<0.75)的特征
re.append('V31')
for i in range(len(re)):
    if re[i] in feature:
        feature.remove(re[i])
print(re)
# 特征预处理
X_train = data_train[feature]
y_train = data_train[['target']]
x_test = data_test[feature]

# new_feature
all_data = pd.concat((X_train, x_test))

all_data['V0-s2'] = all_data['V0']**2
all_data['V0-s3'] = all_data['V0']**3

all_data['V1-s2'] = all_data['V1']**2
all_data['V1-s3'] = all_data['V1']**3

all_data['V8-s2'] = all_data['V8']**2
all_data['V8-s3'] = all_data['V8']**3


X_train = all_data[:2888]
print(X_train.shape)
x_test = all_data[2888:]
print(x_test.shape)
# 验证集
trainx, testx, trainy, testy = train_test_split(X_train, y_train, test_size=0.3, random_state=2018)


def get_oof(reg, X_train, y_train, X_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((5, ntest))
    for i, (train_index, test_index)in enumerate(kf.split(X_train)):
        # print(test_index.shape)
        kf_X_train = X_train.iloc[train_index]
        kf_y_train = y_train.iloc[train_index]
        kf_X_test = X_train.iloc[test_index]
        reg.fit(kf_X_train, kf_y_train)
        regpre1 = reg.predict(kf_X_test)
        oof_train[test_index] = np.reshape(regpre1, -1)
        regpre2 = reg.predict(X_test)
        oof_test_skf[i, :] = np.reshape(regpre2, -1)
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# 模型
lr = LinearRegression()
mlpr = MLPRegressor(random_state=2018)
svr = SVR()
rfr = RandomForestRegressor(random_state=2018)
ls = Lasso(random_state=2018, max_iter=500)
rg = Ridge(random_state=2018, max_iter=500)
xgb = XGBRegressor(random_state=2018, max_depth=8)
gbdt = GradientBoostingRegressor(random_state=2018, max_depth=8)
adb = AdaBoostRegressor(random_state=2018)
lgbm = LGBMRegressor(random_state=2018, num_leaves=3)
regs = [lr, mlpr, svr, rfr, ls, rg, xgb, gbdt, adb, lgbm]
# model_level-1

ntrain = data_train.shape[0]  # 2888
ntest = data_test.shape[0]  # 1925
kf = KFold(n_splits=5, random_state=2018)

train_final = pd.DataFrame()
test_final = pd.DataFrame()
i = 0
for reg in regs:
    i += 1
    oof_train, oof_test = get_oof(reg, X_train, y_train, x_test)
    oof_train = pd.DataFrame(oof_train, columns=['x'+str(i)])
    oof_test = pd.DataFrame(oof_test, columns=['x'+str(i)])
    train_final = pd.concat([train_final, oof_train], axis=1)
    test_final = pd.concat([test_final, oof_test], axis=1)
new_train = pd.concat([train_final, data_train['target']], axis=1)
# print(new_train)
new_test = test_final
# 第二层模型
lr = LinearRegression()
svr = SVR()
xgb = XGBRegressor(random_state=2018)

m = lr
fea = []
for i in range(1, len(regs)+1):
    fea.append('x'+str(i))
train_x = new_train[fea]
train_y = new_train['target']
x1, x2, y1, y2 = train_test_split(train_x, train_y, test_size=0.1, random_state=2018)
m.fit(x1, y1)
# valid预测
y_pred = m.predict(x2)
print(metrics.mean_squared_error(y2, y_pred))
# test predict
test = new_test[fea]
target = m.predict(test)
# print(target)
# 写入文件
f = open('zq0527.txt', 'w')
for i in target:
    f.write(str(i)+'\n')
f.close()
