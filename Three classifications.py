from sklearn import linear_model
from readmat import readmat
import numpy as np
import pandas as pd

HCdatapath='C:\\Users\\73150\\Desktop\\MALSAR-master\\specific(0.9,0.1,1000)264三分类\\HC'
NCdatapath='C:\\Users\\73150\\Desktop\\MALSAR-master\\specific(0.9,0.1,1000)264三分类\\NC'
MCIdatapath='C:\\Users\\73150\\Desktop\\MALSAR-master\\specific(0.9,0.1,1000)264三分类\\MCI'
HCcategory='HC'
NCcategory='NC'
MCIcategory = 'MCI'
HCdatamatrix, HClabelmatrix = readmat(HCdatapath,HCcategory)
NCdatamatrix, NClabelmatrix = readmat(NCdatapath,NCcategory)
MCIdatamatrix, MCIlabelmatrix = readmat(MCIdatapath,MCIcategory)

# 选择矩阵全部特征
for i in range(0,np.shape(HCdatamatrix)[0]):
    HCiterdatamatrix = HCdatamatrix[i][0]
    for m in range(1,np.shape(HCdatamatrix)[1]):
        HCiterdatamatrix = np.hstack([HCiterdatamatrix,HCdatamatrix[i][m]])
    if i == 0:
        HCiterdatamatrix0 = HCiterdatamatrix
    else:
        HCiterdatamatrix0 = np.vstack([HCiterdatamatrix0,HCiterdatamatrix])

for i in range(0,np.shape(NCdatamatrix)[0]):
    NCiterdatamatrix = NCdatamatrix[i][0]
    for m in range(1,np.shape(NCdatamatrix)[1]):
        NCiterdatamatrix = np.hstack([NCiterdatamatrix,NCdatamatrix[i][m]])
    if i == 0:
        NCiterdatamatrix0 = NCiterdatamatrix
    else:
        NCiterdatamatrix0 = np.vstack([NCiterdatamatrix0,NCiterdatamatrix])

for i in range(0,np.shape(MCIdatamatrix)[0]):
    MCIiterdatamatrix = MCIdatamatrix[i][0]
    for m in range(1,np.shape(MCIdatamatrix)[1]):
        MCIiterdatamatrix = np.hstack([MCIiterdatamatrix,MCIdatamatrix[i][m]])
    if i == 0:
        MCIiterdatamatrix0 = MCIiterdatamatrix
    else:
        MCIiterdatamatrix0 = np.vstack([MCIiterdatamatrix0,MCIiterdatamatrix])

# 选择矩阵下半特征
# for i in range(0, np.shape(HCdatamatrix)[0]):
#     HCiterdatamatrix = HCdatamatrix[i][1][0]
#     for m in range(2, np.shape(HCdatamatrix)[1]):
#         for a in range(0,m):
#             HCiterdatamatrix = np.hstack([HCiterdatamatrix, HCdatamatrix[i][m][a]])
#     if i == 0:
#         HCiterdatamatrix0 = HCiterdatamatrix
#     else:
#         HCiterdatamatrix0 = np.vstack([HCiterdatamatrix0, HCiterdatamatrix])
#
# for i in range(0, np.shape(NCdatamatrix)[0]):
#     NCiterdatamatrix = NCdatamatrix[i][1][0]
#     for m in range(2, np.shape(NCdatamatrix)[1]):
#         for a in range(0,m):
#             NCiterdatamatrix = np.hstack([NCiterdatamatrix, NCdatamatrix[i][m][a]])
#     if i == 0:
#         NCiterdatamatrix0 = NCiterdatamatrix
#     else:
#         NCiterdatamatrix0 = np.vstack([NCiterdatamatrix0, NCiterdatamatrix])
#
# for i in range(0, np.shape(MCIdatamatrix)[0]):
#     MCIiterdatamatrix = MCIdatamatrix[i][1][0]
#     for m in range(2, np.shape(MCIdatamatrix)[1]):
#         for a in range(0,m):
#             MCIiterdatamatrix = np.hstack([MCIiterdatamatrix, MCIdatamatrix[i][m][a]])
#     if i == 0:
#         MCIiterdatamatrix0 = MCIiterdatamatrix
#     else:
#         MCIiterdatamatrix0 = np.vstack([MCIiterdatamatrix0, MCIiterdatamatrix])

# 选择矩阵上半部分
# for i in range(0, np.shape(HCdatamatrix)[0]):
#     HCiterdatamatrix = HCdatamatrix[i][158][0]
#     for m in range(2, np.shape(HCdatamatrix)[1]):
#         for a in range(0,m):
#             HCiterdatamatrix = np.hstack([HCiterdatamatrix, HCdatamatrix[i][159-m][159-a]])
#     if i == 0:
#         HCiterdatamatrix0 = HCiterdatamatrix
#     else:
#         HCiterdatamatrix0 = np.vstack([HCiterdatamatrix0, HCiterdatamatrix])
# #
# for i in range(0, np.shape(NCdatamatrix)[0]):
#     NCiterdatamatrix = NCdatamatrix[i][158][0]
#     for m in range(2, np.shape(NCdatamatrix)[1]):
#         for a in range(0,m):
#             NCiterdatamatrix = np.hstack([NCiterdatamatrix, NCdatamatrix[i][159-m][159-a]])
#     if i == 0:
#         NCiterdatamatrix0 = NCiterdatamatrix
#     else:
#         NCiterdatamatrix0 = np.vstack([NCiterdatamatrix0, NCiterdatamatrix])
#
# for i in range(0, np.shape(MCIdatamatrix)[0]):
#     MCIiterdatamatrix = MCIdatamatrix[i][158][0]
#     for m in range(2, np.shape(MCIdatamatrix)[1]):
#         for a in range(0,m):
#             MCIiterdatamatrix = np.hstack([MCIiterdatamatrix, MCIdatamatrix[i][159-m][159-a]])
#     if i == 0:
#         MCIiterdatamatrix0 = MCIiterdatamatrix
#     else:
#         MCIiterdatamatrix0 = np.vstack([MCIiterdatamatrix0, MCIiterdatamatrix])

label1 = np.zeros((84,1))
label2 = np.ones((40,1))
label3 = label2 * 2

finaldata = np.vstack([HCiterdatamatrix0,MCIiterdatamatrix0[0:40],NCiterdatamatrix0[0:40]])
finallabel = np.vstack([label1,label2,label3]).ravel()

# 过采样
from imblearn.over_sampling import SMOTE
# sample = SMOTE(random_state=49)
# finaldata, finallabel = sample.fit_resample(finaldata,finallabel)
# print(finaldata.shape,finallabel.shape)

# 数据特征重要性筛选及模型训练
from sklearn.feature_selection import SelectFromModel
from foldclassify import RFClassifier,AdaBoost,svclassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import Lasso,Ridge
import pandas as pd
from sklearn.svm import SVC,LinearSVC
# feat_label = pd.DataFrame(finaldata).columns

# RF 特征选取
acc=[]
# clf = RandomForestClassifier(n_estimators=2000).fit(finaldata,finallabel)
# for i in np.linspace(0.00045,0.00053,10):
#     sf = SelectFromModel(estimator = clf ,threshold = i)
#     x_selected = sf.fit_transform(finaldata,finallabel)
#     print(x_selected.shape)
#     acclist = RFClassifier(x_selected,finallabel)
#     acc.append(np.mean(acclist))
# sf = SelectFromModel(estimator=clf, threshold=0.00035)
# x_selected = sf.fit_transform(finaldata, finallabel)
# # importance = clf.feature_importances_
# res_list = np.argsort(importance)[::-1]


# SVC 特征选取
svc = SVC(kernel = "linear", C = 1.2,gamma = 1.5)
svc.fit(finaldata,finallabel)
model = SelectFromModel(svc, prefit =True)
imp = np.argsort(svc.coef_)[0,0:80]
index = model.get_support(indices = True)
x_selected = np.array(pd.DataFrame(finaldata[:,imp]))

# 特征重要性排序
fea_dict = {}
list1 = []
list2 = []
for i in range(30):
    clf = RandomForestClassifier(n_estimators=15000).fit(finaldata,finallabel)
    importance = clf.feature_importances_.tolist()
    res_list = np.argsort(importance)[::-1][:30].tolist()
    for i in res_list:
        if i not in fea_dict:
            fea_dict[i] = importance[i]
        else:
            fea_dict[i] = (fea_dict[i] + importance[i])/2
print(sorted(fea_dict.items(),key = lambda x:x[1], reverse = True)[0:20])

from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from split_sets_10_fold import Split_Sets_10_Fold
from sklearn.metrics import accuracy_score,recall_score
train_index1, test_index1 = Split_Sets_10_Fold(10, x_selected)
train_index2, test_index2 = Split_Sets_10_Fold(10,label1)
train_index3, test_index3 = Split_Sets_10_Fold(10,label2)
# print(train_index2,test_index2)
# print(train_index3,test_index3)

# 三分类
# acclist = []
# for i in range(0, 10):
#     train_data = x_selected[train_index1[i], :, ]
#     test_data = x_selected[test_index1[i], :, ]
#     # train_label = np.vstack([label1[train_index2[i]],label2[train_index3[i]],label3[train_index3[i]]]).ravel()
#     # test_label = np.vstack([label1[test_index2[i]],label2[test_index3[i]],label3[test_index3[i]]]).ravel()
#     train_label = finallabel[train_index1[i]]
#     test_label = finallabel[test_index1[i]]
#     # dtc = OneVsRestClassifier(SVC(kernel = "linear"))
#     # dtc = RandomForestClassifier(n_estimators=2000)
#     tree = DecisionTreeClassifier(criterion="gini",
#                                   random_state=0,
#                                   splitter="best",
#                                   min_samples_leaf=6,
#                                   min_samples_split=6)
#     dtc = AdaBoostClassifier(base_estimator=tree,
#                                  random_state=0,
#                                  n_estimators=1000,
#                                  algorithm="SAMME",
#                                  learning_rate=0.05)
#     dtc.fit(train_data, train_label)
#     predict_label = dtc.predict(test_data)
#     acclist.append(accuracy_score(test_label, predict_label))

# 二分类
from sklearn.svm import LinearSVC
# acclist = []
# for i in range(0, 10):
#     train_data = x_selected[train_index1[i], :, ]
#     train_label = finallabel[train_index1[i]]
#     test_data = x_selected[test_index1[i], :, ]
#     test_label = finallabel[test_index1[i]]
#     dtc = LinearSVC(penalty = 'l2')
#     # dtc = RandomForestClassifier(n_estimators = 2000,
#     #                              min_samples_leaf = 3,
#     #                              min_samples_split = 6)
#     dtc.fit(train_data, train_label)
#     predict_label = dtc.predict(test_data)
#     acclist.append(accuracy_score(test_label, predict_label))
#
# print(acclist)
# print(np.mean(acclist))

# print(max(acc))
# print("ave:",np.mean(acc),"std:",np.std(acc))

