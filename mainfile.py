from sklearn import linear_model
from readmat import readmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

HCdatapath='C:\\Users\\73150\\Desktop\\MALSAR-master\\specific(0.9,0.1,1000)160二分类\\HC'
OSAdatapath='C:\\Users\\73150\\Desktop\\MALSAR-master\\specific(0.9,0.1,1000)160二分类\\OSA'
HCcategory='HC'
OSAcategory='OSA'
HCdatamatrix, HClabelmatrix = readmat(HCdatapath,HCcategory)
OSAdatamatrix, OSAlabelmatrix = readmat(OSAdatapath,OSAcategory)

# 选择矩阵全部特征
# for i in range(0,np.shape(HCdatamatrix)[0]):
#     HCiterdatamatrix = HCdatamatrix[i][0]
#     for m in range(1,np.shape(HCdatamatrix)[1]):
#         HCiterdatamatrix = np.hstack([HCiterdatamatrix,HCdatamatrix[i][m]])
#     if i == 0:
#         HCiterdatamatrix0 = HCiterdatamatrix
#     else:
#         HCiterdatamatrix0 = np.vstack([HCiterdatamatrix0,HCiterdatamatrix])
#
# for i in range(0,np.shape(OSAdatamatrix)[0]):
#     OSAiterdatamatrix = OSAdatamatrix[i][0]
#     for m in range(1,np.shape(OSAdatamatrix)[1]):
#         OSAiterdatamatrix = np.hstack([OSAiterdatamatrix,OSAdatamatrix[i][m]])
#     if i == 0:
#         OSAiterdatamatrix0 = OSAiterdatamatrix
#     else:
#         OSAiterdatamatrix0 = np.vstack([OSAiterdatamatrix0,OSAiterdatamatrix])

# 选择矩阵下半部分特征
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
# for i in range(0, np.shape(OSAdatamatrix)[0]):
#     OSAiterdatamatrix = OSAdatamatrix[i][1][0]
#     for m in range(2, np.shape(OSAdatamatrix)[1]):
#         for a in range(0,m):
#             OSAiterdatamatrix = np.hstack([OSAiterdatamatrix, OSAdatamatrix[i][m][a]])
#     if i == 0:
#         OSAiterdatamatrix0 = OSAiterdatamatrix
#     else:
#         OSAiterdatamatrix0 = np.vstack([OSAiterdatamatrix0, OSAiterdatamatrix])

# 选择矩阵上半部分特征
for i in range(0, np.shape(HCdatamatrix)[0]):
    HCiterdatamatrix = HCdatamatrix[i][158][159]
    for m in range(2, np.shape(HCdatamatrix)[1]):
        for a in range(0,m):
            HCiterdatamatrix = np.hstack([HCiterdatamatrix, HCdatamatrix[i][159-m][159-a]])
    if i == 0:
        HCiterdatamatrix0 = HCiterdatamatrix
    else:
        HCiterdatamatrix0 = np.vstack([HCiterdatamatrix0, HCiterdatamatrix])

for i in range(0, np.shape(OSAdatamatrix)[0]):
    OSAiterdatamatrix = OSAdatamatrix[i][158][159]
    for m in range(2, np.shape(OSAdatamatrix)[1]):
        for a in range(0,m):
            OSAiterdatamatrix = np.hstack([OSAiterdatamatrix, OSAdatamatrix[i][159-m][159-a]])
    if i == 0:
        OSAiterdatamatrix0 = OSAiterdatamatrix
    else:
        OSAiterdatamatrix0 = np.vstack([OSAiterdatamatrix0, OSAiterdatamatrix])

label1 = np.zeros((84,1))
label2 = np.ones((82,1))
finaldata = np.vstack([HCiterdatamatrix0,OSAiterdatamatrix0])
finallabel = np.vstack([label1,label2]).ravel()
print(finaldata.shape)
# print(finallabel.shape)
# print(np.std(finaldata))

# 数据特征重要性筛选及模型训练
from sklearn.feature_selection import SelectFromModel
from foldclassify import RFClassifier,AdaBoost,svclassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso,Ridge
import pandas as pd
# feat_label = pd.DataFrame(finaldata).columns
acc = []

# # RF 特征选取
# feature_label_1 = []
clf = RandomForestClassifier(n_estimators=15000).fit(finaldata,finallabel)
# # for i in np.linspace(0.00030,0.00035,10):
# #     sf = SelectFromModel(estimator = clf ,threshold = i)
# #     x_selected = sf.fit_transform(finaldata,finallabel)
# #     print(x_selected.shape)
# #     acclist = RFClassifier(x_selected,finallabel)
# #     acc.append(np.mean(acclist))
# sf = SelectFromModel(estimator=clf, threshold=0.00035)
# x_selected = sf.fit_transform(finaldata, finallabel)
# print(x_selected.shape)
# acclist = svclassifier(x_selected,finallabel)
# importance = clf.feature_importances_
# res_list = np.argsort(importance)[::-1][:30]
# print(importance[res_list])
# print(res_list)
# feature_label_1 = []
# for i in res_list:
#     feature_label_1.append(feat_label[i])

# SVC 特征选取
from sklearn.svm import SVC
# svc = SVC(kernel = "linear")
# svc.fit(finaldata,finallabel)
# model = SelectFromModel(svc, prefit =True)
# index = model.get_support(indices = True)
# x_selected = pd.DataFrame(finaldata[:,index])
# index1 = np.argsort(svc.coef_)[0,0:100].tolist()
# acclist = RFClassifier(np.array(x_selected),finallabel)

# Lasso 特征选取
ls = Lasso(alpha = 0.0001)
ls.fit(finaldata,finallabel)
mask = SelectFromModel(ls)
x_selected = finaldata.loc[:,mask]
acclist = svclassifier(np.array(x_selected),finallabel)

#互信息素提取
# from sklearn.feature_selection import SelectKBest,mutual_info_classif
# a=mutual_info_classif(finaldata,finallabel)
# a = a!=0
# x_selected = SelectKBest(a,k=150,prefit=True).transform()
# print(a)

# 十次特征重要性排序
# fea_dict = {}
# list1 = []
# list2 = []
# for i in range(30):
#     clf = RandomForestClassifier(n_estimators=15000).fit(finaldata,finallabel)
#     importance = clf.feature_importances_.tolist()
#     res_list = np.argsort(importance)[::-1][:30].tolist()
#     for i in res_list:
#         if i not in fea_dict:
#             fea_dict[i] = importance[i]
#         else:
#             fea_dict[i] = (fea_dict[i] + importance[i])/2
# print(sorted(fea_dict.items(),key = lambda x:x[1], reverse = True)[0:20])


# 特征相似度
# n = []
# m = []
# for i in res_list[0:10]:
#     if i in index1[0:10]:
#          n.append(i)
# print(n)
# for i in res_list[0:20]:
#     if i in index1[0:20]:
#          m.append(i)
# print(m)

# acclist = []
# acclist.append(RFClassifier(finaldata,finallabel))

accaverage = np.mean(acclist)
print(acclist)
print(accaverage)

# print(max(acc))
# print("ave:",np.mean(acc),"std:",np.std(acc))
