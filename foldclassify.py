#基于特征选择后的特征，构建分类器，采用十则交叉验证，实现分类
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from split_sets_10_fold import Split_Sets_10_Fold
from sklearn.model_selection import cross_val_score
import numpy as np
def AdaBoost( selectdata, label, total_fold=10,):
    [train_index, test_index] = Split_Sets_10_Fold(total_fold, selectdata)
    acclist = []
    tree = DecisionTreeClassifier(criterion="gini",
                                  random_state = 0,
                                  splitter= "best",
                                  min_samples_leaf = 6,
                                  min_samples_split = 6)
    for i in range(total_fold):
        train_data = selectdata[train_index[i], :, ]     #得到训练数据
        train_label = label[train_index[i]]           	#得到训练数据标签
        test_data = selectdata[test_index[i], :, ]
        test_label = label[test_index[i]]
        clf = AdaBoostClassifier(base_estimator = tree,
                                 random_state=0,
                                 n_estimators=1000,
                                 algorithm = "SAMME",
                                 learning_rate= 0.05)
        clf.fit(train_data, train_label)
        predict_label = clf.predict(test_data)
        acc = accuracy_score(test_label, predict_label)
        acclist.append(acc)
    return acclist

def RFClassifier( selectdata,label,):
    [train_index, test_index] = Split_Sets_10_Fold(10, selectdata)
    acclist = []
    for i in range(0, 10):
        train_data = selectdata[train_index[i], :, ]  # 得到训练数据
        train_label = label[train_index[i]]  # 得到训练数据标签
        test_data = selectdata[test_index[i], :, ]
        test_label = label[test_index[i]]
        dtc = RandomForestClassifier(n_estimators = 2000,
                                     random_state = 0,
                                     min_samples_leaf = 4,
                                     max_depth = 40,
                                      )
        dtc.fit(train_data, train_label)
        predict_label = dtc.predict(test_data)
        acclist.append(accuracy_score(test_label, predict_label))
    return acclist

def svclassifier( selectdata,label,):
    [train_index, test_index] = Split_Sets_10_Fold(10, selectdata)
    acclist = []
    for i in range(0, 10):
        train_data = selectdata[train_index[i], :, ]  # 得到训练数据
        train_label = label[train_index[i]]  # 得到训练数据标签
        test_data = selectdata[test_index[i], :, ]
        test_label = label[test_index[i]]
        dtc = SVC(kernel = 'poly',C=1.1,gamma=1.5 )
        dtc.fit(train_data, train_label)
        predict_label = dtc.predict(test_data)
        acclist.append(accuracy_score(test_label, predict_label))
    return acclist
