from sklearn.model_selection import KFold
def Split_Sets_10_Fold(total_fold, data):
	#total_fold是你设定的几折，我这里之后带实参带10就行，data就是我需要划分的数据
	#train_index,test_index用来存储train和test的index（索引）
    train_index = []
    test_index = []
    kf = KFold(n_splits=total_fold, shuffle=True, random_state=True)
    #这里设置shuffle设置为ture就是打乱顺序在分配
    for train_i, test_i in kf.split(data):
        train_index.append(train_i)
        test_index.append(test_i)
    return train_index, test_index
