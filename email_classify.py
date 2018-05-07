# -*- encoding: utf-8 -*-
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.externals import joblib

##加载数据集
df = pd.read_csv(r'SMSSpamCollection', delimiter='\t', header=None)

##分割数据集
X_train_raw, X_test_raw, y_train, y_test =train_test_split(df[1],df[0])

##构建词频特征向量
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train_raw)#数据类型为scipy的稀疏矩阵
X_test = vectorizer.transform(X_test_raw)

X_train = X_train.toarray()#转为numpy数组
X_test = X_test.toarray()

clf = GaussianNB()
clf.fit(X_train,y_train)

##预测结果
predictions = clf.predict(X_test)

##准确率
scores = cross_val_score(clf, X_train, y_train, cv=5)
print(u'准确率：',np.mean(scores), scores)

##保存模型
joblib.dump(clf,'span_bayes.m',compress = 1)
