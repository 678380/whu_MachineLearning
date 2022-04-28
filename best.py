##最优参数下各分类器的比较

from unittest import result
import numpy as np
from time import time
from sklearn.datasets import fetch_20newsgroups#引入新闻数据包
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.feature_selection import SelectKBest, chi2#卡方检验——特征筛选

from sklearn.naive_bayes import MultinomialNB#引入多项式朴素贝叶斯
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn import metrics


TARGET = 0

###  基准模型方法
def benchmark(clf,name,x_train,y_train,x_test,y_test):
    print (u'分类器：', clf)
    
    ##  设置最优参数，并使用5折交叉验证获取最优参数值
    alpha_can = np.logspace(-2, 1, 10)#等比数列
    model = GridSearchCV(clf, param_grid={'alpha': alpha_can}, cv=5)
    m = alpha_can.size
    
    ##hasattr 如果模型有一个参数是alpha，进行设置
    if hasattr(clf, 'alpha'):
        model.set_params(param_grid={'alpha': alpha_can})
        m = alpha_can.size
    ## 如果模型有一个k近邻的参数，进行设置
    if hasattr(clf, 'n_neighbors'):
        neighbors_can = np.arange(1, 15)#等差数列
        model.set_params(param_grid={'n_neighbors': neighbors_can})
        m = neighbors_can.size
    ## LinearSVC最优参数配置
    if hasattr(clf, 'C'):
        C_can = np.logspace(1, 3, 3)
        model.set_params(param_grid={'C':C_can})
        m = C_can.size
    ## SVM最优参数设置
    if hasattr(clf, 'C') & hasattr(clf, 'gamma'):
        C_can = np.logspace(1, 3, 3)
        gamma_can = np.logspace(-3, 0, 3)
        model.set_params(param_grid={'C':C_can, 'gamma':gamma_can})
        m = C_can.size * gamma_can.size
    ##感知机最优参数设置
    if hasattr(clf, 'eta0'):
        eta_can = np.arange(0.05, 1, 0.05) #起点、终点、步长
        model.set_params(param_grid={'eta0':eta_can})
        m =eta_can.size
    
    ## 模型训练
    t_start = time()
    model.fit(x_train, y_train)
    t_end = time()
    t_train = (t_end - t_start) / (5*m)
    print (u'5折交叉验证的训练时间为：%.3f秒/(5*%d)=%.3f秒' % ((t_end - t_start), m, t_train))
    print (u'最优超参数为：', model.best_params_)
    
    ## 模型预测
    t_start = time()
    y_hat = model.predict(x_test)
    t_end = time()
    t_test = t_end - t_start
    print (u'测试时间：%.3f秒' % t_test)
    
    ## 模型效果评估
    train_acc = metrics.accuracy_score(y_train, model.predict(x_train))
    test_acc = metrics.accuracy_score(y_test, y_hat)
    print (u'训练集准确率：%.2f%%' % (100 * train_acc))
    print (u'测试集准确率：%.2f%%' % (100 * test_acc))
    
def main():
    remove = ('headers', 'footers', 'quotes')
    categories = 'alt.atheism','sci.space','rec.sport.baseball'
    data_train = fetch_20newsgroups(data_home='./datas/',subset='train', 
                                    categories=categories, shuffle=True, random_state=0, remove=remove)
    data_test = fetch_20newsgroups(data_home='./datas/',subset='test', 
                                    categories=categories, shuffle=True, random_state=0, remove=remove)

    x_train = data_train.data
    y_train = data_train.target
    x_test = data_test.data
    y_test = data_test.target
    for i in range(len(y_train)):
        if y_train[i] != TARGET:
            y_train[i] = -1
        else: 
            y_train[i] = 1
    for i in range(len(y_test)):
        if y_test[i] != TARGET:
            y_test[i] = -1
        else: y_test[i] = 1

    vectorizer = CountVectorizer(input='content', stop_words='english', 
                              max_df=0.65, min_df=0.05)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    # reslut =Perceptron().get_params().keys()
    # print(reslut)

    ### 使用不同的分类器对数据进行比较
    print (u'分类器的比较：\n')
    clfs = [
        [KNeighborsClassifier(), 'KNN'],
        [MultinomialNB(), 'MultinomialNB'],
        [SVC(), 'SVM'],
        [LinearSVC(loss='squared_hinge', penalty='l1', dual=False, tol=1e-4), 'LinearSVC-l1'],
        [LinearSVC(loss='squared_hinge', penalty='l2', dual=False, tol=1e-4), 'LinearSVC-l2'],
        [Perceptron(), 'Perceptron']
    ]
    for clf,name in clfs:
        benchmark(clf,name,x_train,y_train,x_test,y_test)
        print()

if __name__ == '__main__':
   main()

