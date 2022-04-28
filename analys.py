##超参数敏感性分析
from calendar import c
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
'''感知机'''
from sklearn.linear_model import Perceptron

def perceptron_penalty(x_train, y_train, x_test, y_test):
    score_val = []
    para_list = ['l2','l1','elasticnet','None']
    for penalty in para_list:
        lr = Perceptron(random_state=1, penalty=penalty)
        lr.fit(x_train, y_train)
        score_val.append(metrics.accuracy_score(y_test,lr.predict(x_test)))
    plt.plot(para_list, score_val)
    plt.title('Perceptron - sensitivity of penalty')
    plt.ylabel('accuracy of test') #纵坐标
    plt.show()

def perceptron_eta0(x_train, y_train, x_test, y_test):
    score_val = []
    etas=[0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,0.9, 0.95, 1.0]
    for eta in etas:
        lr = Perceptron(random_state=1, eta0=eta)
        lr.fit(x_train, y_train)
        score_val.append(metrics.accuracy_score(y_test,lr.predict(x_test)))
    plt.plot(etas, score_val)
    plt.title('Perceptron - sensitivity of eta0')
    plt.ylabel('accuracy of test') #纵坐标
    plt.show()

def perceptron_max_iter(x_train, y_train, x_test, y_test):
    score_val = []
    max = [500, 1000, 1250, 1500, 2000, 2500, 3000]
    for i in max:
        lr = Perceptron(random_state=1, max_iter=i)
        lr.fit(x_train, y_train)
        score_val.append(metrics.accuracy_score(y_test,lr.predict(x_test)))
    plt.plot(max, score_val)
    plt.title('Perceptron - sensitivity of max_iter')
    plt.ylabel('accuracy of test') #纵坐标
    plt.show()    

'''朴素贝叶斯'''
from sklearn.naive_bayes import MultinomialNB #引入多项式朴素贝叶斯

def naive_bayes(x_train, y_train, x_test, y_test):
    score_val = []
    alphas = [0.4, 0.5, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775,
              0.8, 0.825, 0.85, 0.875,0.9, 0.925, 0.95, 0.975, 1]
    for alpha in alphas:
        a = MultinomialNB(alpha=alpha)
        a.fit(x_train, y_train)
        score_val.append(metrics.accuracy_score(y_test,a.predict(x_test)))
    plt.plot(alphas, score_val)
    plt.title('MuktinomiaINB - sensitivity of alpha')
    plt.ylabel('val accuracy') #纵坐标
    plt.show()    

'''k近邻'''
from sklearn.neighbors import KNeighborsClassifier
def Kneighbors(x_train, y_train, x_test, y_test):
    score_val = []
    ks = [5, 20, 50, 100, 200, 300, 400, 500]
    for k in ks:
        a = KNeighborsClassifier(n_neighbors=k)
        a.fit(x_train, y_train)
        score_val.append(metrics.accuracy_score(y_test,a.predict(x_test)))
    plt.plot(ks, score_val)
    plt.title('KNeighborsClassifier - sensitivity of k')
    plt.ylabel('val accuracy') #纵坐标
    plt.show()    

'''SVM'''
from sklearn.svm import SVC
def SVC_C(x_train, y_train, x_test, y_test):
    score_val = []
    C_can = np.logspace(1, 3, 3)
    for c in C_can:
        a = SVC(C=c)
        a.fit(x_train, y_train)
        score_val.append(metrics.accuracy_score(y_test,a.predict(x_test)))
    plt.plot(C_can, score_val)
    plt.title('SVM - sensitivity of C')
    plt.ylabel('val accuracy') #纵坐标
    plt.show()   

     
def SVC_gamma(x_train, y_train, x_test, y_test):
    score_val = []
    gamma_can = np.logspace(-3, 0, 3)
    for gamma in gamma_can:
        a = SVC(gamma=gamma)
        a.fit(x_train, y_train)
        score_val.append(metrics.accuracy_score(y_test,a.predict(x_test)))
    plt.plot(gamma_can, score_val)
    plt.title('SVM - sensitivity of gamma')
    plt.ylabel('val accuracy') #纵坐标
    plt.show()   
'''main: 获取数据并调用'''
from sklearn.datasets import fetch_20newsgroups#引入新闻数据包
from sklearn.feature_extraction.text import CountVectorizer
TARGET = 0

def main():

    ##要求删除一些信息，且只要某些类数据
    remove = ('headers', 'footers', 'quotes')
    categories = 'alt.atheism','sci.space'
    data_train = fetch_20newsgroups(data_home='./datas/',subset='train', 
                                    categories=categories, shuffle=True, random_state=0, remove=remove)
    data_test = fetch_20newsgroups(data_home='./datas/',subset='test', 
                                    categories=categories, shuffle=True, random_state=0, remove=remove)

    ### 数据重命名
    x_train = data_train.data
    y_train = data_train.target
    x_test = data_test.data
    y_test = data_test.target

    x_train_Y = []  #正例
    x_train_N = []  #负例
    for i in range(len(y_train)):
        if y_train[i] != TARGET:
            y_train[i] = -1
            x_train_N.append(x_train[i])
        else: 
            y_train[i] = 1
            x_train_Y.append(x_train[i])
    for i in range(len(y_test)):
        if y_test[i] != TARGET:
            y_test[i] = -1
        else: y_test[i] = 1

    ### 文档转换为向量
    vectorizer = CountVectorizer(input='content', stop_words='english', 
                              max_df=0.65, min_df=0.05)

    ##训练集
    x_train = vectorizer.fit_transform(x_train)
    x_train = x_train.toarray()

    ##测试集
    x_test = vectorizer.transform(x_test)
    x_test = x_test.toarray()

    perceptron_penalty(x_train, y_train, x_test, y_test)
    perceptron_eta0(x_train, y_train, x_test, y_test)
    perceptron_max_iter(x_train, y_train, x_test, y_test)

    naive_bayes(x_train, y_train, x_test, y_test)
    Kneighbors(x_train, y_train, x_test, y_test)

    SVC_C(x_train, y_train, x_test, y_test)
    SVC_gamma(x_train, y_train, x_test, y_test)
    
if __name__ == '__main__':
   main()
