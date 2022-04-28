##感知机、k近邻、朴素贝叶斯文本分类


from ast import Lambda
import numpy as np
from sklearn.datasets import fetch_20newsgroups#引入新闻数据包
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import heapq

TARGET = 0
Lambda = 1

Eta = 0.6 #全局变量，感知机学习率

def perceptron(x_train,y_train, x_test, y_test):
    '''初始化'''
    w = np.zeros(len(x_train[0]))  # 初始化w,b
    b = 0
    def sign_y(x,w,b):             # 计算w*x+b的值
        y1 = np.dot(x,w) + b
        return y1
    '''训练'''
    turn = 2000 #迭代次数
    for i in range(turn):   # 下面定义了循环结束条件，只要还有误分类点，就会一直循环下去
        wrong_count = 0    
        for i in range(len(x_train)): # 样本数
            x = x_train[i]
            yi = y_train[i]
            if yi * sign_y(x,w,b)<=0: # 说明误分类了，更新w，b
                w = w + Eta * yi * x
                b = b + Eta * yi
                wrong_count += 1
        if wrong_count == 0:    # 定义函数结束条件，一次循环下来，误分类点为0，即所有点都正确分类了
            break
    # print(w,b)
    '''测试'''
    result = []
    for test in x_test:
        if  sign_y(test,w,b)<=0: # 说明误分类了，更新w，b
               result.append(-1)
        else: result.append(1)
    '''准确率'''
    ans = 0
    for i in range(len(result)):
        if result[i] == y_test[i]:
            ans = ans + 1
    ans = ans / len(result)
    print('感知机准确率：')
    print(ans)

class Node:
    def __init__(self,data,sp=0,left=None,right=None):
        self.data = data
        self.sp = sp  #kd树层数
        self.left = left #左子树的根结点
        self.right = right #右子树的根结点
        self.nearest_dist = -np.inf  
        #我们需要使用最小堆来模拟最大堆，我们设置默认距离-∞，实际就是+∞
        
    def __lt__(self, other):
        return self.nearest_dist < other.nearest_dist
    
class KDTree:
    def __init__(self,data):
        self.k = data.shape[1]   #有多少属性
        self.root = self.createTree(data,0)  #深度为0，看第0维特征
        self.heap = [] #初始化一个堆

    def createTree(self,dataset,sp):
        if len(dataset) == 0:
            return None

        dataset_sorted = dataset[np.argsort(dataset[:,sp])] #按当前特征列进行排序
        #取中位数
        mid = len(dataset) // 2
        #生成节点
        left = self.createTree(dataset_sorted[:mid], (sp+1) % self.k)   #深度为sp+1，看第(sp+1)%k维特征
        right = self.createTree(dataset_sorted[mid+1:],(sp+1) % self.k)
        parentNode = Node(dataset_sorted[mid],sp,left,right)
       
        return parentNode
    
    def nearest(self, x, k, y_train, x_train):
        def visit(node):
            if node != None:
                #从根节点一直往下访问，直到访问到子节点
                if(node.data[node.sp] >= x[node.sp]):
                    visit(node.left)
                else: visit(node.right)
                
                #查看当前节点到目标节点的距离 二范数求距离
                curr_dis = np.linalg.norm(x-node.data,2)
                node.nearest_dist = -curr_dis
                #更新节点
                if len(self.heap) < k: #直接加入
                    heapq.heappush(self.heap,node)
                else:
                    #先获取最大堆最大值，比较后决定
                    if heapq.nsmallest(1,self.heap)[0].nearest_dist < -curr_dis:
                        heapq.heapreplace(self.heap, node)   
                        
                #判断是否要访问另一个子节点
                if len(self.heap) < k or abs(heapq.nsmallest(1,self.heap)[0].nearest_dist) > abs(node.data[node.sp] - x[node.sp]): 
                    if(node.data[node.sp] < x[node.sp]):
                        visit(node.left)
                    else: visit(node.right)
        
        #从根节点开始查找
        node = self.root
        visit(node)
        '''将结果存入result'''
        nds = heapq.nlargest(k,self.heap) #结点
        yes = 0
        no = 0
        row=[] #x_train行号，对应文档编号
        x_train = np.matrix.tolist(x_train)#将矩阵转化为列表
        data=[]
        for i in range(k):
            nd = nds[i]
            data = np.matrix.tolist(nd.data)
            r = x_train.index(data)
            row.append(r)
        for i in row:
            if y_train[i] == -1:
                yes = yes+1
            else: no = no + 1
        # print('邻近的文档号：')
        # print(row)
        if yes >= no: return 1
        else: return -1
            

def bayes(x_train_Y, x_train_N, x_test, y_train, y_test, vocab):
    #len(vocab) = x_train_N行数 = x_train_Y行数
    '''计算P(c)'''
    #categories[y_train[i]]
    P_yes = len(x_train_Y)/ len(y_train)
    P_no = len(x_train_N) / len(y_train)
    '''计算先验概率'''
    f = open('train_N.txt', "r")
   
    while  True:
        file = open ('P_train_N.txt','a',encoding ='utf-8')
        # Get next line from file
        line  =  f.readline()
        # If line is empty then end of file reached
        if  not  line  :
            break
        line = line.split()
        count = int(line[1])
        result = (count + Lambda)/ (len(x_train_N) + Lambda*len(vocab)) 
        
        file.write('%lf\n' %( result ) )
        file.close()
    f.close()

    f = open('train_Y.txt', "r")
    while  True:
        file = open ('P_train_Y.txt','a',encoding ='utf-8')
        # Get next line from file
        line  =  f.readline()
        # If line is empty then end of file reached
        if  not  line  :
            break
        line = line.split()
        count = int(line[1])
        result = (count + Lambda)/ (len(x_train_Y) + Lambda*len(vocab))
        file.write('%lf\n' %( result ) )
        file.close()
    f.close()


    '''计算后验'''
    result = {} #计算出来的y_test集合
    test = [] #用于存储x_test中的一行（一个文档）
    i_sample = 0 #用于确认result脚标，将文档与结果一一对应
    for test in x_test:
        pn = P_no
        py = P_yes
        fn = open('P_train_N.txt', "r")
        fy = open('P_train_Y.txt', "r") 
        '''假设为负例的后验概率'''
        i_word = 0 #用于遍历词项
        lines = fn.readlines() #读取先验概率文件的所有行
        for line in lines:
            #若词项读取完毕，结束该循环
            #（为解决文档最后一行为\n导致的test[i_word]溢出）
            if i_word == len(vocab):
                break
            #调整数据类型
            line = float(line) 
            #若该文档没有该词项，直接跳到下一词项
            if int(test[i_word]) == 0:
                i_word = i_word + 1
                continue
            #因为**总会出现数据类型导致的错误，所以用循环实现乘方
            k = 0
            for k in range(int(test[i_word])):
                pn = float(pn *  line) 
                k + 1
            #开始下一词项
            i_word = i_word + 1
        '''假设为正例的后验概率'''
        i_word = 0
        lines = fy.readlines()
        for line in lines:
            if i_word == len(vocab):
                break
            line = float(line) 
            if int(test[i_word]) == 0:
                i_word = i_word + 1
                continue
            k=0
            for k in range(int(test[i_word])):
                py = float(py *  line) 
            i_word = i_word + 1

        # print(py, pn)
        '''确定结果'''
        if py >= pn :
            result[i_sample]=1
        else: result[i_sample]=-1
        i_sample = i_sample + 1
        fn.close()
        fy.close()
    '''计算准确率'''
    ans = 0
    for i in result:
        if result[i] == y_test[i]:
            ans = ans + 1
    ans = ans / len(result)
    # print('p:')
    # print(P_yes)
    # print(P_no)
    print('朴素贝叶斯准确率：')
    print(ans)
  
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
    '''
    for i in range(6):
        if y_train[i] == 1:
            print (u'文本%d(属于: %s):' % (i+1, categories[y_train[i]-1]) )
        elif y_train[1] == -1:
            print (u'文本%d(不属于: %s):' % (i+1,categories[y_train[i]-1]) )
        print (x_train[i])
    '''
    ### 文档转换为向量
    vectorizer = CountVectorizer(input='content', stop_words='english', 
                              max_df=0.65, min_df=0.05)
    ##给朴素贝叶斯分正负例
    x_train_Y = vectorizer.fit_transform(x_train_Y) 
    x_train_N = vectorizer.fit_transform(x_train_N) 

    ##k近邻和感知机直接用train
    x_train = vectorizer.fit_transform(x_train)
    x_train = x_train.toarray()

    #测试集
    x_test = vectorizer.transform(x_test)
    x_test_array = x_test.toarray()
    
    '''Tfid特征
    transformer = TfidfTransformer()
    x_train_Y = transformer.fit_transform(x_train_Y)
    x_train_N = transformer.fit_transform(x_train_N)
    
    x_test = transformer .transform(x_test)
    '''
    ##获取属性
    vocab = vectorizer.get_feature_names_out()

    '''朴素贝叶斯数据处理'''
    ##获取正例词频
    x_train_Y_array = x_train_Y.toarray()
    dist = np.sum(x_train_Y_array, axis=0)#列相加 
    ##写入txt文件
    fY = open ('train_Y.txt','w',encoding ='utf-8')
    for tag, count in zip(vocab,dist):
        fY.write('%s %d\n' %(tag, count))
    fY.close()
    ##获取负例词频
    x_train_N_array = x_train_N.toarray()
    dist_N = np.sum(x_train_N_array, axis=0)#列相加
    ##写入txt文件
    fN = open ('train_N.txt','w',encoding ='utf-8')
    for tag, count in zip(vocab,dist_N):
        fN.write('%s %d\n' %(tag, count))
    fN.close()
    '''处理完毕'''

    bayes(x_train_Y_array, x_train_N_array, x_test_array, y_train, y_test, vocab)
    perceptron(x_train, y_train, x_test_array, y_test)
    '''k近邻'''
    result=[]
    kdtree = KDTree(x_train)  #创建KDTree
    for test in x_test_array:
        res = kdtree.nearest(test,100, y_train, x_train) #100个距离test最近的
        result.append(res)

    '''计算k近邻准确率'''
    ans = 0
    for i in range(len(result)):
        if result[i] == y_test[i]:
            ans = ans + 1
    ans = ans / len(result)
    print('k近邻准确率：')
    print(ans)

    f=open('P_train_N.txt', "r+")
    f.truncate()
    f.close()
    f=open('P_train_Y.txt', "r+")
    f.truncate()
    f.close()

if __name__ == '__main__':
   main()
