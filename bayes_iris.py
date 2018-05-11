#coding:utf-8
import math

Iris_setosa_data=[]
Iris_versicolor_data=[]
Iris_virginica_data=[]
#读取训练数据集，这里我将每种花取前45条数据，剩下的5条数据另外存入测试数据集
def read_train_data(filename):
    f=open(filename,'r')
    all_lines=f.readlines()
    for line in all_lines[0:45]:
        line=line.strip().split(',')
        Iris_setosa_data.append(line[0:4])
        #Iris_setosa_label+=1
    for line in all_lines[51:95]:
        line=line.strip().split(',')
        Iris_versicolor_data.append(line[0:4])
        #Iris_versicolor_label+=1
    for line in all_lines[101:145]:
        line=line.strip().split(',')
        Iris_virginica_data.append(line[0:4])
        #Iris_virginica_label+=1
    return Iris_setosa_data,Iris_versicolor_data,Iris_virginica_data

test_data=[]
#读取测试数据集
def read_test_data(testname):
    f=open(testname,'r')
    all_lines=f.readlines()
    for line in all_lines[0:]:
        line=line.strip().split(',')   #以逗号为分割符拆分列表
        test_data.append(line)
    return test_data

#计算均值和方差
def calculate_junzhi_and_fangcha(train_data):
    x1_sum=0.0
    x2_sum=0.0
    x3_sum=0.0
    x4_sum=0.0

    for x in train_data: #计算各个特征的和
        x1_sum+=float(x[0])
        x2_sum+=float(x[1])
        x3_sum+=float(x[2])
        x4_sum+=float(x[3])
        #print(x[0],x[1],x[2],x[3])
    #计算样本在各个属性上取值的均值
    u_x1=x1_sum/45
    u_x2=x2_sum/45
    u_x3=x3_sum/45
    u_x4=x4_sum/45
   
    k1=0.0
    k2=0.0
    k3=0.0
    k4=0.0
    #计算各类样本在第i个属性上的方差
    for x in train_data:
        k1+=(float(x[0])-u_x1)**2
        k2+=(float(x[1])-u_x2)**2
        k3+=(float(x[2])-u_x3)**2
        k4+=(float(x[3])-u_x4)**2
    variance_x1=k1/45
    variance_x2=k2/45
    variance_x3=k3/45
    variance_x4=k4/45

    return u_x1,u_x2,u_x3,u_x4,variance_x1,variance_x2,variance_x3,variance_x4

#计算每个属性估计条件概率    
def calculate_P_xi_c(u_x1,u_x2,u_x3,u_x4,variance_x1,variance_x2,variance_x3,variance_x4,line_data):
    p_x1_c=(1/math.sqrt(2*math.pi))*math.exp(-(float(line_data[0])-u_x1)**2/(2*variance_x1))
    p_x2_c=(1/math.sqrt(2*math.pi))*math.exp(-(float(line_data[1])-u_x2)**2/(2*variance_x2))
    p_x3_c=(1/math.sqrt(2*math.pi))*math.exp(-(float(line_data[2])-u_x3)**2/(2*variance_x3))
    p_x4_c=(1/math.sqrt(2*math.pi))*math.exp(-(float(line_data[3])-u_x4)**2/(2*variance_x4))

    return p_x1_c,p_x2_c,p_x3_c,p_x4_c


    
if __name__ == '__main__':
    filename='iris_data.txt'
    testname='iris_test_data.txt'
    Iris_setosa_data,Iris_versicolor_data,Iris_virginica_data=read_train_data(filename)
    
    #Iris_setosa种类的各个特征属性上的均值和方差
    Iris_setosa_u_x1,Iris_setosa_u_x2,Iris_setosa_u_x3,Iris_setosa_u_x4,\
    Iris_setosa_variance_x1,Iris_setosa_variance_x2,Iris_setosa_variance_x3,\
    Iris_setosa_variance_x4=calculate_junzhi_and_fangcha(Iris_setosa_data)
    #Iris_versicolor种类的各个特征属性上的均值和方差
    Iris_versicolor_u_x1,Iris_versicolor_u_x2,Iris_versicolor_u_x3,Iris_versicolor_u_x4,\
    Iris_versicolor_variance_x1,Iris_versicolor_variance_x2,Iris_versicolor_variance_x3,\
    Iris_versicolor_variance_x4=calculate_junzhi_and_fangcha(Iris_versicolor_data)
    #Iris_virginica种类的各个特征属性上的均值和方差
    Iris_virginica_u_x1,Iris_virginica_u_x2,Iris_virginica_u_x3,Iris_virginica_u_x4,\
    Iris_virginica_variance_x1,Iris_virginica_variance_x2,Iris_virginica_variance_x3,\
    Iris_virginica_variance_x4=calculate_junzhi_and_fangcha(Iris_virginica_data)

    test_data=read_test_data(testname)
    #print ('test_data',test_data)
    #估计类先验概率
    p1=len(Iris_setosa_data)/(len(Iris_versicolor_data)+len(Iris_virginica_data)+len(Iris_setosa_data))
    p2=len(Iris_versicolor_data)/(len(Iris_versicolor_data)+len(Iris_virginica_data)+len(Iris_setosa_data))
    p3=len(Iris_virginica_data)/(len(Iris_versicolor_data)+len(Iris_virginica_data)+len(Iris_setosa_data))
    for x in test_data:
        #在Iris_setosa种类上的各个特征属性的条件概率
        P_x1_Iris_setosa,P_x2_Iris_setosa,P_x3_Iris_setosa,P_x4_Iris_setosa=calculate_P_xi_c(Iris_setosa_u_x1,Iris_setosa_u_x2,Iris_setosa_u_x3,Iris_setosa_u_x4,\
        Iris_setosa_variance_x1,Iris_setosa_variance_x2,Iris_setosa_variance_x3,Iris_setosa_variance_x4,x)
        #print(P_x1_Iris_setosa,P_x2_Iris_setosa,P_x3_Iris_setosa,P_x4_Iris_setosa)
        
        #在Iris_versicolor种类上的各个特征属性的条件概率
        P_x1_Iris_versicolor,P_x2_Iris_versicolor,P_x3_Iris_versicolor,P_x4_Iris_versicolor=calculate_P_xi_c(Iris_versicolor_u_x1,Iris_versicolor_u_x2,Iris_versicolor_u_x3,Iris_versicolor_u_x4,\
        Iris_versicolor_variance_x1,Iris_versicolor_variance_x2,Iris_versicolor_variance_x3,Iris_versicolor_variance_x4,x)
        #print(P_x1_Iris_versicolor,P_x2_Iris_versicolor,P_x3_Iris_versicolor)

        #在Iris_virginica种类上的各个特征属性的条件概率
        P_x1_Iris_virginica,P_x2_Iris_virginica,P_x3_Iris_virginica,P_x4_Iris_virginica=calculate_P_xi_c(Iris_virginica_u_x1,Iris_virginica_u_x2,Iris_virginica_u_x3,Iris_virginica_u_x4,\
        Iris_virginica_variance_x1,Iris_virginica_variance_x2,Iris_virginica_variance_x3,Iris_virginica_variance_x4,x)
        #print(P_x1_Iris_virginica,P_x2_Iris_virginica,P_x3_Iris_virginica,P_x4_Iris_virginica)

        #计算各个种类上的后验概率
        P_Iris_setosa=p1*P_x1_Iris_setosa*P_x2_Iris_setosa*P_x3_Iris_setosa*P_x4_Iris_setosa
        #print( P_Iris_setosa)
        P_Iris_versicolor=p2*P_x1_Iris_versicolor*P_x2_Iris_versicolor*P_x3_Iris_versicolor*P_x4_Iris_versicolor
        #print( P_Iris_versicolor)
        P_Iris_virginica=p3*P_x1_Iris_virginica*P_x2_Iris_virginica*P_x3_Iris_virginica*P_x4_Iris_virginica
        #print( P_Iris_virginica)

        if P_Iris_setosa>P_Iris_versicolor and P_Iris_setosa>P_Iris_virginica:
            print(x[0],x[1],x[2],x[3],":这行数据属于Iris_setosa类")
        if P_Iris_versicolor>P_Iris_setosa and P_Iris_versicolor>P_Iris_virginica:
            print(x[0],x[1],x[2],x[3],":这行数据属于Iris_versicolor类")      
        if P_Iris_virginica>P_Iris_setosa and P_Iris_virginica>P_Iris_versicolor:
            print(x[0],x[1],x[2],x[3],":这行数据属于Iris_virginica类")
        
    
    


    