'''
读取iris数据集中鸢尾花萼片、花瓣长度数据（csv格式）
进行排序、去重、求和、累计和、
均值、标准差、方差、最小值、最大值。
数据集下载：
https://github.com/cc200723/-Machine-learning-and-AI/tree/main/iris
'''

import numpy as np
import csv

#读取数据
iris_data =[]
with open("iris.csv") as csvfile:
    csv_reader =csv.reader(csvfile)
    for row in csv_reader:
            iris_data.append(row)
iris_list = []
for row in iris_data:
    iris_list.append(tuple(row[0:]))
#print(iris_list)#打印结果
             

#创建数据类型
datatype = np.dtype([("Sepal.Length",np.str_,40),
                     ("Sepal.Width",np.str_,40),
                     ("Petal.Length",np.str_,40),
                     ("Petal.Width",np.str_,40),
                     ("Species",np.str_,40),])
#打印结果
#print(datatype)

#创建二维数组
iris_data= np.array(iris_list,dtype=datatype)
#打印结果
#print(iris_data)

#数据类型转化
SepalLength=iris_data["Sepal.Length"].astype(float)
SepalWidth=iris_data["Sepal.Width"].astype(float)
PetalLength=iris_data["Petal.Length"].astype(float)
PetalWidth=iris_data["Petal.Width"].astype(float)
'''
#打印结果
print(SepalLength)
print(SepalWidth)
print(PetalLength)
print(PetalWidth)
'''
#排序
SepalLength_sort = np.sort(SepalLength)
SepalWidth_sort = np.sort(SepalLength)
PetalLength_sort = np.sort(SepalLength)
PetalWidth_sort = np.sort(SepalLength)
'''
#打印结果
print(SepalLength_sort)
print(SepalWidth_sort)
print(PetalLength_sort)
print(PetalWidth_sort)
'''

#数据去重
SepalLength_unique = np.unique(SepalLength)
SepalWidth_unique = np.unique(SepalLength)
PetalLength_unique = np.unique(SepalLength)
PetalWidth_unique = np.unique(SepalLength)
'''
#打印结果
print(SepalLength_unique)
print(SepalWidth_unique)
print(PetalLength_unique)
print(PetalWidth_unique)
'''

#求和
SepalLength_sum = np.sum(SepalLength)
SepalWidth_sum = np.sum(SepalLength)
PetalLength_sum = np.sum(SepalLength)
PetalWidth_sum = np.sum(SepalLength)
'''
#打印结果
print(SepalLength_sum)
print(SepalWidth_sum)
print(PetalLength_sum)
print(PetalWidth_sum)
'''

#均值
SepalLength_mean = np.mean(SepalLength)
SepalWidth_mean = np.mean(SepalLength)
PetalLength_mean = np.mean(SepalLength)
PetalWidth_mean = np.mean(SepalLength)
'''
#打印结果
print(SepalLength_mean)
print(SepalWidth_mean)
print(PetalLength_mean)
print(PetalWidth_mean)
'''

#标准差
SepalLength_std = np.std(SepalLength)
SepalWidth_std = np.std(SepalLength)
PetalLength_std = np.std(SepalLength)
PetalWidth_std = np.std(SepalLength)
'''
#打印结果
print(SepalLength_std)
print(SepalWidth_std)
print(PetalLength_std)
print(PetalWidth_std)
'''

#方差
SepalLength_var = np.var(SepalLength)
SepalWidth_var = np.var(SepalLength)
PetalLength_var = np.var(SepalLength)
PetalWidth_var = np.var(SepalLength)
'''
#打印结果
print(SepalLength_var)
print(SepalWidth_var)
print(PetalLength_var)
print(PetalWidth_var)
'''

#最小值
SepalLength_min = np.min(SepalLength)
SepalWidth_min = np.min(SepalLength)
PetalLength_min = np.min(SepalLength)
PetalWidth_min = np.min(SepalLength)
'''
#打印结果
print(SepalLength_min)
print(SepalWidth_min)
print(PetalLength_min)
print(PetalWidth_min)
'''

#最大值
SepalLength_max = np.max(SepalLength)
SepalWidth_max = np.max(SepalLength)
PetalLength_max = np.max(SepalLength)
PetalWidth_max = np.max(SepalLength)
'''
#打印结果
print(SepalLength_max)
print(SepalWidth_max)
print(PetalLength_max)
print(PetalWidth_max)
'''




