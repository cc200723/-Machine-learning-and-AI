import numpy as np
import csv

#读取数据
iris_data =[]
with open("iris.csv") as csvfile:
    csv_reader =csv.reader(csvfile)
    #birth_header =next(csv_reader)
    for row in csv_reader:
            iris_data.append(row)
iris_list = []
for row in iris_data:
    iris_list.append(tuple(row[0:]))
print(iris_list)
             

#创建数据类型
datatype = np.dtype([("Sepal.Length",np.str_,40),
                     ("Sepal.Width",np.str_,40),
                     ("Petal.Length",np.str_,40),
                     ("Petal.Width",np.str_,40),
                     ("Species",np.str_,40),])
print(datatype)

#创建二维数组
iris_data= np.array(iris_list,dtype=datatype)
print(iris_data)

