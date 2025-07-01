import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

fashion_mnist=keras.datasets.fashion_mnist
(x_train_full,y_train_full),(x_test,y_test)=fashion_mnist.load_data()
#导入库并加载数据集
#接下来划分验证集并对数据进行缩放
x_valid,x_train=x_train_full[:5000]/255.0,x_train_full[5000:]/255.0
y_valid,y_train=y_train_full[:5000],y_train_full[5000:]
#创建标签 因为原始标签为数字
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#1基础型 构建简单顺序模型
modol=keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300,activation='relu'),      
    keras.layers.Dense(100,activation='relu'),      
    keras.layers.Dense(10,activation='softmax'),      

])

"""
可以使用get_weights（​）和set_weights（​）方法访问层的所有参数
可以使用get_weights（​）和set_weights（​）方法访问层的所有参数
创建模型后，你必须调用compile（​）方法来指定损失函数和要使用的优化器
fit（​）方法可对模型进行训练
evaluate（​）方法对模型在测试集上进行评估
模型的summary（​）方法显示模型的所有层
"""
modol.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=0.1),metrics=['accuracy'])
# 这里有中括号是因为metrics应当传一个列表
history=modol.fit(x_train,y_train,epochs=30,validation_data=(x_valid,y_valid))

pd.DataFrame(history.history).plot(figsize=(16, 10))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
