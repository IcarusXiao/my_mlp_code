import tensorflow as tf
from tensorflow import keras
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from scikeras.wrappers import KerasClassifier
#加载数据
fashion_mnist=keras.datasets.fashion_mnist
(x_train_full,y_train_full),(x_test,y_test)=fashion_mnist.load_data()
#导入库并加载数据集
#接下来划分验证集并对数据进行缩放
x_valid,x_train=x_train_full[:5000]/255.0,x_train_full[5000:]/255.0
y_valid,y_train=y_train_full[:5000],y_train_full[5000:]
#创建标签 因为原始标签为数字
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
#定义构建模型的函数
def build_model(n_hidden=2,n_neurons=20,learning_rate=1e-3,input_shape=(28,28)):
    model=keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons,activation='relu'))
    model.add(keras.layers.Dense(10,activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
                  metrics=['accuracy'] )
    return model
#包装模型1
keras_clf = KerasClassifier(
    model=build_model,
    epochs=70,  # 可以在这里设置默认训练轮数
    verbose=1
)

#定义回调点
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=3,  # 容忍3轮验证指标不提升
    restore_best_weights=True  # 训练结束后自动恢复最佳权重
)
# 定义 ModelCheckpoint 回调
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "my_keras_model.h5",  # 文件路径
    save_best_only=True,  # 只保存最佳模型
    verbose=1  # 打印日志信息
)


param_dis = {
    "model__n_hidden": [1, 2, 3],  # 注意前缀 "model__"
    "model__n_neurons": np.arange(10, 100, 10),  # 神经元数量范围
    "optimizer__learning_rate": reciprocal(3e-4, 3e-2),  # 学习率
    "batch_size": [32, 64, 128, 256]
}
#接下来开始搜索
rnd_scv=RandomizedSearchCV(keras_clf,param_distributions=param_dis,n_iter=10,cv=3)#实例化网格搜索对象
 #执行随机搜索并保存最佳模型
rnd_scv.fit(
    x_train, y_train,
    epochs=100,  # 增加epoch数量以充分训练
    validation_data=(x_valid, y_valid),
    callbacks=[early_stopping_cb, checkpoint_cb]  # 添加两个回调
)

# 获取训练好的最优模型
best_model = rnd_scv.best_estimator_
best_model.save("my_keras_model_full.h5")#保存模型
best_model.evaluate(x_test,y_test)