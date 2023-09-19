import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# 创建 PINN 模型
class PINN(keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = keras.layers.Dense(32, activation='tanh')
        self.dense2 = keras.layers.Dense(32, activation='tanh')
        self.dense3 = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        output = self.dense3(x)
        return output

# 定义微分方程
def differential_equation(x, y):
    with tf.GradientTape() as g:
        g.watch(x)
        with tf.GradientTape as gg:
            gg.watch(x)
            dy_dx=gg.gradient(y,x)
            dy2_dx2=g.gradient(dy_dx,x)
            equation=dy2_dx2+y-x
    return equation

# 创建训练数据
x_train = np.linspace(0, 1, 100)[:, np.newaxis]  # 输入数据 x
y_train = np.sin(2 * np.pi * x_train)  # 目标输出数据 y

# 创建 PINN 模型实例
model = PINN()

# 定义损失函数和优化器
def custom_loss(y_true, y_pred):
    equation_pred = differential_equation(x_train, y_pred)  # 预测的微分方程
    loss = tf.reduce_mean(tf.square(y_true - y_pred) + tf.square(equation_pred))
    return loss

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 训练模型
epochs = 10000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = model(x_train)
        loss = custom_loss(y_train, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.numpy()}")

# 使用训练好的模型进行预测
x_test = np.linspace(0, 1, 1000)[:, np.newaxis]  # 测试数据
y_pred = model(x_test)

# 打印预测结果
print(y_pred)
plt.figure()
plt.plot(x_test, y_pred, label='PINN')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
