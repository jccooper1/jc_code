# 导入必要的库
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定义一维Poisson方程的右端项和真解
def f(x):
    return -np.exp(-x/5.0) * (1.0/25.0 * np.sin(x) - 1.0/5.0 * np.cos(x))

def u_exact(x):
    return np.exp(-x/5.0) * np.sin(x)

# 定义神经网络模型
class PINN(tf.keras.Model):
    def __init__(self, n_hidden):
        super(PINN, self).__init__()
        # 输入层
        self.input_layer = tf.keras.layers.Dense(n_hidden, activation='tanh')
        # 隐藏层
        self.hidden_layer = tf.keras.layers.Dense(n_hidden, activation='tanh')
        # 输出层
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, x):
        # 前向传播
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x

    def loss(self, x, u):
        # 计算损失函数，包括边界条件和物理约束
        # 边界条件
        u0 = self(tf.constant([[0.0]])) # u(0) = 0
        u1 = self(tf.constant([[10.0]])) # u(10) = 0
        loss_bc = tf.reduce_mean(u0**2 + u1**2)
        # 物理约束
        with tf.GradientTape() as tape:
            tape.watch(x)
            u_pred = self(x) # 神经网络预测的u值
            u_pred_x = tape.gradient(u_pred, x) # 对x求导得到u_x值
            u_pred_xx = tape.gradient(u_pred_x, x) # 对x求导得到u_xx值
            f_pred = -u_pred_xx # 神经网络预测的f值，根据Poisson方程f=-u_xx
            f_true = f(x.numpy()) # 真实的f值，根据已知函数f(x)
            loss_pde = tf.reduce_mean((f_pred - f_true)**2) # 物理约束的均方误差
        # 总损失函数，加权求和
            loss = loss_bc + loss_pde
        return loss

# 定义训练数据，包括边界点和内部点
n_bc = 10 # 边界点个数
n_domain = 100 # 内部点个数

x_bc_left = np.zeros((n_bc, 1)) # 左边界点x坐标为0
x_bc_right = np.ones((n_bc, 1)) * 10.0 # 右边界点x坐标为10
x_domain = np.random.rand(n_domain, 1) * 10.0 # 内部点x坐标为[0,10]之间的随机数

x_train = np.vstack([x_bc_left, x_bc_right, x_domain]) # 合并所有训练点的x坐标

# 将训练数据转换为TensorFlow张量
x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)

# 创建神经网络模型，设置隐藏层神经元个数为20
model = PINN(n_hidden=20)

# 定义优化器，使用Adam算法，学习率为0.01
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# 定义训练轮数，每隔一定轮数打印损失函数值
n_epochs = 2000
print_every = 100

# 训练神经网络
for epoch in range(n_epochs):
    # 前向传播，计算损失函数
    loss = model.loss(x_train_tensor, None)
    # 反向传播，更新参数
    optimizer.minimize(loss, model.trainable_variables)
    # 打印损失函数值
    if (epoch+1) % print_every == 0:
        print(f'Epoch {epoch+1}, Loss {loss.numpy():.4f}')

# 定义测试数据，包括边界点和内部点
n_test = 100 # 测试点个数

x_test = np.linspace(0, 10, n_test).reshape(-1, 1) # 测试点x坐标为[0,10]之间的均匀分布

# 将测试数据转换为TensorFlow张量
x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)

# 使用神经网络预测测试数据的u值
u_pred_tensor = model(x_test_tensor)

# 将TensorFlow张量转换为Numpy数组
u_pred = u_pred_tensor.numpy()

# 计算测试数据的真实u值，用于比较
u_true = u_exact(x_test)

# 绘制预测值和真实值的对比图
plt.plot(x_test, u_true, 'b-', label='Exact')
plt.plot(x_test, u_pred, 'r--', label='Prediction')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()
