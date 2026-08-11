import numpy as np
import matplotlib.pyplot as plt

# 定义Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 生成数据
x = np.linspace(-5, 5, 100)
y = sigmoid(x)

# 绘制图形
plt.figure(figsize=(8, 4))
plt.plot(x, y, label='Sigmoid Function', color='blue', linewidth=2)
plt.title('Sigmoid Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(y=0.5, color='red', linestyle=':', linewidth=1)
plt.axvline(x=0, color='red', linestyle=':', linewidth=1)
plt.legend()
plt.savefig('sigmoid.png', dpi=500, bbox_inches='tight')
plt.show()
