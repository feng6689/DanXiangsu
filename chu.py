import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from PIL import Image

img_path = "3.jpg"
target_size = 128
noise_type = 'gaussian'
noise_param = 0.50
#show_plot = True

# 预处理
img = Image.open(img_path).convert('L') #转灰度
img = img.resize((target_size, target_size)) #重塑
scene = np.array(img, dtype=np.float32) / 255.0 #归一
H, W = scene.shape
#print(H,W)
pixels = H * W #总像素

# hadamard矩阵
H_mat = hadamard(pixels)

# 含噪测量（噪声加在测量信号上）
measurements = []
for i in range(pixels): # 获取第i对差分图案
    row = H_mat[i, :]
    pattern_pos = (row.reshape((H, W)) + 1) / 2 # 正
    pattern_neg = 1 - pattern_pos # 负

    # 干净的差分测量
    I_pos_clean = np.sum(scene * pattern_pos)
    I_neg_clean = np.sum(scene * pattern_neg)
    measurement_clean = I_pos_clean - I_neg_clean

    # 在测量信号上加噪
    if noise_type == 'gaussian':
        # 等效：sum(noise.flatten() * row)，其中 noise ~ N(0, noise_param)
        # 由于 row[i] 为±1，sum(noise.flatten() * row) ~ N(0, noise_param * sqrt(pixels))
        noise_signal = np.random.normal(0, noise_param * np.sqrt(pixels))
        measurement_noisy = measurement_clean + noise_signal
    elif noise_type == 'salt_pepper':
        # 生成噪声掩码（不修改原图），计算噪声对测量值的影响
        salt = np.random.rand(*scene.shape) < noise_param / 2
        pepper = np.random.rand(*scene.shape) < noise_param / 2
        
        # 计算噪声引起的测量值变化
        # noisy_scene - scene：对于盐噪声是 (1 - scene)，对于胡椒噪声是 (0 - scene)
        noise_effect = np.sum((1.0 - scene[salt]) * row.reshape((H, W))[salt])
        noise_effect += np.sum((0.0 - scene[pepper]) * row.reshape((H, W))[pepper])
        
        measurement_noisy = measurement_clean + noise_effect
    elif noise_type == 'none':
        measurement_noisy = measurement_clean
    
    measurements.append(measurement_noisy) # 加到数组末尾

measurements = np.array(measurements)

# 重建
img_vec = H_mat.T @ measurements # 转置
img_vec = img_vec / pixels # 求逆
reconstructed_img = img_vec.reshape((H, W))
reconstructed_img = np.clip(reconstructed_img, 0, 1)

# 显示单次加噪
noisy_example = scene.copy()
if noise_type == 'gaussian':
    noise_ex = np.random.normal(0, noise_param, noisy_example.shape)
    noisy_example = noisy_example + noise_ex
elif noise_type == 'salt_pepper':
    salt_ex = np.random.rand(*noisy_example.shape) < noise_param / 2
    pepper_ex = np.random.rand(*noisy_example.shape) < noise_param / 2
    noisy_example[salt_ex] = 1.0
    noisy_example[pepper_ex] = 0.0
elif noise_type == 'none':
    pass
noisy_example = np.clip(noisy_example, 0.0, 1.0)


plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(scene, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_example, cmap='gray')
plt.title(f'Noisy ({noise_type})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(reconstructed_img, cmap='gray')
plt.title('hadamard SPI)')
plt.axis('off')

plt.tight_layout()
plt.show()
