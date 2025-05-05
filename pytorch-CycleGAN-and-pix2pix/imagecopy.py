import os, shutil

# 正确的图像目录路径
result_dir = './results/pix2pix/test_120/images'  # 注意这里添加了 /images
output_dir = './results/pix2pix/only_fake'
os.makedirs(output_dir, exist_ok=True)

# 检查目录是否存在
if not os.path.exists(result_dir):
    print(f"错误: 目录 {result_dir} 不存在!")
    # 尝试查找正确的目录
    base_dir = './results/pix2pix/test_120'
    if os.path.exists(base_dir):
        print(f"找到基础目录: {base_dir}")
        print("列出其中的内容:")
        for item in os.listdir(base_dir):
            print(f" - {item}")
    else:
        print(f"基础目录 {base_dir} 也不存在!")
else:
    # 目录存在，继续处理
    fake_files = [f for f in os.listdir(result_dir) if '_fake.png' in f]
    print(f"找到 {len(fake_files)} 个包含 '_fake.png' 的文件")
    
    for file in fake_files:
        new_name = file.replace('_fake.png', '.png')
        shutil.copy(os.path.join(result_dir, file), os.path.join(output_dir, new_name))
    
    print(f'已复制所有 fake 图像到 {output_dir}')
    print(f'共复制了 {len(fake_files)} 个文件')