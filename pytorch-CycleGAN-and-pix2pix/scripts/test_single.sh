set -ex
python test.py --dataroot datasets/bdd100k/A/val --name pix2pix --model test --netG resnet_9blocks --preprocess none --batch_size 2 --epoch 120 --dataset_mode single --norm batch 

# Then run a script to rename and organize images
python -c "
import os, shutil
result_dir = './results/pix2pix/test_120'
output_dir = './results/pix2pix/only_fake'
os.makedirs(output_dir, exist_ok=True)
for file in os.listdir(result_dir):
    if '_fake.png' in file:
        new_name = file.replace('_fake.png', '.png')
        shutil.copy(os.path.join(result_dir, file), os.path.join(output_dir, new_name))
print(f'Copied all fake images to {output_dir}')
"