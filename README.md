### Abstract

Autonomous vehicles (AVs) leverage machine-learning perception models to detect and classify critical objects such as road signs, vehicles, lane lines, hazards, and pedestrians, enabling self-driving functionalities. With the nationwide proliferation of AVs, the demand for safe, secure, accurate, and rapid driving perception models has surged dramatically. Panoptic perception models have been proposed to offer advanced object detection and segmentation capabilities for AVs. This work explores the robustness of panoptic perception models against adversarial attacks, focusing on the YOLO-P (You Only Look Once for Panoptic Driving Perception) model. To comprehensively evaluate the safety of panoptic perception models, the model is subjected to various adversarial attacks, including the Fast Gradient Sign Method (FGSM), Jacobian-based Saliency Map Attack (JSMA), Color Channel Perturbations (CCP), and Universal Adversarial Perturbations (UAP), to assess their effects on model performance. Subsequent defenses, including image pre-processing techniques and the deployment of a Defense Generative Adversarial Network (GAN), are implemented to mitigate attack effects.
### Repository Contents

- **YOLOP Source Code**
  - Available at: [YOLOP GitHub Repository](https://github.com/hustvl/YOLOP)

- **Customized Attacks**
  - FGSM (Fast Gradient Sign Method)
  - JSMA (Jacobian-based Saliency Map Attack)
  - UAP (Universal Adversarial Perturbations)
  - CCP (Color Channel Perturbations)

- **Customized Defenses**
  - Pre-Processing Techniques
  - Defense GAN

  See `requirements.txt` for additional dependencies and version requirements.
  
  ```setup
  pip install -r requirements.txt
  ```
  
  Start evaluating:
  
  ```shell
  python tools/test.py --weights weights/End-to-end.pth --attack [FGSM|JSMA|UAP|CCP|None]
  ```   

#### Attack Choices
#### FGSM Attack

- **--epsilon:** Epsilon value for FGSM or CCP attack (default: `0.1`).
- **--fgsm_attack_type:** Type of FGSM attack (options: `FGSM`, `FGSM_WITH_NOISE`, `ITERATIVE_FGSM`).

#### JSMA Attack

- **--num_pixels:** Number of pixels to be perturbed after saliency calculation (default: `10`).
- **--jsma_perturbation:** Perturbation value for JSMA attack (default: `0.1`).
- **--jsma_attack_type:** Type of perturbation for JSMA attack (options: `add`, `set`, `noise`).

#### UAP Attack

- **--uap_max_iterations:** Maximum number of iterations for UAP attack (default: `10`).
- **--uap_eps:** Epsilon value for UAP attack (default: `0.03`).
- **--uap_delta:** Delta value for UAP attack (default: `0.8`).
- **--uap_num_classes:** Number of classes for UAP attack.
- **--uap_targeted:** Whether the UAP attack is targeted or not (default: `False`).
- **--uap_batch_size:** Batch size for UAP attack (default: `6`).

#### CCP Attack

- **--color_channel:** Color channel to perturb (`R`, `G`, `B`).

  
#### Defense Application

Apply a certain defense to a specific attack.

```shell
python tools/test.py --weights <path to pretrained model> --attack <attack type> --<defense type> <options>

Eg:
python tools/test.py --weights weights/End-to-end.pth --attack FGSM --resizer 64x64
```   
#### Pre-processing Choices

- **--resizer:** Desired `WIDTHxHEIGHT` of your resized image.
- **--quality:** Desired quality for JPEG compression output (`0-100`).
- **--border_type:** Border type for Gaussian Blurring (`default`, `constant`, `reflect`, `replicate`).
- **--gauss:** Apply Gaussian Blurring to image. Specify `ksize` as `WIDTHxHEIGHT`.
- **--noise:** Add Gaussian Noise to image. Specify `sigma` value for noise generation.
- **--bit_depth:** Choose bit value between `1-8`.

#### Utilizing the Defense GAN

The same image size is used as YOLOP, but in order not to lose any image information, the attack image and the original image are saved in .npy format for pix2pix training.

For the usage of pix2pix, such as training and evaluation, please refer to the source code [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

```shell
python pytorch-CycleGAN-and-pix2pix/train.py --dataroot datasets/bdd100k --name train --model pix2pix --netG resnet_9blocks --preprocess none --batch_size 1 --dataset_mode aligned
```
Use pix2pix to defend against a specific attack method.

```shell
python tools/test.py --weights <path to pretrained model> --attack <attack type> --use_pix2pix
```
### Resources and Links

As part of the DEFENSE GAN training, 8000 adversarial versions of images within the BDD100K training image set were generated. Access is provided via the following Google Drive links:

- [BDD100K Adversarial Dataset](https://drive.google.com/file/d/1KwMhYnrA73iJYGfE-Pb-cjVdPgZPD13j/view?usp=drive_link)

### Outside Sources

- [YOLOP GitHub Repository](https://github.com/hustvl/YOLOP)
- [BDD100K Dataset](https://github.com/bdd100k/bdd100k)
- [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [BDD100K Dataset Download](https://drive.google.com/drive/folders/1X5fSRvaxh52aTvtHfkf5Rn24UqjXFN_9?usp=sharing)
