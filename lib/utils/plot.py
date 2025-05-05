## Process the .json file of pred results, draw images
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random


def plot_img_and_mask(img, mask, index,epoch,save_dir):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.savefig(save_dir+"/batch_{}_{}_seg.png".format(epoch,index))

def show_seg_result(img, result, index, epoch, save_dir=None, is_ll=False,palette=None,is_demo=False,is_gt=False, attack_type=None):
    # img = mmcv.imread(img)
    # img = img.copy()
    # seg = result[0]
    if palette is None:
        palette = np.random.randint(
                0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3 # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    
    # Create segmentation mask
    if not is_demo:
        if result.shape[:2] != img.shape[:2]:
            result = cv2.resize(
                result,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Create color segmentation map
        color_seg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else:
        # Process demo mode
        # First adjust the result size
        if result[0].shape[:2] != img.shape[:2]:
            result = [
                cv2.resize(result[0], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST),
                cv2.resize(result[1], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            ]
        
        color_seg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        color_seg[result[0] == 1] = [0, 255, 0]    # Driving area
        color_seg[result[1] == 1] = [255, 0, 0]    # Lane line

    color_seg = color_seg[..., ::-1]
    
    # Create mask and mix image
    color_mask = np.mean(color_seg, 2) != 0
    img_copy = img.copy() 
    img_copy[color_mask] = img_copy[color_mask] * 0.5 + color_seg[color_mask] * 0.5
    
    img_copy = img_copy.astype(np.uint8)
    img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_LINEAR)

    if not is_demo:
        if not is_gt:
            if not is_ll:
                if attack_type:
                    cv2.imwrite(save_dir + "/batch_{}_{}_{}_da_segresult.png".format(epoch, index, attack_type), img)
                else:
                    cv2.imwrite(save_dir + "/batch_{}_{}_da_segresult.png".format(epoch, index), img)
            else:
                if attack_type:
                    cv2.imwrite(save_dir + "/batch_{}_{}_{}_ll_segresult.png".format(epoch, index, attack_type), img)
                else:
                    cv2.imwrite(save_dir + "/batch_{}_{}_ll_segresult.png".format(epoch, index), img)
        else:
            if not is_ll:
                if attack_type:
                    cv2.imwrite(save_dir + "/batch_{}_{}_{}_da_seg_gt.png".format(epoch, index, attack_type), img)
                else:
                    cv2.imwrite(save_dir + "/batch_{}_{}_da_seg_gt.png".format(epoch, index), img)
            else:
                if attack_type:
                    cv2.imwrite(save_dir + "/batch_{}_{}_{}_ll_seg_gt.png".format(epoch, index, attack_type), img)
                else:
                    cv2.imwrite(save_dir + "/batch_{}_{}_ll_seg_gt.png".format(epoch, index), img)
    return img

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)


if __name__ == "__main__":
    pass