#!/usr/bin/env python
import os
import glob


def build_list(
        root, image_path, label_path, lst_path,
        is_fine=True, sample_rate=1, is_crop=True):
    index = 0
    lst = []
    label_prefix = 'gtFine_labelIds' if is_fine else 'gtCoarse_labelIds'

    all_images = glob.glob(os.path.join(root, image_path, '*/*.png'))
    all_images.sort()
    for p in all_images:
        l = p.replace(
                image_path, label_path).replace('leftImg8bit', label_prefix)
        index += 1
        if index % sample_rate == 0:
            if is_crop:
                for i in range(1, 8):
                    lst.append([str(index), p, l, "512", str(256 * i)])
            else:
                lst.append([str(index), p, l])

    train_out = open(lst_path, "w")
    for line in lst:
        print >> train_out, '\t'.join(line)


def main():
    root = '/home/acgtyrant/BigDatas/Cityscapes'
    train_image_path = 'leftImg8bit/train'
    train_label_path = 'gtFine/train'
    train_lst_path = 'train_bigger_patch.lst'
    build_list(
            root, train_image_path, train_label_path, train_lst_path)
    val_image_path = 'leftImg8bit/val'
    val_label_path = 'gtFine/val'
    val_lst_path = 'val_bigger_patch.lst'
    build_list(
            root, val_image_path, val_label_path, val_lst_path, is_crop=False)

if __name__ == '__main__':
    main()
