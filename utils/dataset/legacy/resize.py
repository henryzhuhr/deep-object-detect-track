import os
import yaml
import tqdm
import cv2
import argparse
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-path', type=str, default='~/data/objdect-dataset')
    parser.add_argument('--height',type=int, default=640)
    parser.add_argument('--is_rename', action='store_true', default=True)
    args = parser.parse_args()
    return args



def isImage(file_path: str) -> bool:
    return True if os.path.splitext(file_path)[-1][1:] in ['jpg', 'jpeg', 'png'] else False


if __name__ == '__main__':
    args=parse_args()
    dataset_path:str=os.path.expanduser(args.dataset_path)
    is_rename:bool=not args.is_rename
    height=args.height
    img_dir = os.path.join(dataset_path, 'src')
    ren_dir = os.path.join(dataset_path, 'labeled')
    for class_name in os.listdir(img_dir):
        if not os.path.isdir(os.path.join(img_dir,class_name)):
            continue
        print()
        print(' -- Copy Class: %s' % class_name)
        src_class_dir = os.path.join(img_dir, class_name)
        ren_class_dir = os.path.join(ren_dir, class_name)

        pbar = tqdm.tqdm(os.listdir(src_class_dir))
        not_image_list = []
        n_renamed = 0
        for file in pbar:
            if isImage(file):
                os.makedirs(os.path.join(ren_dir, class_name), exist_ok=True)
                file_name = '%s-%05d.jpg' % (class_name, n_renamed) if is_rename else file
                img = Image.open(os.path.join(img_dir, class_name, file))
                h = height
                w = int(h*img.width/img.height)
                img = img.resize((w, h))
                img.save(os.path.join(ren_dir, class_name, file_name))
                n_renamed += 1
            else:
                not_image_list.append(os.path.join(src_class_dir, file))
            pbar.set_description('[%s] %s' % (class_name, file))