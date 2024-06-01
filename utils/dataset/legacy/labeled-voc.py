import os
import tqdm
import yaml
import shutil
import xml.etree.ElementTree as ET

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-path', type=str, default='~/data/objdect-dataset')
    parser.add_argument('--shuffle', action='store_true', default=True)
    return parser.parse_args()

if __name__=="__main__":
    args=parse_args()

    dataset_path:str=os.path.expanduser(args.dataset_path)
    SHUFFLE: bool = args.shuffle


    VOC_DIR = os.path.join(dataset_path, 'VOC')
    LABLED_DIR = os.path.join(dataset_path, 'labeled')

    for dir in ['JPEGImages', 'Annotations', 'ImageSets/Main']:
        os.makedirs(os.path.join(VOC_DIR, dir), exist_ok=True)

    CLASS_LIST=[]
    labeled_dict = {
        'train': [],
        'val': []
    }

    for class_name in os.listdir(LABLED_DIR):
        if not os.path.isdir(os.path.join(LABLED_DIR, class_name)):
            continue

        if class_name not in CLASS_LIST:
            CLASS_LIST.append(class_name)
        print(chr(128640), 'Devide class: %s' % class_name)
        pbar = tqdm.tqdm(os.listdir(os.path.join(LABLED_DIR, class_name)))
        no_labeled = []
        labeled_list = []
        for file in pbar:
            pbar.set_description(file)

            # check file type
            file_type = os.path.splitext(file)[-1]
            if file_type not in ['.jpg', '.jpeg', '.png']:
                continue

            # check pair: img-xml
            src_img = os.path.join(LABLED_DIR, class_name, file)
            src_xml = os.path.join(LABLED_DIR, class_name,
                                os.path.splitext(file)[0]+'.xml')
            if not os.path.exists(src_xml):
                no_labeled.append(src_img)
                continue
            else:
                dst_img = os.path.join(VOC_DIR, 'JPEGImages', file)
                dst_xml = os.path.join(
                    VOC_DIR, 'Annotations', os.path.splitext(file)[0]+'.xml')
                labeled_list.append({
                    'file_name': os.path.splitext(file)[0],
                    'src_img': src_img,
                    'src_xml': src_xml,
                    'dst_img': dst_img,
                    'dst_xml': dst_xml,
                })

        if len(no_labeled) > 0:
            print('these image may not be labeled', os.linesep, no_labeled)

        num_train = len(labeled_list)
        for d in labeled_list[:num_train]:
            labeled_dict['train'].append(d)
        for d in labeled_list[num_train:]:
            labeled_dict['val'].append(d)

    with open(os.path.join(VOC_DIR, 'ImageSets/Main/classes.names'), 'w')as f:
        for c in CLASS_LIST:
            f.write(c+'\n')

    print()
    sets = ['train', 'val']
    for set_type in sets:
        with open(os.path.join(VOC_DIR, 'ImageSets/Main/%s.txt' % (set_type)), 'w')as f:
            print(chr(128640), 'copy %5s set' % set_type)
            pbar=tqdm.tqdm(labeled_dict[set_type])
            for data_dict in pbar:
                pbar.set_description(data_dict['file_name'])
                f.write(data_dict['file_name']+'\n')
                shutil.copyfile(data_dict['src_img'],data_dict['dst_img'])
                shutil.copyfile(data_dict['src_xml'],data_dict['dst_xml'])