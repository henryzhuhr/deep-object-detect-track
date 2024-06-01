import os
import tqdm
import yaml
import shutil
import xml.etree.ElementTree as ET
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-path', type=str, default='~/data/objdect-dataset')
    return parser.parse_args()

if __name__=="__main__":
    args=parse_args()

    dataset_path:str=os.path.expanduser(args.dataset_path)


    VOC_DIR = os.path.join(dataset_path,'VOC')

    with open(os.path.join(VOC_DIR,'ImageSets','Main','classes.names'),'r') as f:
        CLASS_LIST=list(f)
    for i in range(len(CLASS_LIST)):
        CLASS_LIST[i]=CLASS_LIST[i].replace('\n','')
    print(chr(128640),'find class: ',CLASS_LIST,os.linesep)



    def convert(size, box):
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x, y, w, h)
    def convert_annotation(image_id):
        in_file = open(VOC_DIR + '/Annotations/%s.xml' % (image_id))
        out_file = open(VOC_DIR + '/labels/%s.txt' % (image_id), 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in CLASS_LIST or int(difficult) == 1:
                continue
            cls_id = CLASS_LIST.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
                xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " +
                        " ".join([str(a) for a in bb]) + '\n')

    sets = ['train', 'val']
    for set_type in sets:
        os.makedirs(os.path.join(VOC_DIR,'labels'),exist_ok=True)
        with open(os.path.join(VOC_DIR,'ImageSets','Main','%s.txt'%set_type),'r')as f:
            file_names=f.read().split('\n')[:-1]
            for i in range(len(file_names)): # ignore data like 'img_00 -1', convert to 'img_00'
                file_names[i]=str(file_names[i]).split(' ')[0]
        
        pbar=tqdm.tqdm(file_names)
        for file_n in pbar:
            pbar.set_description('convert %5s set: %s'%(set_type,file_n))
            convert_annotation(file_n)
        