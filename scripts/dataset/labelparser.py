import datetime
import os
import logging
import shutil
import time
from .logger import Logger
import tqdm
from .voc import convert_annotation

logger=Logger()




class ObjDetectLabelParse():
    def __init__(
        self,
        dataset_root,  # Dataset root directory (include)
    ) -> None:
        self.dataset_root = dataset_root
        self.__check_dir_exsit(self.dataset_root,'dataset')
        self.voc_dir=os.path.join(self.dataset_root,'VOCdevkit','VOC2012') # TODO: VOC2012 / VOC2007
        self.coco_dir=os.path.join(self.dataset_root,'coco')

    # def label2voc(self):
        # for dir in ['JPEGImages', 'Annotations', 'ImageSets/Main']:
        # os.makedirs(os.path.join(VOC_DIR, dir), exist_ok=True)    
    def voc2yolo(self):
        """
            └── VOCdevkit
                └── VOC2012
                    ├── Annotations
                    ├── ImageSets
                    │   └── Main
                    ├── JPEGImages
                    └── labels
        """
        self.__check_dir_exsit(self.voc_dir,'VOC directroy')
        voc_dir=self.voc_dir
        os.makedirs(os.path.join(voc_dir,'labels'),exist_ok=True)   # save yolo format label

        with open(os.path.join(voc_dir,'ImageSets','Main','classes.names'),'r') as f:
            class_list=list(f)
        for i in range(len(class_list)):
            class_list[i]=class_list[i].replace('\n','')
        logger.info('find class: %s'%(class_list))

        
        sets = ['train', 'val','test']
        for set_type in sets:
            set_txt_file=os.path.join(voc_dir,'ImageSets','Main','%s.txt'%set_type)
            if not os.path.exists(set_txt_file):
                logger.warning('%s set file(%s.txt) not found: %s'%(set_type,set_type,set_txt_file))
            else:
                with open(set_txt_file,'r')as f:
                    file_names=f.read().split('\n')[:-1]
                    for i in range(len(file_names)): # ignore data like 'img_00 -1', convert to 'img_00'
                        file_names[i]=str(file_names[i]).split(' ')[0]
                
                logger.info('Read %s file:'%set_type)
                pbar=tqdm.tqdm(file_names)
                for file_n in pbar:
                    pbar.set_description('convert %5s set: %s'%(set_type,file_n))
                    convert_annotation(voc_dir,class_list,file_n)

    def yolov2coco(self):
        voc_dir=self.voc_dir
        self.__check_dir_exsit(voc_dir,'VOC directroy')
        self.__check_dir_exsit(os.path.join(voc_dir,'labels'),'yolo format label')

        coco_dir=os.path.join(self.dataset_root, 'coco')
        os.makedirs(os.path.join(coco_dir, 'images','train'), exist_ok=True)
        os.makedirs(os.path.join(coco_dir, 'images','val'), exist_ok=True)
        os.makedirs(os.path.join(coco_dir, 'labels','train'), exist_ok=True)
        os.makedirs(os.path.join(coco_dir, 'labels','val'), exist_ok=True)
        sets = ['train', 'val']
        for set_type in sets:
            print(chr(128640),'%6s set'%set_type)
            with open(os.path.join(voc_dir, 'ImageSets', 'Main', '%s.txt' % set_type), 'r')as f:
                file_names=f.read().split('\n')[:-1]
                for i in range(len(file_names)): # ignore data like 'img_00 -1', convert to 'img_00'
                    file_names[i]=str(file_names[i]).split(' ')[0]
            pbar=tqdm.tqdm(file_names)
            for file_name in pbar:
                pbar.set_description(file_name)
                # from VOC
                jpg_file = os.path.join(voc_dir, 'JPEGImages', file_name+'.jpg')
                xml_file = os.path.join(voc_dir, 'labels', file_name+'.txt')
                if not os.path.exists(jpg_file):
                    raise FileNotFoundError('%s %s' %(chr(128561), jpg_file))
                if not os.path.exists(xml_file):
                    raise FileNotFoundError('%s %s' %(chr(128561), xml_file))

                # to coco
                image_file=os.path.join(coco_dir,'images',set_type,file_name+'.jpg')
                label_file=os.path.join(coco_dir,'labels',set_type,file_name+'.txt')

                # Copy
                shutil.copyfile(jpg_file,image_file)
                shutil.copyfile(xml_file,label_file)

    def __check_dir_exsit(self,dir:str,name=''):
        if not os.path.exists(dir):
            logger.error('%s not found%s' % (name, dir))
            # raise FileNotFoundError('%s%s %s not found%s' % (
            #     COLORS.LRED,name, dir,COLORS.RESET))
        else:
            logger.info('Found %s: %s' %(name,dir))