import shutil
import os

collaboration_dir = "C:/Users/leere/Documents/assembly dataset/Data Labelling/collaboration_annotations"
recognition_dir = "C:/Users/leere/Documents/assembly dataset/Data Labelling/Action/label_tool/action segmentation annotations/action_recognition_labels"
segmentation_dir = "C:/Users/leere/Documents/assembly dataset/Data Labelling/Action/label_tool/action segmentation annotations/action_segmentation_labels"

annotation_dir = "C:/Users/leere/Desktop/HAViD_temporalAnnotation"

def move_collaboration():
    check = 'check'
    for collaboration_file in os.listdir(collaboration_dir):
        if 'action' in collaboration_file:
            for subdataset_folder in os.listdir(annotation_dir):
                subdataset_dir = os.path.join(annotation_dir,subdataset_folder) 
                if os.path.isdir(subdataset_dir):
                    src = os.path.join(collaboration_dir,collaboration_file)
                    collaboration_renamed = collaboration_file.split('_')[0]+'_'+subdataset_folder+'_collaboration.txt'
                    dst = os.path.join(subdataset_dir,'collaboration_timestamps',collaboration_renamed)
                    
                    if check != 'auto':
                        check = input('collab: type auto to autorun')
                    shutil.copy(src,dst)

def move_temporal():
    check = 'check'
    for temporal_file in os.listdir(recognition_dir):
            if 'lh' in temporal_file:
                h = 'lh'
            else:
                h = 'rh'
            if 'pt' in temporal_file:
                l = 'pt'
            else:
                l = 'aa'
            for i in range(3):
                subdataset_folder = 'view'+str(i)+'_'+h+'_'+l
                subdataset_dir = os.path.join(annotation_dir,subdataset_folder)
                src = os.path.join(recognition_dir,temporal_file)
                temporal_renamed = temporal_file.split('_')[0]+'_'+subdataset_folder+'_temporal.txt'
                dst = os.path.join(subdataset_dir,'temporal_timestamps',temporal_renamed)
                
                if check != 'auto':
                    check = input('recog: type auto to autorun')
                shutil.copy(src,dst)

move_temporal()