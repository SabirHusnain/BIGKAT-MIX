import os
import shutil
from PIL import Image as img


def mergeFiles(pathFrom, offset, pathSave):
    img_list = os.listdir(pathFrom)
    for item in img_list:
        if (item.endswith('.tiff')):
            name = item.split('_')
            #   print(name)
            number, ext = os.path.splitext(name[3])
            print(number)
            with img.open(os.path.join(pathFrom, item)) as im:
                im.save(os.path.join(
                    pathSave, name[0]+'_'+name[1]+'_'+name[2]+'_'+str(int(number)+offset)+'.tiff'))


mergeFiles('Calib 4/stereo/client_stereo_right',
           173, 'Calib F/stereo/client_stereo_right')
