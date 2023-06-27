import os
import cv2
from tqdm import tqdm
import shutil

def createFolder(folder):
    try:
        os.mkdir(folder)
    except FileExistsError as fe:
        print('Folder already exists')
    except OSError as oe:
        print(oe)
        print('Directory cannot be created')
        exit()

def removeDirectory(folder):
    try:
        shutil.rmtree(folder)
        print("directory is removed successfully")
    except OSError as x:
        print("Error occured: %s : %s" % (folder, x.strerror))

def main():
    parentDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    datasetPath = os.path.join(os.path.join(parentDir, 'datasets'), 'Orginal')

    
    imagesDir = os.path.join(datasetPath, "imgs")
    maskDir = os.path.join(datasetPath, "masks")
    resultImg = os.path.join(datasetPath, "result_images")
    resultMask = os.path.join(datasetPath, "result_masks")

    #create result image and masks directory
    createFolder(resultImg)
    createFolder(resultMask)


    os.makedirs(resultImg, exist_ok=True)
    os.makedirs(resultMask, exist_ok=True)

    for file in tqdm(os.listdir(imagesDir)):
        if not(file.endswith(".jpeg") or file.endswith(".jpg") or file.endswith(".png")):
            continue
        image = cv2.imread(os.path.join(imagesDir, file))
        cv2.imwrite(os.path.join(resultImg, file.replace(".jpg", ".png").replace(".jpeg", ".png")), image)
    
    for file in tqdm(os.listdir(maskDir)):
        if not(file.endswith(".jpeg") or file.endswith(".jpg") or file.endswith(".png")):
            continue
        image = cv2.imread(os.path.join(maskDir, file))
        cv2.imwrite(os.path.join(resultMask, file.replace(".jpg", ".png").replace(".jpeg", ".png")), image)
    

    removeDirectory(imagesDir)
    removeDirectory(maskDir)

    shutil.copytree(resultImg, imagesDir)
    shutil.copytree(resultMask, maskDir)

    removeDirectory(resultMask)
    removeDirectory(resultImg)
    


if __name__ == "__main__":
    main()