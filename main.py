import os
import cv2 as cv
import numpy as np
from PIL import Image
from utils import model, tools
import torch



def generate_mask(target_people):

    for target in target_people:
        # Charger le modèle et appliquer les transformations à l'image
        seg_model, transforms = model.get_model()

        # Ouvrir l'image et appliquer les transformations
        image_path = os.path.join(target["folder"], target["src"])
        image = Image.open(image_path).convert('RGB')
        transformed_img = transforms(image)

        # Effectuer l'inférence sur l'image transformée sans calculer les gradients
        with torch.no_grad():
            output = seg_model([transformed_img])

        # Traiter le résultat de l'inférence
        result = tools.process_inference(output,image,target["folder"])




#calculate Histogram -> add parameter values such as the image or what ever
def calc_histogram(image,mask_file,npy_format):
    # Color space conversion for robust masking
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Load mask (handle both NumPy array and image formats)
    if npy_format:
        mask = np.load(mask_file)
        if not isinstance(mask, np.ndarray):
            mask = np.asarray(mask)  # Convert to NumPy array if necessary
    else:
        mask = cv.imread(mask_file, cv.IMREAD_GRAYSCALE)

    # Ensure mask size matches image
    if mask.shape[:2] != image.shape[:2]:
        mask = cv.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv.INTER_NEAREST)

    # Convert boolean mask to single-channel integer image (recommended)
    if mask.dtype == bool:
        mask_uint8 = np.zeros_like(image, dtype=np.uint8)
        mask_uint8[mask] = 255
        mask = mask_uint8

    # Perform bitwise AND using a copy of the image for efficiency
    hsv_masked = cv.bitwise_and(hsv_image.copy(), hsv_image.copy(), mask=mask)

    # Efficiently calculate full and half image histograms
    hist_full = cv.calcHist([hsv_masked], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv.normalize(hist_full, hist_full, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    hist_half = cv.calcHist([hsv_masked[:hsv_masked.shape[0] // 2, :]], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv.normalize(hist_half, hist_half, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    return hist_full, hist_half



# Point d'entrée principal du script
if __name__ == "__main__":

    target_people = [
            #person1
            {"folder":"targets/target1", "src" : "person_1.png"},
            #person2
            {"folder":"targets/target2", "src" : "person_2.png"},
            #person3
            {"folder":"targets/target3", "src" : "person_3.png"},
            #person4
            {"folder":"targets/target4", "src" : "person_4.png"},
            #person5
            {"folder":"targets/target5", "src" : "person_5.png"},
    ]


    #create a mask from the 5 base images in the folder and store them in their respective subfolder
    #generate_mask(target_people)


    for target in target_people:
        target_mask_path=os.path.join(target["folder"],"saved_mask.npy")
        target_image_path=os.path.join(target["folder"],target["src"])
        target_image=cv.imread(target_image_path)
        full_and_half_baseImg_hist=calc_histogram(target_image,target_mask_path,True)

        # Iterate through each file in the folder cam0
        for filename in os.listdir("images/cam0"):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Load the image
                image_path = os.path.join("images/cam0", filename)
                image = cv.imread(image_path)

                # Convert the image to grayscale
                gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

                # Threshold the image to separate people
                _, thresholded_image = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY)

                # Find contours of people in the image
                contours, _ = cv.findContours(thresholded_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                # Iterate through each person in the image
                for contour in contours:
                    # Create a mask for the current person
                    mask = np.zeros(gray_image.shape, np.uint8)
                    cv.drawContours(mask, [contour], 0, 255, -1)

                    # Calculate histogram of the current person's mask
                    full_and_half_person_hist = calc_histogram(image,mask,True)

                    
                    #select the highest value from the 4 comparisons
                    for baseImgHist in full_and_half_baseImg_hist:
                        for queryImgHist in full_and_half_person_hist:
                            # Compare histograms using correlation coefficient
                            correlation = cv.compareHist(baseImgHist, queryImgHist, cv.HISTCMP_CORREL)

                            # Optionally, set a threshold to determine if the initial person matches the current person
                            threshold = 0.9  # Adjust the threshold as needed
                            if correlation > threshold:
                                target_folder="targets/target4/output"
                                os.mkdir(target_folder,exit_ok=True) ## create folder if doesn't exist
                                print(f"Initial person matches person in image: {filename} (Correlation: {correlation})")
                                cv.imwrite(target_folder,filename)
                                print(f"Image saved in the folder {target_folder}")
                   
    