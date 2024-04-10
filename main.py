import os
import cv2 as cv
import numpy as np
from PIL import Image
from utils import model, tools
import torch




# Charger le modèle et appliquer les transformations à l'image
seg_model, transforms = model.get_model()


def draw_bounding_box_around_person(image, contour):
    # Calculate bounding box coordinates for the contour
    x, y, w, h = cv.boundingRect(contour)
    # Draw the bounding box on the image
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

def generate_mask(target_people):

    for target in target_people:

        # Ouvrir l'image et appliquer les transformations
        image_path = os.path.join(target["folder"], target["src"])
        image = Image.open(image_path).convert('RGB')
        transformed_img = transforms(image)

        # Effectuer l'inférence sur l'image transformée sans calculer les gradients
        with torch.no_grad():
            output = seg_model([transformed_img])

        # Traiter le résultat de l'inférence
        result_image = tools.process_inference(output,image,target["folder"],False)
    
def extract_masks(imageFile):

    # Ouvrir l'image et appliquer les transformations
    image = Image.open(imageFile).convert('RGB')
    transformed_img = transforms(image)

    # Effectuer l'inférence sur l'image transformée sans calculer les gradients
    with torch.no_grad():
        output = seg_model([transformed_img])

    # Traiter le résultat de l'inférence
    masks = tools.process_inference(output,image,target["folder"],True)
    
    #show each mask
    #for i, mask in enumerate(masks):
    #    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    #    mask_image.show()
    #    cv.waitKey(0)
        
    return masks   


def load_mask(mask_path):
    mask = np.load(mask_path).astype(np.uint8)
    return mask

#calculate Histogram -> add parameter values such as the image or what ever
def calc_histogram(image, mask):
    # Color space conversion for robust masking
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Convert boolean mask to single-channel integer image (recommended)
    if mask.ndim > 2:
        mask = np.logical_or.reduce(mask, axis=1).astype(np.uint8) * 255
    
        
    # Ensure mask size matches image
    if mask.shape != image.shape[:2]:
        mask = cv.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv.INTER_NEAREST)

    # Perform bitwise AND using a copy of the image for efficiency
    hsv_masked = cv.bitwise_and(hsv_image, hsv_image, mask=mask)

    # Efficiently calculate full and half image histograms
    hist_full = cv.calcHist([hsv_masked], [0, 1], None, [16, 16], [0, 180, 0, 256], accumulate=False)
    cv.normalize(hist_full, hist_full, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    hist_half = cv.calcHist([hsv_masked[:hsv_masked.shape[0] // 2, :]], [0, 1], None, [16, 16], [0, 180, 0, 256], accumulate=False)
    cv.normalize(hist_half, hist_half, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    return hist_full, hist_half



# Point d'entrée principal du script
if __name__ == "__main__":

    target_people = [
            #person1
            {"folder":"targets\\target1", "src" : "person_1.png"},
            #person2
            {"folder":"targets\\target2", "src" : "person_2.png"},
            #person3
            {"folder":"targets\\target3", "src" : "person_3.png"},
            #person4
            {"folder":"targets\\target4", "src" : "person_4.png"},
            #person5
            {"folder":"targets\\target5", "src" : "person_5.png"},
    ]


    #create a mask from the 5 base images in the folder and store them in their respective subfolder
    #generate_mask(target_people)


    for target in target_people:
        target_mask_path = os.path.join(target["folder"],"saved_mask.npy")
        target_image_path = os.path.join(target["folder"],target["src"])
        target_image = cv.imread(target_image_path)
        target_mask = load_mask(target_mask_path)
        full_and_half_baseImg_hist = calc_histogram(target_image, target_mask)
        masks=[]

        #Variable containing the two folders where to read images
        allFolders=["images/cam0","images/cam1"]

        for folder in allFolders:
            # Iterate through each file in the folder cam0/ca1
            for filename in os.listdir(folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    # Load the image

                    image_path = os.path.join(folder, filename)
                    image = cv.imread(image_path)


                    # Convert the image to grayscale
                    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

                    blur_image = cv.GaussianBlur(gray_image, (5, 5), 0)
                    # Threshold the image to separate people
                    _, thresholded_image = cv.threshold(blur_image, 70, 255, cv.THRESH_BINARY_INV)
                    # edges = cv.Canny(blur_image, 150, 350)

                    #extract mask of each person in the image
                    masks=extract_masks(image_path)

                    

                    threshold = 0.8  # Adjust the threshold as needed
                    best_correlation = 0
                    best_contour = None
                    best_mask=None
                    area_threshold = 200 # Adjust the area threshold

                    # Iterate through each mask of people found
                    for mask in masks:
                        # Find contours in the mask
                        contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                        for contour in contours:
                            x, y, w, h = cv.boundingRect(contour)
                            # aspect_ratio = float(w) / h

                            if cv.contourArea(contour) < area_threshold:
                                continue

                            # if 0.8 >= aspect_ratio or aspect_ratio >= 1.2:
                            #     continue

                            # Create a mask for the current person
                            person_mask = np.zeros(mask.shape, np.uint8)
                            cv.drawContours(person_mask, [contour], 0, 255, -1)

                            #cv.imshow('HSV Image', thresholded_image)
                            #cv.imshow('Mask', person_mask)  # Visualize the mask in a way that makes it visible (mask * 255)
                            #cv.waitKey(0)


                            # Calculate histogram of the current person's mask. Crop it to the contour's bounding rect
                            full_and_half_person_hist = calc_histogram(image[y:y+h, x:x+w],person_mask[y:y+h, x:x+w])

                            #select the highest value from the 4 comparisons
                            for baseImgHist in full_and_half_baseImg_hist:
                                for queryImgHist in full_and_half_person_hist:
                                    # Compare histograms using correlation coefficient
                                    correlation = cv.compareHist(baseImgHist, queryImgHist, cv.HISTCMP_CORREL)

                                    # Optionally, set a threshold to determine if the initial person matches the current person
                                    if correlation > best_correlation:
                                        best_correlation = correlation
                                        best_contour = contour
                    
                    if best_correlation > threshold and best_contour is not None:
                        draw_bounding_box_around_person(image, best_contour)
                        target_folder=os.path.join(target["folder"], "output")
                        os.makedirs(target_folder,exist_ok=True) ## create folder if doesn't exist
                        print(f"Initial person matches person in image: {filename} (Correlation: {best_correlation})")
                        save_path = os.path.join(target_folder, filename)
                        cv.imwrite(save_path,image)
                        print(f"Image saved in the folder {target_folder}")
                    
        