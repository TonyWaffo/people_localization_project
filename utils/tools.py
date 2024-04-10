import os
import numpy as np
from utils.config import THRESHOLD, PERSON_LABEL, MASK_COLOR, ALPHA
from PIL import Image

# Fonction pour traiter les sorties d'inférence du modèle
def process_inference(model_output, image,mask_output_dir,generate_mask_only):
    np_masks = []
    # Extraire les masques, les scores, et les labels de la sortie du modèle
    masks = model_output[0]['masks']
    scores = model_output[0]['scores']
    labels = model_output[0]['labels']

    # Convertir l'image en tableau numpy
    img_np = np.array(image)

    # Parcourir chaque prédiction pour appliquer le seuil et le label
    for i, (mask, score, label) in enumerate(zip(masks, scores, labels)):
    
        # Appliquer le seuil et vérifier si le label correspond à une personne
        if score > THRESHOLD and label == PERSON_LABEL:
            
            # Convertir le masque en tableau numpy et l'appliquer à l'image            
            mask_np = mask[0].mul(255).byte().cpu().numpy() > (THRESHOLD * 255) 
            np_masks.append(mask_np)            

            for c in range(3):
                img_np[:, :, c] = np.where(mask_np, 
                                        (ALPHA * MASK_COLOR[c] + (1 - ALPHA) * img_np[:, :, c]),
                                        img_np[:, :, c])
    ## (Optional) save mask
        if generate_mask_only is False:
            #mask_filename = os.path.splitext(os.path.basename(image.filename))[0] + '_mask.npy'
            mask_path = os.path.join(mask_output_dir, "saved_mask.npy")
            with open(mask_path, 'wb') as f:
                np.save(f,np_masks)

    # renvoie uniquement la liste des masques de l'image
    if generate_mask_only is True:
        return np_masks
    
    # Convertir en image à partir d'un tableau numpy et le renvoyer            
    return Image.fromarray(img_np.astype(np.uint8))

def apply_saved_mask(image):

    # Convertir l'image en tableau numpy
    img_np = np.array(image)
    masks = np.load('examples/output/saved_masks.npy')
    # Parcourir chaque prédiction pour appliquer le seuil et le label
    for i, mask in enumerate(masks):  
        for c in range(3):
            img_np[:, :, c] = np.where(mask, 
                                    (ALPHA * MASK_COLOR[c] + (1 - ALPHA) * img_np[:, :, c]),
                                    img_np[:, :, c])
    
    # Convertir en image à partir d'un tableau numpy et le renvoyer            
    return Image.fromarray(img_np.astype(np.uint8))