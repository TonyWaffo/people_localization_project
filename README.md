 
CSI 4533 
Projet – Partie 3
 

Membres du groupe
•	Tony Waffo
•	Arild Yonkeu Tchana





Introduction
Dans le cadre de ce projet, il était question d'utiliser des histogrammes pour identifier les personnes dans une séquence vidéo. Afin d’avoir les résultats optimaux, nous combinons alors la comparaison d’histogramme et le modèle de détection de personnes fourni dans un répertoire GitHub

Méthodologie et explication
Dans un premier temps, nous avons créé une fonction « generate_mask » qui extrait le masque des personnes qu’on recherche dans nos séquences vidéo. Ainsi pour chacune de ces personnes nous avons créé un répertoire (« target1 », « target2 », « target3 », « target4 » ou « target5 » ) à l’intérieur du dossier « targets ». Dans ces dossiers individuels, nous avons donc une image de la personne recherchée, son masque, ainsi qu’un dossier « output » contenant les résultats de recherche de cette personne.
En appliquant les modèles et en utilisant le comparateur d’histogramme, nous arrivons ainsi à identifier des corps humains dans les séquences des dossiers « images/cam0 » et « images/cam1 », puis nous les comparons avec la personne courante recherchée. S’il y a un match, on encadre l’individu d’une boite englobante verte et on enregistre l’image en question dans notre dossier « output » situé dans le répertoire « targets/target1 » (pour la personne numéro 1 par exemple).


Conclusion
Notre approche basée sur les histogrammes et les modèles s'avère efficace pour la détection de personnes dans des vidéos. Cependant, des erreurs de détection peuvent survenir lorsque deux entités distinctes ont des variances de couleurs similaires. Pour améliorer la précision, nous pouvons explorer plusieurs pistes: améliorer la discrimination des couleurs, définir un seuil dynamique en fonction du contexte de la vidéo, intégrer des techniques de suivi de mouvement, utiliser des modèles d'apprentissage profond:
En conclusion, notre approche est prometteuse et offre un grand potentiel pour la détection de personnes dans des vidéos. En explorant les pistes d'amélioration suggérées, nous pouvons encore accroître la performance de notre méthode.




## Vous utiliserez GIT pour récupérer le code sur votre ordinateur. Assurez-vous que GIT est installé. Veuillez suivre cette documentation :
```
https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
```

## La première étape consiste à cloner le dépôt dans un emplacement de votre choix avec cette commande :
```
git clone https://gitlab.com/zatoitche/inst_seg.git
```


## Téléchargez images.zip pour la deuxième partie de votre laboratoire à partir du lien suivant :
```
https://drive.google.com/file/d/1potC4tmKjvLAlXSmhaGH59u-g5u4qg5-/view?usp=drive_link
```


## Extrayez images.zip et placez le dossier "images" extrait dans le répertoire du projet que vous avez cloné précédemment.

## Dans le Terminal ou le CommandLine, naviguez jusqu'au répertoire du projet que vous avez cloné.

## Créez un environnement virtuel appelé "env" avec la commande suivante dans votre Terminal ou invite de commande:
```
python3 -m venv env
```

## Activez l'environnement virtuel avec cette commande (assurez-vous d'être dans le répertoire du projet):
```
source env/bin/activate
```

## Installez les dépendances du code avec la commande suivante:
```
pip install -r requirements.txt
```


## Vous êtes maintenant prêt à exécuter le code. Le fichier main.py contient un exemple de code pour obtenir la segmentation d'instances des personnes dans une image. Le code pointe vers les images du dossier "examples". Il traitera les images de "examples/source" et les placera dans "examples/output" pour vos tests. Vous pouvez modifier le code pour traiter à la place les images du dossier "images" pour votre projet.