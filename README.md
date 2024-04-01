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