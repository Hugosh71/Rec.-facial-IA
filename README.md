# Rec.-faciale-IA
Projet étudiant portant sur "la reconnaissance des émotions faciales par l'intelligence artificielle"

## Structure du projet
---------
**DeepFace :** Il s'agit d'un répertoire qui contient un fichier Python Webcam, et un fichier Notebook DeepFace_notebook.
- Webcam.py est un fichier qui utilise la bibliothèque DeepFace pour renseigner les caractéristiques faciales du visage observé à la WebCam de l'utilisateur.  (Pressez 'q' pour arrêter)       
- DeepFace_notebook.ipynb est un fichier qui utilise la bibliothèque DeepFace pour renseigner les caractéristiques faciales d'un visage sur une image en entrée.     
- haarcascade_frontalface_default.xml est un modèle pré-entraîné qui permet de cadrer le visage observé. Nous l'utilisons dasn les fichiers WEBCAM et EMOTION pour montrer le cadre qui se créer lors de la compilation et montrer la cohérence de notre utilisation.
- Des fichiers JPG représentant les émotions les plus communes ont été pré-chargés pour permettre à l'utilisateur de tester l'algorithme et attester de la véracité de notre exploitation. 

**Pytorch :** Contient notament un fichier main.py et search_best_parameters.py
- Dossier function : 
    - Architectures : contient des architectures de modele CNN compatible avec le fichier main et search_best_parameters
        - Model48.py est une architecture personnalisé
        - OurModel.py est également une architecture personnalisé
        - ResNet.py est une des architectures ResNet deja existante
        - VGG.py est une architecture répandu et déjà courament utilisé dans des CNN
    - data.py charge les donnée du fichier fer2013.csv et organise les données dans les tensor il contient 2 fonctions
    - save.py sauvegarde les données de notre modele pytorch dans un repertoir modele créer automatiquement au lancement du code. Il sauvegarde également 2 graphique de métriques de mesure d'entrainement du modele (accuracy / loss)
    - train.py contient toutes les etapes d'entrainement des données notament les calculs des entrainement des images ou encore apres des  mesures du modele
    - transformation.py contient les etapes d'augmentation artificiel du dataset
 - Dossier model : Endroit de sauvegarde des modeles pré entrainé pytorch. 2 modeles pré entrainé sont deja dans le dossier
 - Dossier graphs : Enregistrement des metrics d'entrainement de modele. Plusieurs graphiques y sont deja présent sur des tests deja effectué
 - main.py fonction de lancement de l'entrainement de données. Les parametres présent dans le fichier sont les meilleurs hyperparametres que nous avons trouvé pour entrainer nos données et obtenir 60% d'accuracy environ
 - search_best_parameters.py utilise la librairie optuna et le fichier functions pour chercher des hyperparametre optimaux a notre probleme sur n_iteration à définir

**Dataset :**
    dataset_to_csv.py : permet si le dataset est sous forme d'image de les faires passer sous la meme forme que fer2013 afin de pouvoir rajouter des données
    
**mysite :**
- polls/views.py & polls/camera.py: traitement de l'image envoyé soit par la caméra soit par une importation d'image et prédiction de l'image proposé. Soit en utilisant la librairie deepFace ou le model pré-entrainé model.pth qui se trouve dans le meme repertoir courant.
- templates/index.html : page html qui permet l'interface de la webcam et envoie l'image de la webcam toutes les secondes et recupere l'emotion associé
- les autres fichiers sont des fichiers de projet django qui sont nécéssaire au bon fonctionement du projet

Dataset  
--------
Nous avons utilisés un dataset via Kaggle pour notre projet, ce dataset contient 39900 images des 7 emotions étiqueté : [Dataset's Web site](https://www.kaggle.com/datasets/deadskull7/fer2013).
Pour un bon fonctionnement du projet le fichier fer2013.csv est à mettre dans le dossier Dataset prévu à cet effet.

## Lancer le rendu caméra
--------
Afin de pouvoir visualiser le rendu caméra, il faut télécharger le dossier **mysite**
-En premier, dans polls/views.py : ligne 89 : il faut modifier le chemin d'acces au fichier model.pth (mettre un chemin absolu)
    -faire de meme dans polls/camera.py ligne 89
- installer django sur votre machine :
    ``` pip install django ```
- Ouvrir le terminal dans le repertoir courant du fichier manage.py qui se trouve à la racine du projet. Puis dans le terminal lancer la commande : ```py manage.py runserver```
- Enfin dans votre navigateur copiez cette url : [Django run local](http://localhost:8000/polls/)

Sur le site, vous pouvez comparer la librairie DeepFace deja existante et le model ResNet que nous avons entrainé. Vous pouvez inserer une image et ainsi afficher l'émotion detecté de cette personne par notre modele ResNet.

## Lancer un entrainement de modele
--------
- Telecharger le fichier Pytorch et le fichier Dataset puis installer le fichier fer2013.csv à l'interieur de ce dernier
- lancer le main.py en modifiant les parametres si souhaité.
    - les parametres deja présent sont les meilleurs hyperparametres que nous avons trouvé pour notre entrainement.
