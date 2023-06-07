# Rec.-faciale-IA
Projet étudiant portant sur "la reconnaissance des émotions faciales par l'intelligence artificielle"

Structure du projet
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
