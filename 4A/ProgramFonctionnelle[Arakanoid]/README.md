# Arkanoid



## Introduction

Ce projet vise à réaliser un jeu de type casse briques (Arkanoid) en Ocaml. Dans ce jeu, on trouve une balle, une raquette, des briques, un score (correspondant au nombre de briques cassées), un score de niveau (correspondant au nombre de niveau) et enfin le nombre de vie restant. Le but est de détruire le maximum de briques sans que la balle passe derrière la raquette. Vous êtes en charge de déplacer la raquette et de faire rebondir la balle sur celle-ci pour casser un maximum de briques. Vous possédez un certain nombre de vies pour faire cela et votre score s'affichera à l'écran. 

## Comment installer le jeu ?
Si vous souhaitez télécharger et jouer à notre jeu vous devez :

- Créer un dossier en local dans votre ordinateur ou vous allez stocker le jeu :

```
mkdir arkanoid 
```

- Ensuite, vous devez télécharger le jeu dans votre dossier en local :

```
cd arkanoid 
git clone https://gitlab.com/cassandramussard/arakanoid.git
```

- Pour continuer, vous devez vous placer dans le dossier bin : 
```
cd arakanoid/src/bin
```

- Maintenant vous pouvez exécuter le jeu (il faudra bien évidemment avoir ocaml installé sur votre ordinateur): 
```
eval $(opam env)
dune exec ./main.exe
```


## Comment jouer au jeu ?

En faisant la commande précédente une fenêtre contenant le menu principal s'ouvre. 
Deux options s'offrent à vous : 

- Si vous ne voulez pas essayer le jeu vous pouvez appuyer sur la touche E du clavier et la fenêtre se fermera.

- Si vous voulez essayer le jeu vous pouvez appuyer sur la touche J du clavier. 

En appuyant sur J vous démarrez le jeu. 

Dans un premier temps la balle est attachée à la raquette et vous pouvez les déplacer en utilisant la souris. Une fois que vous êtes sûr de vouloir commencer vous pouvez cliquer sur la souris. Ceci lancera la balle qui se déplacera alors toute seule. 
Vous serez en charge de déplacer la raquette. 
Comme expliqué dans l'introduction vous devez déplacer la raquette de telle manière à ce que la balle ne tombe jamais derrière la raquette (auquel cas vous perdrez une vie). La balle se déplace toute seule en fonction des collisions qu'elle rencontre (mur, brique, raquette). Lorsque la balle rencontre une brique, elle rebondit et la brique disparaît engendrant une augmentation du score. 

Si jamais la balle passe derrière la raquette vous perdez une vie et vous perdez le jeu car vous n'avez qu'une vie. Vous verrez à l'écran votre score et deux options s'offrent à vous. Soit vous pouvez quitter le jeu en cliquant avec la souris sur le bouton 'quitter', soit vous pouvez rejouer en cliquant avec la souris sur le bouton 'rejouer'.

Par contre, si vous réussisez à détruire toutes les briques vous passerez au niveau suivant et le compteur de niveau s'incrémentera de un. 

Si à tout moment vous souhaitez quitter le jeu vous pouvez taper sur la lettre E du clavier et la fenêtre se fermera.


## Comment changer les paramètres du jeu ?
Dans le répertoire bin se trouve un fichier nommé Parametres.ml. Ce fichier contient toutes les constantes necessaires au jeu. Si vous souhaitez changer ces paramètres (vitesse de la balle, taille des briques, couleur des briques, taille de la fenetre, taille de la raquette, ...) il vous suffit de changer le fichier Parametres.ml


## Comment lancer les tests ?

Nous avons rédigé des tests pour chaque fichier et chaque fonction dans ces fichiers. Pour lancer les tests il vous suffit de lancer ces commandes :

```
cd ..
cd test
dune runtest
```
Si rien ne se passe cela signifie que les tests ont réussi.

