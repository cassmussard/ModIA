# Projet AIF

## 1. Système de recommandation

This gradio demo app has 2 functionnality : uploading a film poster and it recommands 5 similar posters or writing a description of a film and it recommands 5 similar film with their synopsis.     
![Alt Text](./image_readme/readme.png)

## How to run
1. Install Docker and Docker compose
2. Clone this repository
3. Run `sudo docker-compose up` in the root directory of this repository
4. Open `localhost:7860` in your browser
5. Upload an image and see the results
6. To stop the server, run 
```sudo docker-compose down``` in the root directory of this repository
7. To remove the containers, run `sudo docker-compose rm` in the root directory of this repository
8. To remove the images, run `sudo docker image prune -a` in the root directory of this repository

## 2. Méthode d'explicabilité (RISE)

Le notebook XAIF_notebook implémente la méthode RISE avec deux réseaux de neurones fine tuné sur le dataset Imagenette. 
Il regroupe aussi deux métriques et la comparaison avec une autre méthode d'explicabilité. 

## Authors

- [Mickael Song](https://github.com/mickaelsong)
- [Karima Ghamnia](https://github.com/KARIIII123)
- [Remi Colin](https://github.com/remicolin2)
- [Cassandra Mussard](https://github.com/cassmussard)
