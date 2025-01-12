(* Dimensions de la fenêtre *)
let window_width = 800
let window_height = 600

(* Facteurs d'accélération et de gravité *)
let acceleration_factor = 1e-12
let gravity_factor = 0.9


(* Dimensions de la balle *)
let ball_radius = 10
let ball_color = Graphics.black

(* Vitesse initiale de la balle *)
let initial_ball_speed = (2, 2)

(* Paramètres de la raquette *)
let racquet_width = 100 
let racquet_height = 10
let racquet_y = 90
let racquet_color = Graphics.black



(*Paramètres briques (taille, couleur)*)
let largeur_briques = 50  (* Largeur d'une brique *)
let hauteur_briques = 10  (* Hauteur d'une brique *)
let couleur_brique = Graphics.green  (* Couleur des briques *)


(* Nombre de lignes de briques *)
let lignes_briques = 8

(* Nombre de briques par ligne *)
let briques_par_lignes = 14

(* Espacement entre les briques *)
let espacement_briques = 12

(* Paramètres du joueur *)
let vies_joueur = 1
let score_initial = 0
let niveau_initial = 1

let points_par_brique_cassee = 1
let gain_de_score_pour_niveau_termine = 5

(* Taille de textes *)
let taille_game_over = 2