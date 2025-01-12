open Parametres
open Brique
type ball = int * int * int * int 
(*   mutable position_ball : int * int;
  mutable velocity : int * int;
  mutable attached : bool; *)


(* Création de la balle *)
(*CONTRAT
Fonction qui crée une balle à partir des coordonnées de la raquette et des paramètres prédéfinis
Paramètre : racquet_x, la coordonnée x de la raquette
Paramètre : racquet_width, la largeur de la raquette
Paramètre : racquet_y, la coordonnée y de la raquette
Paramètre : racquet_height, la hauteur de la raquette
Résultat : une balle représentée par un tuple (x, y, dx, dy)
*)
let create_ball (racquet_x, _, _, _) : ball =
  let (x, y) = (racquet_x+racquet_width/2, racquet_y+racquet_height) in
  let (dx, dy) = initial_ball_speed in
  (x, y, dx, dy)

(* Affichage de la balle *)
(*CONTRAT
Fonction qui dessine une balle à l'écran avec les coordonnées spécifiées
Paramètre : x, la coordonnée x de la balle
Paramètre : y, la coordonnée y de la balle
Résultat : dessine la balle à l'écran avec les coordonnées et la couleur spécifiées
*)
let draw_ball (x, y, _, _) =
  Graphics.set_color ball_color;
  Graphics.fill_circle x y ball_radius

(* Mise à jour de la position de la balle avant d'avoir cliqué *)
(*CONTRAT
Fonction qui met à jour la position de la balle avant un clic de souris en utilisant les coordonnées de la raquette et les paramètres prédéfinis
Paramètre : racquet_x, la coordonnée x de la raquette
Paramètre : racquet_width, la largeur de la raquette
Paramètre : racquet_y, la coordonnée y de la raquette
Paramètre : racquet_height, la hauteur de la raquette
Résultat : une balle représentée par un tuple (x, y, dx, dy) avec une position mise à jour avant un clic
*)
let update_ball_position_before_click (racquet_x, _, _, _)=
  let (dx, dy) = initial_ball_speed in
  (racquet_x+racquet_width/2, racquet_y+racquet_height, dx, dy)

(* Mise à jour de la position de la balle *)
(*CONTRAT
Fonction qui met à jour la position de la balle en fonction des coordonnées et des vitesses données
Paramètre : x, la coordonnée x actuelle de la balle
Paramètre : y, la coordonnée y actuelle de la balle
Paramètre : dx, la vitesse horizontale de la balle
Paramètre : dy, la vitesse verticale de la balle
Résultat : une nouvelle position de la balle représentée par un tuple (new_x, new_y, dx, dy)
*)
let update_ball_position (x, y, new_dx, new_dy) briques time =
  let dx = new_dx + int_of_float (float_of_int new_dx *. acceleration_factor *. time) in
  let dy = new_dy + int_of_float (float_of_int new_dy *. acceleration_factor *. time) - int_of_float (gravity_factor) in
  let new_x = x + dx in
  let new_y = y + dy in

  (* Collision avec la fenêtre *)
  let collision_with_horizontal =
    new_x - ball_radius <= 0 || new_x + ball_radius >= window_width
  in
  let dx' = if collision_with_horizontal then -dx else dx in

  let collision_with_top = new_y + ball_radius >= window_height in
  let dy' = if collision_with_top then -dy else dy in

  (* Vérifier la collision avec les briques *)
  let collision_with_brick =
    List.exists (collision_brique_balle (new_x, new_y, dx', dy')) briques
  in
  let dy'' = if collision_with_brick then -dy' else dy' in

  let collision_with_bottom = new_y - ball_radius <= 0  in
  let game_over = collision_with_bottom in

  if not game_over then
    (new_x, new_y, dx', dy'')
  else 
    (new_x, new_y, 0, 0)

(* Collision avec la raquette *)
(*CONTRAT
Fonction qui vérifie s'il y a une collision entre la balle et la raquette
Paramètre : rx, la coordonnée x de la raquette
Paramètre : ry, la coordonnée y de la raquette
Paramètre : rw, la largeur de la raquette
Paramètre : rh, la hauteur de la raquette
Paramètre : bx, la coordonnée x de la balle
Paramètre : by, la coordonnée y de la balle
Résultat : un booléen indiquant s'il y a une collision entre la balle et la raquette
*)
let check_collision_racquet (rx, ry, rw, rh) (bx, by, _, _) : bool =
  bx >= rx && bx <= rx + rw && by >= ry && by <= ry + rh

(*CONTRAT
Fonction qui vérifie si la balle a une collision avec le sol
Paramètre : balle, les coordonnées de la balle
Résultat : un booléen indiquant s'il y a une collision avec le sol
*)
let collision_sol (_, y, _, _) = 
  y - ball_radius <= 0

