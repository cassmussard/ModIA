open Parametres

type racquet = int * int * int * int 
(*   mutable position_racquet : int;
  width_racquet : int;
  height_racquet : int;*)


(* Création de la raquette *)
(*CONTRAT
Fonction qui crée une raquette à partir des coordonnées et dimensions spécifiées
Paramètre : x, la coordonnée x de la raquette
Paramètre : y, la coordonnée y de la raquette
Paramètre : width, la largeur de la raquette
Paramètre : height, la hauteur de la raquette
Résultat : une raquette représentée par un tuple (x, y, width, height)
*)
let create_racquet (x, y, width, height) : racquet =
  (x, y, width, height)




(* Déplacer la raquette*)
(*CONTRAT
Fonction qui déplace une raquette en fonction de la position de la souris
Paramètre : raquette, la raquette à déplacer représentée par un tuple (_, y, width, height)
Résultat : une nouvelle raquette déplacée que horizontalement en fonction de la position de la souris
*)

let move_racquet (_, y, width, height) : racquet =
  let mouse_x, _ = Graphics.mouse_pos () in
  let max_x = window_width - width in
  let new_x = max 0 (min mouse_x max_x) in
  (new_x, y, width, height)

(* Affichage de la raquette *)
(*CONTRAT
Fonction qui affiche une raquette à l'écran avec les paramètres spécifiés
Paramètre : x, la coordonnée x de la raquette
Paramètre : y, la coordonnée y de la raquette
Paramètre : width, la largeur de la raquette
Paramètre : height, la hauteur de la raquette
Effet de bord : affiche la raquette à l'écran avec les dimensions et la couleur spécifiées
*)
let draw_racquet (x, y, width, height) =
  Graphics.set_color racquet_color;
  Graphics.fill_rect x y width height
