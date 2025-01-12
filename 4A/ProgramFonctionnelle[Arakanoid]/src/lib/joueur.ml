open Graphics
open Parametres


type joueur = int * int * int
(*   ives : int;
  score : int; 
  niveau : int;
  *)

(*CONTRAT
Fonction qui crée un joueur avec un nombre initial de vies
Résultat : un tuple représentant le joueur avec un nombre initial de vies et un score initial de 0
*)
let create_joueur (score, niveau) : joueur =
  (vies_joueur, score, niveau) 

(*CONTRAT
Fonction qui décrémente le nombre de vies du joueur
Paramètre : joueur, un tuple représentant le joueur
Résultat : un tuple représentant le joueur avec un nombre de vies diminué de 1 et le score inchangé
*)
let decrementer_vie (joueur : joueur) : joueur =
  let vies, score, niveau = joueur in
  (vies - 1, score, niveau)

(*CONTRAT
Fonction qui vérifie si le joueur est mort (nombre de vies inférieur ou égal à 0)
Paramètre : joueur, un tuple représentant le joueur
Résultat : un booléen indiquant si le joueur est mort (true) ou non (false)
*)
let joueur_mort (joueur : joueur) : bool =
  let vies, _, _ = joueur in 
  vies <= 0


(*CONTRAT
Fonction qui incrémente le score du joueur
Paramètre : joueur, un tuple représentant le joueur
Résultat : un tuple représentant le joueur avec le score augmenté du nombre de points par brique cassée et les vies inchangées
*)
let incrementer_score (joueur : joueur) : joueur =
  let vies, score, niveau = joueur in
  (vies, score + points_par_brique_cassee, niveau)


(* Fonction pour afficher le score du joueur *)
(*CONTRAT
Fonction qui affiche le score du joueur à l'écran
Paramètre : joueur, un tuple représentant le joueur
Effet de bord : affiche le score à l'écran
*)
let draw_score joueur =
  let _, score, _ = joueur in
  let score_string = "Score: " ^ string_of_int score in
  let score_width, score_height = text_size score_string in
  let score_x = (window_width - score_width) / 2 - 100 in
  let score_y = score_height + 10 in
  set_color white;  
  fill_rect score_x score_y score_width score_height;
  set_color black;  
  moveto score_x score_y;
  draw_string score_string;;


(* Fonction pour afficher la vie du joueur *)
(*CONTRAT
Fonction qui affiche le nombre de vies du joueur à l'écran
Paramètre : joueur, un tuple représentant le joueur
Effet de bord : affiche le nombre de vies à l'écran
*)
let draw_vie joueur =
  let vies, _, _ = joueur in
  let vie_string = "Vie: " ^ string_of_int vies in
  let vie_width, vie_height = text_size vie_string in
  let vie_x = (window_width - vie_width) / 2  in
  let vie_y = vie_height + 10 in  (* Position en dessous du score *)
  set_color white;
  fill_rect vie_x vie_y vie_width vie_height;
  set_color red;
  moveto vie_x vie_y;
  draw_string vie_string;;


 (* CONTRAT
   Fonction qui dessine le niveau du joueur sur l'écran
   Type : int * int * int -> unit
   Paramètre : joueur, un triplet représentant les coordonnées et le niveau du joueur (seul le niveau est utilisé)
   Résultat : la fonction ne renvoie rien (type unit)
*)

let draw_niveau joueur =
  let _,_,niveau = joueur in
  let niveau_string = "Niveau: " ^ string_of_int niveau in
  let niveau_width, niveau_height = text_size niveau_string in
  let niveau_x = (window_width - niveau_width) / 2 + 100 in
  let niveau_y = niveau_height + 10 in  (* Position en dessous du score *)
  set_color white;
  fill_rect niveau_x niveau_y niveau_width niveau_height;
  set_color blue;
  moveto niveau_x niveau_y;
  draw_string niveau_string;;