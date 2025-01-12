open Graphics
open Parametres


(*Définition du type brique*)
type briques = int * int * bool

(* CONTRAT
   Fonction qui crée une brique avec les coordonnées spécifiées
   Type : int -> int -> int * int * bool
   Paramètre : x, la coordonnée x de la brique
   Paramètre : y, la coordonnée y de la brique
   Résultat : un triplet (x, y, true) représentant la brique créée
*)
let creation_brique x y =
  (x, y, true)

(* CONTRAT
   Fonction qui dessine une brique à partir des coordonnées spécifiées
   Type : int * int * 'a -> unit
   Paramètre : (x, y, _), un triplet représentant les coordonnées de la brique (seul x et y sont utilisés)
   Effet de bord : la brique est dessinée à l'écran avec les paramètres de couleur et de dimensions spécifiés
*)
let dessiner_brique (x, y, _) =
  set_color couleur_brique;
  fill_rect (x - largeur_briques / 2) (y - hauteur_briques / 2) largeur_briques hauteur_briques

(* CONTRAT
   Fonction qui vérifie s'il y a une collision entre une brique et une balle à partir de leurs coordonnées
   Type : int * int * 'a * 'a -> int * int * bool -> bool
   Paramètre : (balle_x, balle_y, _, _), un quadruplet représentant les coordonnées de la balle (seuls balle_x et balle_y sont utilisés)
   Paramètre : (brique_x, brique_y, collision), un triplet représentant les coordonnées de la brique et un booléen indiquant s'il y a déjà une collision
   Résultat : un booléen indiquant s'il y a une collision entre la brique et la balle
*)
let collision_brique_balle (balle_x, balle_y, _, _) (brique_x, brique_y, collision) =
  if not collision then
    false
  else
    let brique_gauche = brique_x - largeur_briques / 2 in
    let brique_droite = brique_x + largeur_briques / 2 in
    let brique_haut = brique_y - hauteur_briques / 2 in
    let brique_bas = brique_y + hauteur_briques / 2 in
    balle_x +ball_radius >= brique_gauche && balle_x -ball_radius <= brique_droite 
    && balle_y +ball_radius >= brique_haut && balle_y - ball_radius <= brique_bas

(* CONTRAT
   Fonction qui met à jour la visibilité d'une brique après une collision
   Type : int * int * bool -> int * int * bool
   Paramètre : brique, un triplet représentant les coordonnées de la brique et sa visibilité actuelle
   Résultat : un triplet représentant les coordonnées de la brique avec sa visibilité mise à jour
*)
let update_visibilte_brique brique =
  match brique with
  | (x, y, true) -> (x, y, false)
  | _ -> brique

  
(* CONTRAT
   Fonction qui supprime une brique en collision avec une balle d'une liste de briques
   Type : int * int * 'a -> ('a list) -> ('a list)
   Paramètre : ball, un quadruplet représentant les coordonnées de la balle (seuls les éléments ball_x et ball_y sont utilisés)
   Paramètre : briques, une liste de briques
   Résultat : une nouvelle liste de briques où la brique en collision avec la balle est supprimée
*)
  let supprimer_brique ball briques =
    let rec aux acc = function
      | [] -> acc
      | brique :: reste ->
          if collision_brique_balle ball brique then
            aux acc reste
          else
            aux (brique :: acc) reste
    in
    aux [] briques


  let creer_block_briques = 
    let largeur_brique_espacement = largeur_briques + espacement_briques in
    let hauteur_brique_espacement = hauteur_briques + espacement_briques in
    let total_width = briques_par_lignes * largeur_brique_espacement - espacement_briques in
    let total_height = lignes_briques * hauteur_brique_espacement - espacement_briques in

    let x_initial = (window_width - total_width) / 2 in
    let y_initial = window_height - total_height - 50 in

    let rec creer_ligne x y lignes_briques_restantes acc =
      if lignes_briques_restantes <= 0 then
        acc
      else
        let nouvelle_brique = creation_brique x y in
        let nouveau_x = x + largeur_brique_espacement in
        creer_ligne nouveau_x y (lignes_briques_restantes - 1) (nouvelle_brique :: acc)
    in

    let rec creer_plusieurs_lignes y lignes_briques_restantes acc =
      if lignes_briques_restantes <= 0 then
        acc
      else
        let ligne = creer_ligne x_initial y briques_par_lignes [] in
        let nouveau_y = y + hauteur_brique_espacement in
        creer_plusieurs_lignes nouveau_y (lignes_briques_restantes - 1) (ligne @ acc)
    in

    creer_plusieurs_lignes y_initial lignes_briques []


(* CONTRAT
   Fonction qui détecte les collisions entre une balle et une liste de briques
   Type : int * int * 'a -> ('a list) -> bool
   Paramètre : ball, un quadruplet représentant les coordonnées de la balle (seuls les éléments ball_x et ball_y sont utilisés)
   Paramètre : briques, une liste de briques
   Résultat : un booléen indiquant s'il y a une collision entre la balle et au moins une brique de la liste
*)
let detecter_collisions ball briques =
  List.fold_left
    (fun acc brique -> acc || collision_brique_balle ball brique)
    false briques

(* CONTRAT
   Fonction qui met à jour la liste de briques après une collision avec une balle
   Type : int * int * 'a -> ('a list) -> ('a list)
   Paramètre : ball, un quadruplet représentant les coordonnées de la balle (seuls les éléments ball_x et ball_y sont utilisés)
   Paramètre : briques, une liste de briques
   Résultat : une nouvelle liste de briques où la brique en collision avec la balle est supprimée
*)
let mettre_a_jour_briques ball briques =
  supprimer_brique ball briques