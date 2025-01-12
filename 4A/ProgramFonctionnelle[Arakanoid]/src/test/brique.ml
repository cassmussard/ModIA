open Graphics
open Arkanoid.Parametres
open Arkanoid.Brique

(* TESTS *)
let%test "Test creation_brique" =
  let x = 100 in
  let y = 200 in

  let result = creation_brique x y in

  (* Vérification des conditions *)
  result = (x, y, true)


let%test "Test dessiner_brique" =
  init_window ();
  
  let x = 100 in
  let y = 200 in
  
  dessiner_brique (x, y, ());
  
  (* Vérification des conditions *)
  let is_filled = is_rect_filled (x - largeur_briques / 2, y - hauteur_briques / 2, largeur_briques, hauteur_briques) in
  is_filled


let%test "Test collision_brique_balle" =
  let balle_x = 100 in
  let balle_y = 200 in
  
  let brique_x = 100 in
  let brique_y = 200 in
  let collision = true in
  
  let result = collision_brique_balle (balle_x, balle_y, (), ()) (brique_x, brique_y, collision) in
  
  (* Vérification des conditions *)
  result = true


let%test "Test update_visibilte_brique" =
  let brique = (100, 200, true) in
  
  let result = update_visibilte_brique brique in
  
  (* Vérification des conditions *)
  result = (100, 200, false)



let%test "Test supprimer_brique" =
  let ball = (100, 200, (), ()) in
  
  let brique1 = (100, 200, true) in
  let brique2 = (200, 300, false) in
  let brique3 = (300, 400, true) in
  let briques = [brique1; brique2; brique3] in
  
  let result = supprimer_brique ball briques in
  
  (* Vérification des conditions *)
  result = [brique2]




let%test "Test detecter_collisions" =
  let ball = (100, 200, (), ()) in
  
  let brique1 = (100, 200, true) in
  let brique2 = (200, 300, false) in
  let brique3 = (300, 400, true) in
  let briques = [brique1; brique2; brique3] in
  
  let result = detecter_collisions ball briques in
  
  (* Vérification des conditions *)
  result = true




let%test "Test mettre_a_jour_briques" =
  let ball = (100, 200, (), ()) in
  
  let brique1 = (100, 200, true) in
  let brique2 = (200, 300, false) in
  let brique3 = (300, 400, true) in
  let briques = [brique1; brique2; brique3] in
  
  let result = mettre_a_jour_briques ball briques in
  
  (* Vérification des conditions *)
  result = [brique2]
