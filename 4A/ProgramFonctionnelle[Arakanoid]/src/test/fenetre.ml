open Arkanoid.Parametres
open Graphics
open Arkanoid.Fenetre

(* TESTS *)
let%test "Test init_window" =
  init_window (); 
  (* Vérification des conditions *)
  let window_size = size_of_window () in
  window_size = (window_width, window_height) &&
  is_synchronized ()

let%test "Test clear_screen" =
  (* Préparation de l'état initial *)
  init_window ();
  draw_circle 100 100 50;
  
  clear_screen ();
  
  (* Vérification des conditions *)
  let window_size = size_of_window () in
  let is_empty = is_image_empty (0, 0, window_size) in
  is_empty

