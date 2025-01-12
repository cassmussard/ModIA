open Arkanoid.Parametres
open Arkanoid.Raquette

  (* TESTS *)
let%test "Test create_racquet" =
  let expected_result = (100, 200, 80, 20) in

  (* Appel de la fonction à tester *)
  let result = create_racquet (100, 200, 80, 20) in

  (* Vérification des conditions *)
  result = expected_result


let%test "Test move_racquet" =
  let raquette = (100, 200, 80, 20) in
  let expected_result = (150, 200, 80, 20) in

  (* Simulation de la position de la souris *)
  let mouse_x = 150 in
  Graphics.set_mouse_pos mouse_x 0;

  (* Appel de la fonction à tester *)
  let result = move_racquet raquette in

  (* Vérification des conditions *)
  result = expected_result



let%test "Test draw_racquet" =
  (* Définition des coordonnées et dimensions de la raquette *)
  let x = 100 in
  let y = 200 in
  let width = 80 in
  let height = 20 in

  (* Appel de la fonction à tester *)
  draw_racquet (x, y, width, height);

  (* Vérification visuelle manuelle *)
  true
