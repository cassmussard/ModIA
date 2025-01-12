open Arkanoid.Parametres
open Arkanoid.Brique
open Arkanoid.Balle

(* TESTS *)
let%test "Test create_ball" =
  let racquet_x = 100 in
  let racquet_width = 60 in
  let racquet_y = 200 in
  let racquet_height = 10 in
  
  let result = create_ball (racquet_x, racquet_width, racquet_y, racquet_height) in
  
  (* Vérification des conditions *)
  let expected_x = racquet_x + racquet_width / 2 in
  let expected_y = racquet_y + racquet_height in
  let expected_dx, expected_dy = initial_ball_speed in
  
  result = (expected_x, expected_y, expected_dx, expected_dy)

let%test "Test draw_ball" =
  init_window (); 
  
  let x = 100 in
  let y = 200 in
  
  draw_ball (x, y, 0, 0);
  
  (* Vérification des conditions *)
  let is_filled = is_point_filled (x, y) in
  is_filled



let%test "Test update_ball_position_before_click" =
  let racquet_x = 100 in
  let racquet_width = 60 in
  let racquet_y = 200 in
  let racquet_height = 10 in
  
  let result = update_ball_position_before_click (racquet_x, racquet_width, racquet_y, racquet_height) in
  
  (* Vérification des conditions *)
  let expected_x = racquet_x + racquet_width / 2 in
  let expected_y = racquet_y + racquet_height in
  let expected_dx, expected_dy = initial_ball_speed in
  
  result = (expected_x, expected_y, expected_dx, expected_dy)


let%test "Test update_ball_position" =
  let x = 100 in
  let y = 200 in
  let dx = 2 in
  let dy = 3 in
  let new_dx = 1 in
  let new_dy = 2 in
  let brique1 = creation_brique 100 100
  let brique2 = creation_brique 200 200
  let brique3 = creation_brique 300 300
  let brique4 = creation_brique 400 400

  let briques = [brique1; brique2; brique3; brique4] in
  let time = 0.1 in
  
  let result = update_ball_position (x, y, new_dx, new_dy) briques time in
  
  (* Vérification des conditions *)
  let expected_new_x = x + dx in
  let expected_new_y = y + dy in
  let expected_dx = dx - int_of_float (float_of_int dx *. acceleration_factor *. time) in
  let expected_dy = dy + int_of_float (float_of_int dy *. acceleration_factor *. time) - int_of_float gravity_factor in
  
  result = (expected_new_x, expected_new_y, expected_dx, expected_dy)


let%test "Test check_collision_racquet" =
  let rx = 100 in
  let ry = 200 in
  let rw = 60 in
  let rh = 10 in
  
  let bx = 120 in
  let by = 205 in
  
  let result = check_collision_racquet (rx, ry, rw, rh) (bx, by, 0, 0) in
  
  (* Vérification des conditions *)
  result = true

let%test "Test collision_sol" =
  let balle = (100, ball_radius, 0, 0) in
  
  let result = collision_sol balle in
  
  (* Vérification des conditions *)
  result = true
