open Graphics
open Arkanoid.Parametres
open Arkanoid.Joueur


  (* TESTS *)
let%test "Test create_joueur" =
  let score = 0 in
  let niveau = 1 in

  let result = create_joueur (score, niveau) in

  (* Vérification des conditions *)
  result = (vies_joueur, score, niveau)


let%test "Test decrementer_vie" =
  let joueur = (3, 100, 1) in
  let expected_result = (2, 100, 1) in

  let result = decrementer_vie joueur in

  (* Vérification des conditions *)
  result = expected_result

let%test "Test joueur_mort" =
  let joueur_vivant = (3, 100, 1) in
  let joueur_mort = (0, 100, 1) in

  let result_vivant = joueur_mort joueur_vivant in
  let result_mort = joueur_mort joueur_mort in

  (* Vérification des conditions *)
  result_vivant = false && result_mort = true


let%test "Test incrementer_score" =
  let joueur = (3, 100, 1) in
  let expected_result = (3, 150, 1) in

  let result = incrementer_score joueur in

  (* Vérification des conditions *)
  result = expected_result
