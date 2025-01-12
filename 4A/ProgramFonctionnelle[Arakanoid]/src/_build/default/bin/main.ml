open Graphics
open Unix
open Arkanoid.Balle
open Arkanoid.Raquette
open Arkanoid.Fenetre
open Arkanoid.Parametres
open Arkanoid.Brique
open Arkanoid.Joueur
open Arkanoid.Game_over


(* CONTRAT
   Fonction principale du jeu
   Type : int -> int -> unit
   Paramètres :
     - score : int, le score actuel du joueur
     - niveau : int, le niveau actuel du jeu
   Résultat : la fonction ne renvoie rien (type unit)
*)
let rec main score niveau =
  Unix.sleepf 0.2;
  init_window (); 
  clear_screen ();
  let default_font_size = text_size "Game Over" |> snd in
  let racquet = create_racquet (racquet_width/2, racquet_y - racquet_height, racquet_width, racquet_height) in
  let ball = create_ball racquet in 
  let briques = creer_block_briques in
  let joueur = create_joueur (score, niveau) in
  let mouse_clicked = false in
  

  (* Boucle principale du jeu *)
  let rec game_loop prev_time ball mouse_clicked racquet briques joueur =
    (* Calcul du temps écoulé *)
    let current_time = gettimeofday () in

    (* Condition de sortie pour quitter le jeu en appuyant sur 'e' ou 'E' (exit) *)
    if key_pressed () then
      let ev = read_key () in
      if ev = 'e' || ev = 'E' then
        close_graph ()
      else
        game_loop prev_time ball mouse_clicked racquet briques joueur
    else (
      Unix.sleepf 0.0125;
      clear_screen ();

      (* Déplacement de la raquette avec la souris *)
      let new_racquet = move_racquet racquet in

      (* Mise à jour de la position de la balle *)
      let mouse_clicked' = button_down () || mouse_clicked in
      let new_ball, new_joueur, new_mouse_clicked =
        if mouse_clicked' then
          let collision_with_racquet = check_collision_racquet racquet ball in
          let collision_with_sol = collision_sol ball in
          let (x,y,dx,dy) = ball in
          let collision_with_brick = List.exists (collision_brique_balle (x+dx, y+dy, dx, dy)) briques in
          if collision_with_racquet then
            let (bx, by, dx, dy) = ball in
            let new_ball_pos = update_ball_position (bx, by, dx, -dy) briques current_time in
            new_ball_pos, joueur, mouse_clicked'
          else if collision_with_sol then (
            let new_ball_pos = update_ball_position_before_click new_racquet in
            let joueur_vie_moins_un = decrementer_vie joueur in
            new_ball_pos, joueur_vie_moins_un, false)
          else if collision_with_brick then (
            let new_ball_pos = update_ball_position ball briques current_time in
            let joueur_score_plus_un = incrementer_score joueur in
            new_ball_pos, joueur_score_plus_un, mouse_clicked')
          else 
            let new_ball_pos = update_ball_position ball briques current_time in
            new_ball_pos, joueur, mouse_clicked'
        else
          let new_ball_pos = update_ball_position_before_click new_racquet in
          new_ball_pos, joueur, mouse_clicked
      in

      (* Vérifier les collisions *)
      let collisions = detecter_collisions new_ball briques in

      (* Mise à jour des briques après collision *)
      let briques_mises_a_jour = mettre_a_jour_briques new_ball briques in

      (* Affichage de la raquette et de la balle *)
      List.iter dessiner_brique briques_mises_a_jour;
      draw_racquet new_racquet;
      draw_ball new_ball;
      draw_score new_joueur;
      draw_vie new_joueur;
      draw_niveau joueur;

      (* S'il n'y a plus de briques, on passe au niveau suivant *)
      if List.length briques_mises_a_jour = 0 then 
        main (score + gain_de_score_pour_niveau_termine) (niveau + 1)

      (* Si le joueur est mort *)
      else if joueur_mort new_joueur then (
      let (_, score, niveau) = new_joueur in
      let play_again = draw_game_over score in
      if play_again then(
        set_font (Printf.sprintf "-*-fixed-medium-r-semicondensed--%d-*-*-*-*-*-*-*" default_font_size);
        main score niveau
      )
      else
        close_graph ())
      else(

      synchronize ();
      if collisions then
        game_loop prev_time new_ball new_mouse_clicked new_racquet briques_mises_a_jour new_joueur 
      else
        game_loop current_time new_ball new_mouse_clicked new_racquet briques_mises_a_jour new_joueur 
      )
    )
  in

  let start_time = gettimeofday () in
  game_loop start_time ball mouse_clicked racquet briques joueur ;;


(* CONTRAT
   Fonction qui lance le début du jeu en initialisant la fenêtre et en affichant le menu
   Type : unit -> unit
   Paramètres : la fonction ne prend aucun paramètre
   Résultat : la fonction ne renvoie rien (type unit)
*)
let debut_jeu () = 
  init_window (); 
  clear_screen ();
  let jouer = Arkanoid.Menu.demarrer_jeu () in
  if jouer then
    main score_initial niveau_initial
  else 
    close_graph ();;
  

(* CONTRAT
   Fonction d'exécution principale du jeu
   Type : unit -> unit
   Paramètres : la fonction ne prend aucun paramètre
   Résultat : la fonction ne renvoie rien (type unit)
*)
let () = debut_jeu ()