open Parametres
open Graphics

(* Initialisation de la fenêtre *)
(*CONTRAT
Fonction qui initialise la fenêtre graphique en définissant sa taille
Résultats : ouvre la fenêtre graphique avec la taille spécifiée et active la synchronisation automatique
*)
let init_window () =
  open_graph (Printf.sprintf " %dx%d" window_width window_height);
  auto_synchronize true

(* Fonction pour que la fenêtre de jeu soit vide *)
(*CONTRAT
Fonction qui efface l'écran de la fenêtre graphique
Résultats : efface le contenu affiché sur l'écran de la fenêtre graphique
*)
let clear_screen () =
  clear_graph ()