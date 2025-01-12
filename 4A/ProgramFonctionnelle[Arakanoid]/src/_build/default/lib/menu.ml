open Graphics

(* Fonction qui calcule les coordonnées x et y pour centrer le texte *)
(*CONTRAT
Fonction qui affiche un texte centré horizontalement à une position donnée en y
Paramètres :
text : le texte à afficher
y : la position en y où afficher le texte
Resultat : affiche le texte centré horizontalement à la position spécifiée
*)
let draw_text_centered text y =
  let (text_width, _) = text_size text in
  let x = (800 - text_width) / 2 in
  moveto x y;
  draw_string text






(* Fonction qui affiche le menu du jeu *)
(* CONTRAT
Fonction qui affiche le menu du jeu Arkanoid avec le titre et les options de jeu.
Resultat : ouvre une fenêtre graphique, remplit le fond en noir, affiche le titre en rouge et les options en bleu.
*)

let afficher_menu () =
  open_graph "800x600";
  set_color (rgb 0 0 0); (* Couleur du fond : Noir *)
  fill_rect 0 0 800 600;

  (* Afficher le titre en rouge *)
  set_color (rgb 255 0 0);
  let titre = "Arkanoid" in
  set_font "-*-fixed-medium-r-normal--20-*-*-*-*-*-iso8859-1";  (* Taille du texte du titre *)
  draw_text_centered titre 450;

  (* Afficher les options en bleu *)
  set_color (rgb 0 0 255);
  set_font "-*-fixed-medium-r-normal--20-*-*-*-*-*-iso8859-1";  (* Taille du texte des options *)
  let option1 = "Appuyez sur 'J' pour jouer" in
  draw_text_centered option1 350;

  let option2 = "Appuyez sur 'E' pour quitter" in
  draw_text_centered option2 250;

  synchronize ()

(* Fonction qui lance le jeu en affichant le menu et en attendant le choix de l'utilisateur *)
let demarrer_jeu () : bool =
  afficher_menu ();
  let rec attendre_choix () =
    if key_pressed () then (
      let ev = read_key () in
      if ev = 'j' || ev = 'J' then
        true
      else if ev = 'e' || ev = 'E' then 
        false
      else
        attendre_choix ()
    )
    else
      attendre_choix ()
  in
  attendre_choix ()
;;