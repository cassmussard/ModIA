open Graphics
open Parametres

module Game_over = struct
  (* CONTRAT
    Fonction qui dessine un bouton sur l'écran
    Type : int -> int -> int -> int -> string -> unit
    Paramètre : x, la position horizontale du coin supérieur gauche du bouton
    Paramètre : y, la position verticale du coin supérieur gauche du bouton
    Paramètre : width, la largeur du bouton
    Paramètre : height, la hauteur du bouton
    Paramètre : label, le texte du bouton à afficher
    Résultat : la fonction ne renvoie rien (type unit)
  *)
  let draw_button x y width height label =
    set_color black;
    fill_rect x y width height;
    set_color white;
    draw_rect x y width height;
    let label_width, label_height = text_size label in
    let label_x = x + (width - label_width) / 2 in
    let label_y = y + (height - label_height) / 2 in
    moveto label_x label_y;
    draw_string label

  (* CONTRAT
    Fonction qui affiche l'écran "Game Over" avec les boutons Rejouer et Quitter
    Type : int -> bool
    Paramètre : score, le score du joueur
    Résultat : la fonction renvoie un booléen (true pour le bouton Rejouer, false pour le bouton Quitter)
  *)
  let draw_game_over score =
    set_color red;
    let text = "Game Over - Score: " ^ string_of_int score in
    let text_width, text_height = text_size text in
    let scaled_text_width = taille_game_over * text_width in
    let scaled_text_height = taille_game_over * text_height in
    let x = window_width / 2 - scaled_text_width / 2 in
    let y = window_height / 2 - scaled_text_height / 2 in
    let font_size = scaled_text_height / 2 in
    let font_string = Printf.sprintf "-*-fixed-medium-r-semicondensed--%d-*-*-*-*-*-*-*" font_size in
    set_font font_string;
    moveto x y;
    draw_string text;
    let button_width = 120 in
    let button_height = 40 in
    let button_x = window_width / 2 - button_width / 2 in
    let button_y = y - button_height - 20 in
    draw_button button_x button_y button_width button_height "Rejouer";
    let quit_button_x = window_width / 2 - button_width / 2 in
    let quit_button_y = y - 2 * button_height - 40 in
    draw_button quit_button_x quit_button_y button_width button_height "Quitter";
    synchronize ();
    let rec wait_click () =
      let ev = wait_next_event [Button_down] in
      let mx = ev.mouse_x and my = ev.mouse_y in
      if mx >= button_x && mx <= button_x + button_width && my >= button_y && my <= button_y + button_height then
        (* Rejouer *)
        true
      else if mx >= quit_button_x && mx <= quit_button_x + button_width && my >= quit_button_y && my <= quit_button_y + button_height then
        (* Quitter le jeu *)
        false
      else
        wait_click ()
    in
    wait_click ()
end

include Game_over
