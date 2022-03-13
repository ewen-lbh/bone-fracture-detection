## Titre

- détection fracture osseuse à partir d'imagerie médicale

## Pk

- aider pompiers/secours lors d'événements à savoir rapidement si fracture
- type de fracture
- => fluidifier procédé d'hospitalisation, réduire attente car on envoie dans le bon service d'urgence direct

## Deux approches

- sans ML
  - ML a des desavantages
    - bcp de données 
    - biais
- avec ML
  - si nécéssaire

## Sets de données

- sur internet, car plus d'imperfections => convient mieux pour conditions & qualité de scans non idéales

## Détection des bords

- avec cv2.Canny
<!-- - _TODO? principe de fonctionnement de Canny?_ -->

## Détection des bords: pb de texture

- os poreux => artéfacts si trop sensible

## 

- faut trouver les bons seuils

## seuils(image)

- selon image, seuil optimal différent

## seuils(lumi, cont)?

- est-ce que on peut déduire seuils depuis prop de l'image?

## slide avec les graphes décorrelés

- sur quelques images testées, pas _encore_ de correl. évidente

## autres approches

- statistique: une solution serait d'utiliser différents seuils sur même image, et de prendre le résultat majoritaire
- heuristique:
  - par ex, nombre segments détectés / proportion de pixels blancs
      - artéfacts donne segments trops petits pour être considéré comme tel
  <!-- - (autres auxquelles j'ai pas encore pensé) -->

## détection segments

## vectorisation

- avec potrace
- donne fichiers SVG => stocke image comme instructions de dessins vectoriel

## pb de la vectorisation

- en modifiant la balise contenant le chemin de tracé pour voir les contours
- on se rend compte que ce ne sont pas des lignes
- => difficile pour en déduire des angles

## trasnformée de hough

<!-- - _TODO: principe des transformées de hough?_ -->
- deux approches
  - classique:
    - droites
    - difficile de gérer les courbes car pas de taille de segment
    - on obtient point de départ & pente
  - probabiliste:
    - segments
    - on lui donne une taille de segment maximum
    - on obtient coords point de départ et d'arrivée

- j'ai préféré la probabiliste

## calcul d'angles

- ensuite angles

## calcul d'angles: avec de la trigonométrie

- rapports trigonométriques dans un triangle rectangle

## crit décision

- vient le moment d'utiliser données pour déterminer état de l'os

## crit décision: <formule>

- on choisit un seuil ε, si le segment le plus horizontal est trop horiz => c'est cassé

## compensations

- images pas parfaites

## inclinaison: pb

- certaines penchées
    - il pensera que c'est cassé à cause de l'inclinaison

## inclinaison: fixed



## id typ fracture

## lignes de fractures

- diff noms pour diff fractures

## nom fracture (θ, ...)

- lignes fractures <=> angle & point de départ
