## Titre

- projet

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

## DICOM

- difficle de trouver des bonnes banques
- celle que j'ai trouvé ct pas ce que je cherchais

## Images normales

- sur internet

## Détection des bords

- avec cv2.Canny
- _TODO? principe de fonctionnement de Canny?_

## Détection des bords: pb de texture

- os poreux => artéfacts si trop sensible

## 

- faut trouver les bons seuils

## seuils(image)

_TODO: différentes images au même seuil_

- selon image, seuil optimal différent

## seuils(lumi, cont)?

- est-ce que on peut déduire seuils depuis prop de l'image?

## _TODO: slide avec les graphes décorrelés :/_

##

## vectorisation

- d'abord tenté
- avec potrace
- donne fichiers SVG => stocke image comme instructions de dessins vectoriel

## pb de la vectorisation

- en modifiant la balise contenant le chemin de tracé pour voir les contours
- on se rend compte que ce ne sont pas des lignes
- => difficile pour en déduire des angles

## trasnformée de hough

- _TODO: principe des transformées de hough?_
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

- lignes fractures en bij avec angle & point de départ, et donc transitivement en bij avec nom de fracture

## github

- repo publique dispo ici
