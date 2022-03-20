## Titre

- fluidifier traffic dans services hospitaliers
  
  - evenements

- classifiant fractures

## portative

- avec des appareils embarques

## principe général

- cibler zone de douleur

- detecter bords

- lignes de fractures

- angles

- critère, classification

## Canny

- algo à double seuil:
  
  - en dessous du min: pas un bord
  
  - au dessus: forcément un bord
  
  - entre: dépend des pixels adjacents

## Hough

## Calcul angles

## Critère décision

## Identification type

## Problème de texture

## En floutant?

## recherche seuils optimaux

- on peut pas contourner pb

- il faut trouver les bons seuils pour chaque image

## seuil(lum, cont) ?

## ...non

## machine learning

- requiert beaucoup de données

## sets de données

- 300 sur radiopædia.org

- trop de choses à traiter manuellement

- pas assez pour le pb direct

- traiter un sous-problème plus simple

## machine learning #2

- reseau neuronal ≡ couches de neurones interconnectées
  
  - neurone ≡ nombre calculé avec paramètres
  
  - connexion ≡ accède à valeur du neurone précédent

## premiere couche

- entrée, on rentre les valeurs utile au problème

## feed forward

- on calcule les valeurs de toutes les neurones

## derniere couche

- valeur des dernières neurones ≡ résultats

## erreur

- écart aux valeurs attendues

## l'objectif

- on a traduit le pb en pb de minimisation d'une fonction

- se fait en changeant les params de calcul des neurones

- comment les changer pour réduire le coût?

## rétropropagation

- derivée de ce paramètre selon le coût => comment le coût change si on change ce paramètre

- difficile voire impossible à calculer

## retropropag regle chaine

- regle de la chaine

## retropropag recurrence

- regle de la chaine jusquà arriver au poids que l'on veut changer

## retropropag update poids

- p ≡ poids à calculer

- η ≡ vitesse d'entraînement
  
  - si trop rapide on peut passer à côter du min et osciller
  
  - si trop lent temps d'entraînement prohib

## le problème

- ici coût ≡ luminosité 
