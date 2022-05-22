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

## machine learning classique

- dit supervisé
- compliqué pour mon cas

## reinforcement learning > explication globale

- agent effectue actions sur environement qui donne une récompense
    - ici: environement: image source, augmentation de luminosité et contraste, rayon de flou gaussien
    - ici: agent: thresh++, etc.
    - ici: récompense:
      - distance à la range acceptable de luminosité de l'image des bords
      - si dedans: 0.25 + distance à range acceptable de #segments (sans compter les tout petits, inferieur à 20px en norme)
      - si dedans: 0.5 + distance à range acceptable de nombres d'angle uniques
- pour chaque élément du set de donnée
    - tant que environement décide que l'état actuel n'est pas satisfaisant
        - une step:
            - agent choisi action: aléatoirement ou en choisissant la meilleure (Q-values) (ε)
            - effectue action sur environement
            - environement se modifie en fonction
            - environement répond avec récompense en fonction du nouvel état de l'environement
            - agent répond

## reinforcement learning > apprentissage

- comment l'agent apprend ?
- méthode classique: Q-learning:
  - tableau avec chaque état possible et chaque action possible dans chaque état associé à probabilité que ce soit le meilleur choix
  - initialisé aléatoirement
  - quand choisi aléatoirement on update la q-value en fonction de la proba: (formule de Bel ??)
- ici: taille du tableau: 
    - 3200 actions × 2 · (pixel: [0, 1] samplé à 0.05 => 20) ^ (200 × 200) = ∞ (d'après la commande math de fish) ≈ 10⁵²⁰⁴⁵ (d'après wolfram)

## reinforcement learning > deep q learning

- donc on utilise des réseaux neuronaux pour prédire la meilleure action à la place (deep q-learning)
  - le réseau prédit les actions à effectuer pour maximiser la reward depuis l'état actuel
  - on met à jour les poids du modèle à chaque étape, à chaque action, et la plupart sont aléatoires au début
  - pour l'entraîner, on met à jour la proba que chaque action soit bonne en fonction de la reward, puis on entraîne le réseau sur ces nouvelles probas
    => on met à jour (régression, fit) sur de l'aléatoire
    => poids inutiles (essaie d'apprendre sur de l'aléatoire)
    => surtout au début (car après ε decay)
  - on ne choisit pas les actions de ce modèle mais celles d'un autre qui reçoit les poids de celui-ci toutes les k étapes => plus stable, plus de chance de généraliser

## résultats

- après avoir laissé le réseau s'entraîner pdt plusieurs heures
- choisi flou et augmentation luminosité pour éviter porosité
- mais fracture toujours visible, et détectée
