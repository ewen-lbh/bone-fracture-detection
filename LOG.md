# Logs

Ordre: du plus ancien au plus récent.

- Installation d'un bibliothèque pour lire des fichiers DICOM: problèmes de compilation avec _hdf5_, bibliothèque utilisée par _pydicom_

- Utilisation d'une application standalone pour convertir les dicom en png: je me rend compte que le seul dataset disponible gratuitement et librement n'est pas intéréssant.

- Recherche d'images de radios sur internet, directement (google images)

- Edge detection avec `cv2.Canny`, le meilleur résultat semble être avec les paramètres `high=130, low=90`, mais il reste des artéfacts dus à la porosité du matériau de l'os. Comparaison avec une image "soignée" (ie sans la fracture, via Photoshop)

- autocrop par edge detection

- essai en floutant l'image pour compenser la porosité

- à faire: générer des batchs avec une meilleure résoution, et tester avec différentes intensités de blur

- avantages
  
  - marche sur des petites images
  
  - peut être utilisé dans les camions pompiers/ar des particuliers pour rapidement prendre une décision avec un petit objet

- [fonctionnement Canny](https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html) on pourrait avoir des bords moins déconnectés en autorisant les pixels adjaçents à rester blanc (4. Non-maximum suppression)
    je pense pas que Hough est besoin de bords de 1 pixel

## Reinforcement learning

### Premier essai

- idée d'entrainer un réseau avec du reinforcement learning pour choisir les thresholds de détection de bords, avec pour critère de bon choix une luminosité d'image de sortie (i.e. celle ne contenant que les bords) de 15±2

- premier essai en suivant https://keras.io/examples/rl/actor_critic_cartpole/, mais avec
  ma propre classe d'environnement pour répondre à mon besoin. Le problème est que je ne sais pas trop ce que "state" est sensé représenter, je donne le tensor de l'image des bords mais 
  ça n'a pas l'air d'être ce qui est attendu (problème de dimensions lors de l'appel dans lequel j'appelle `np.squeeze`.)

### Deuxième essai

https://en.wikipedia.org/wiki/Reinforcement_learning

- situation pour mon problème:
  
  - ≈3200 actions: contraste, luminosité, seuils: augmenter ou baisser de incrément dans [0, 100] samplé par steps de 0.25
  
  - état: pixels de l'image des bords actuelle
  
  - reward: différence à une range de proportions optimales de pixels blancs

- pour le machine learning, pas d'accès à un set de données labelled -> unsupervised learning -> RL

- pb avec le processus de décision markovien seul (modèle simple de RL): passage de l'état _s_ par l'action _a_ à l'état _s'_ **ne dépend pas des états et actions antérieurs** => pas assez complexe je pense

- attends la notion de "regret" c'est bien mais il faut comparer à la pref de "l'agent optimal" — mais on l'a pas sinon le pb serait déjà résolu!

- γ donne le discount - pour prioriser les reward d'actions immédiates à celles du futur lointain dans l'évaluation du résultat asymptotique (https://en.wikipedia.org/wiki/Reinforcement_learning#State-value_function)

- dans mon cas: ≈3200 actions: contraste, luminosité, seuils: augmenter ou baisser de incrément dans [0, 100] samplé par steps de 0.25 => brute force: p-ê pas très performant
  
  > These problems can be ameliorated if we assume some structure and allow samples generated from one policy to influence the estimates made for others. 
  
    Je vois pas trop de lien dans mon cas malheureusement

- Monte Carlo: admet que le problème est "épisodique": il y a un certain nombre d'états maximum. mon problème l'est peut-être (on met arbitrairement une limite), mais juste pour une limite, pas un nombre fixe (imagine il tombe sur la bonne solution après _k_ états mais j'ai donné _k + 8_ états et dans les 8 derniers il diverge (je sais pas si c'est possible)) ([machine learning - Monte Carlo for non-episodic tasks - Data Science Stack Exchange](https://datascience.stackexchange.com/questions/77787/monte-carlo-for-non-episodic-tasks))

- temporal difference: m'a l'air pas mal, une bonne explication de la différence avec Monte Carlo: https://towardsdatascience.com/introduction-to-reinforcement-learning-rl-part-6-temporal-difference-td-learning-2a12f0aba9f9#153e
