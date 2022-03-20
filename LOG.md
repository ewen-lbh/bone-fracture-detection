# Logs

Ordre: du plus ancien au plus récent.

- Installation d'un bibliothèque pour lire des fichiers DICOM: problèmes de compilation avec _hdf5_, bibliothèque utilisée par _pydicom_
- Utilisation d'une application standalone pour convertir les dicom en png: je me rend compte que le seul dataset disponible gratuitement et librement n'est pas intéréssant.
- Recherche d'images de radios sur internet, directement (google images)
- Edge detection avec `cv2.Canny`, le meilleur résultat semble être avec les paramètres `high=130, low=90`, mais il reste des artéfacts dus à la porosité du matériau de l'os. Comparaison avec une image "soignée" (ie sans la fracture, via Photoshop)
- autocrop par edge detection
- essai en floutant l'image pour compenser la porosité
- à faire: générer des batchs avec une meilleure résoution, et tester avec différentes intensités de blur

## Avantages
- marche sur des petites images
- peut être utilisé dans les camions pompiers/ar des particuliers pour rapidement prendre une décision avec un petit objet

- idée d'entrainer un réseau avec du reinforcement learning pour choisir les thresholds de détection de bords, avec pour critère de bon choix une luminosité d'image de sortie (i.e. celle ne contenant que les bords) de 15±2

- premier essai en suivant https://keras.io/examples/rl/actor_critic_cartpole/, mais avec
ma propre classe d'environnement pour répondre à mon besoin. Le problème est que je ne sais pas trop ce que "state" est sensé représenter, je donne le tensor de l'image des bords mais 
ça n'a pas l'air d'être ce qui est attendu (problème de dimensions lors de l'appel dans lequel j'appelle `np.squeeze`.)


- [fonctionnement Canny](https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html) on pourrait avoir des bords moins déconnectés en autorisant les pixels adjaçents à rester blanc (4. Non-maximum suppression)
    je pense pas que Hough est besoin de bords de 1 pixel
