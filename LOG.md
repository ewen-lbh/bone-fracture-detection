# Logs

Ordre: du plus ancien au plus récent.

- Installation d'un bibliothèque pour lire des fichiers DICOM: problèmes de compilation avec _hdf5_, bibliothèque utilisée par _pydicom_
- Utilisation d'une application standalone pour convertir les dicom en png: je me rend compte que le seul dataset disponible gratuitement et librement
  n'est pas intéréssant.
- Recherche d'images de radios sur internet, directement (google images)
- Edge detection avec `cv2.Canny`, le meilleur résultat semble être avec les paramètres `high=130, low=90`, mais il reste des artéfacts dus à la porosité du matériau de l'os. Comparaison avec une image "soignée" (ie sans la fracture, via Photoshop)
- autocrop par edge detection
- essai en floutant l'image pour compenser la porosité
