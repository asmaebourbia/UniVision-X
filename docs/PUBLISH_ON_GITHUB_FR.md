# Publier UniVision-X sur GitHub — guide pratique

## Avant de publier

Le ZIP contient **le code E2/E3**, **les schémas SVG**, **les métriques locales**, **deux résultats qualitatifs choisis (lapin, chien)** et **un exemple d'échec (panda)** dans une section secondaire. Il inclut un README **anglais** complet et un [`README_FR.md`](../README_FR.md) en français.

**Ne remplace pas ton dossier du serveur `~/univision_x`** : il contient les checkpoints et les fichiers de données qui n'ont pas vocation à être publics. Le dossier GitHub a été reconstruit à partir des fichiers sources et des correctifs disponibles dans la conversation. Si tu as modifié certains scripts directement sur le serveur L3i, synchronise ces versions avant d'annoncer une reproduction bit à bit.

## Méthode 1 — Via GitHub et ton ordinateur

1. Télécharge **`UniVision-X_GitHub_Final.zip`** sur ton ordinateur.
2. Décompresse-le : un dossier `UniVision-X/` contient `README.md`, `univision/`, `scripts/`, `assets/`, etc.
3. Crée un dépôt GitHub vide : https://github.com/new . Nom suggéré : **`UniVision-X`**. Ne coche pas l'ajout automatique d'un README, car l'archive en contient déjà un.
4. Pour un petit nombre de fichiers, GitHub → **Add file → Upload files**. Afin de ne pas casser l'arborescence, téléverse **le contenu entier du dossier intérieur**, pas le ZIP lui-même.
5. Vérifie que la page d'accueil affiche le `README.md`, l'architecture SVG, le graphique E2/E3 et les deux images de comparaison.

## Méthode 2 — Git en ligne de commande (recommandée)

Ouvre un terminal dans le **dossier `UniVision-X` décompressé**, après avoir créé un dépôt GitHub vide nommé `UniVision-X` :

```bash
git init
git add .
git status
git commit -m "Initial public research portfolio: UniVision-X E2/E3"
git branch -M main
git remote add origin https://github.com/asmaebourbia/UniVision-X.git
git push -u origin main
```

Cette URL est un **exemple à vérifier** : elle doit correspondre à ton identifiant et au dépôt effectivement créé. Le dépôt et ses droits d'accès ne sont pas automatiquement créés par le ZIP. GitHub peut demander une authentification par navigateur ou un token.

## Vérifications importantes

```bash
# Toujours depuis le dossier UniVision-X décompressé
git status
find . -type f -size +20M -print
```

Ne publie **pas** les dossiers locaux `data/`, `outputs/`, `logs/`, `.venv/`, des clés SSH, des secrets ni des poids `*.pt` / `*.pth`. `.gitignore` aide à les exclure, mais vérifie toujours `git status` avant le commit.

Les **3 images de comparaison** sont dérivées de MeViS ; respecte la licence du dataset et les droits éventuels de ton laboratoire. Le code n'a pas de licence open source attribuée par défaut. Lis [`docs/DATA_AND_LICENSE.md`](DATA_AND_LICENSE.md).

## Après la publication

**Description courte suggérée pour GitHub :**

> Research prototype for text-guided video object segmentation with FastViT, CLIP token-level cross-attention, ConvGRU and target-presence prediction. Evaluated on MeViS v2.

**Topics :** `computer-vision`, `deep-learning`, `pytorch`, `video-segmentation`, `vision-language`, `multimodal-learning`, `fastvit`, `clip`, `cross-attention`, `mevis`.

**Ordre de présentation recommandé :** les résultats quantitatifs complets avant les exemples sélectionnés ; le lapin en premier, le chien en deuxième, le panda dans une section repliable « Failure analysis ». Indiquer clairement que les métriques sont locales et que l'évaluation officielle MeViS n'a pas été exécutée.
