# Configuration des Variables d'Environnement

Ce guide explique comment configurer les variables d'environnement pour l'application PDF Extractor.

## 1. Configuration Locale (.env file)

### Étape 1 : Créer le fichier .env

Copiez le fichier `.env.example` en `.env` :

```bash
cp .env.example .env
```

### Étape 2 : Modifier le fichier .env

Ouvrez le fichier `.env` et remplacez les valeurs par vos vraies clés :

```env
MONDAY_API_KEY=eyJhbGciOiJIUzI1NiJ9.votre_vraie_clé_ici
```

**IMPORTANT** : Le fichier `.env` est dans `.gitignore` et ne sera JAMAIS commité sur GitHub.

---

## 2. Configuration dans PyCharm

### Option A : Utiliser le fichier .env (Recommandé)

1. Créez le fichier `.env` comme décrit ci-dessus
2. Installez le plugin "EnvFile" dans PyCharm :
   - File > Settings > Plugins
   - Recherchez "EnvFile"
   - Installez et redémarrez PyCharm

3. Configurez votre Run Configuration :
   - Run > Edit Configurations
   - Sélectionnez votre configuration Python
   - Onglet "EnvFile"
   - Cochez "Enable EnvFile"
   - Ajoutez votre fichier `.env`

### Option B : Variables d'environnement directes

1. Run > Edit Configurations
2. Sélectionnez votre configuration Python (ex: `app.py`)
3. Trouvez la section "Environment variables"
4. Cliquez sur l'icône de dossier à droite
5. Cliquez sur "+" pour ajouter une nouvelle variable
6. Nom : `MONDAY_API_KEY`
7. Valeur : `votre_clé_api_ici`
8. Cliquez sur OK

**Avantages** : Pas besoin de plugin
**Inconvénients** : Doit être configuré pour chaque Run Configuration

---

## 3. Configuration pour Streamlit Cloud

### Dans les Secrets de Streamlit Cloud

1. Allez sur https://share.streamlit.io/
2. Ouvrez votre application
3. Cliquez sur "Settings" > "Secrets"
4. Ajoutez vos secrets au format TOML :

```toml
MONDAY_API_KEY = "eyJhbGciOiJIUzI1NiJ9.votre_vraie_clé_ici"
```

5. Cliquez sur "Save"

### Dans votre code Streamlit (app.py)

Le code utilise déjà `st.secrets` pour accéder aux secrets :

```python
import streamlit as st
api_key = st.secrets.get("MONDAY_API_KEY")
```

---

## 4. Configuration Système (Linux/Mac)

### Temporaire (session actuelle uniquement)

```bash
export MONDAY_API_KEY="votre_clé_api_ici"
```

### Permanent (ajouté au profil bash/zsh)

Ajoutez à `~/.bashrc` ou `~/.zshrc` :

```bash
export MONDAY_API_KEY="votre_clé_api_ici"
```

Puis rechargez :

```bash
source ~/.bashrc  # ou source ~/.zshrc
```

---

## 5. Configuration Système (Windows)

### PowerShell (Temporaire)

```powershell
$env:MONDAY_API_KEY="votre_clé_api_ici"
```

### Variables d'environnement Windows (Permanent)

1. Recherchez "Variables d'environnement" dans le menu Démarrer
2. Cliquez sur "Modifier les variables d'environnement système"
3. Cliquez sur "Variables d'environnement"
4. Dans "Variables utilisateur", cliquez sur "Nouvelle"
5. Nom : `MONDAY_API_KEY`
6. Valeur : `votre_clé_api_ici`
7. Cliquez sur OK

---

## 6. Vérifier la Configuration

### Test en Python

```python
import os
from dotenv import load_dotenv

# Charger le fichier .env
load_dotenv()

# Vérifier la variable
api_key = os.getenv("MONDAY_API_KEY")
if api_key:
    print(f"✓ API Key configurée (longueur: {len(api_key)} caractères)")
else:
    print("✗ API Key NON configurée")
```

### Test en ligne de commande

**Linux/Mac :**
```bash
echo $MONDAY_API_KEY
```

**Windows PowerShell :**
```powershell
echo $env:MONDAY_API_KEY
```

---

## 7. Obtenir votre clé API Monday.com

1. Connectez-vous à Monday.com
2. Allez sur https://monday.com/developers/apps
3. Cliquez sur "Personal API Token" en haut à droite
4. Créez un nouveau token ou copiez un token existant
5. **Copiez le token** (vous ne pourrez plus le voir après)
6. Collez-le dans votre fichier `.env`

---

## Sécurité

**À FAIRE :**
- ✓ Utiliser un fichier `.env` pour le développement local
- ✓ Ajouter `.env` au `.gitignore`
- ✓ Utiliser des secrets Streamlit Cloud pour la production
- ✓ Partager uniquement le fichier `.env.example`

**À NE PAS FAIRE :**
- ✗ Commiter le fichier `.env` sur GitHub
- ✗ Hardcoder les clés API dans le code
- ✗ Partager votre fichier `.env` avec d'autres
- ✗ Envoyer vos clés API par email ou chat

---

## Troubleshooting

### Erreur : "MONDAY_API_KEY environment variable not set!"

**Solutions :**

1. Vérifiez que le fichier `.env` existe à la racine du projet
2. Vérifiez que `python-dotenv` est installé : `pip install python-dotenv`
3. Vérifiez que le fichier `.env` contient bien `MONDAY_API_KEY=...`
4. Dans PyCharm, vérifiez que votre Run Configuration charge bien le `.env`
5. Redémarrez votre IDE après avoir créé le fichier `.env`

### La variable n'est pas chargée

1. Vérifiez que vous êtes dans le bon répertoire
2. Vérifiez qu'il n'y a pas d'espaces autour du `=` dans `.env`
3. Vérifiez que la clé n'a pas de guillemets dans le fichier `.env`

Format correct dans `.env` :
```env
MONDAY_API_KEY=eyJhbGciOiJIUzI1NiJ9.votre_clé
```

Format incorrect :
```env
MONDAY_API_KEY = "eyJhbGciOiJIUzI1NiJ9.votre_clé"  # ✗ Espaces et guillemets
```
