# -*- coding: utf-8 -*-
"""
Script de copie de board Monday.com avec toutes les fonctionnalités implémentées :
- Mapping element → insured_name
- Calcul automatique des colonnes de formules (Com, Boni, Sur-Com)
- Création séquentielle des groupes (un par un)
- Insertion correcte dans les groupes respectifs
"""

import sys
from main import (
    InsuranceCommissionPipeline,
    create_monday_legacy_config
)

# =============================================================================
# CONFIGURATION - Modifiez ces valeurs selon vos besoins
# =============================================================================

# API Key Monday.com
MONDAY_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJ0aWQiOjU3OTYxMDI3NiwiYWFpIjoxMSwidWlkIjo5NTA2NjUzNywiaWFkIjoiMjAyNS0xMC0yOFQxNToxMDo0My40NjZaIiwicGVyIjoibWU6d3JpdGUiLCJhY3RpZCI6MjY0NjQxNDIsInJnbiI6InVzZTEifQ.q54YnC23stSJfLRnd0E9p9e4ZF8lRUK1TLgQM-13kdI"

# Board source (ID du board à copier)
SOURCE_BOARD_ID = 18283488594  # "Copie de Paiement Historique"

# Nom du nouveau board
TARGET_BOARD_NAME = "Nouveau Board Copie"

# Options
COPY_ALL_GROUPS = True  # True = copier tous les groupes, False = copier un seul groupe
SOURCE_GROUP_ID = None  # Si COPY_ALL_GROUPS = False, spécifiez l'ID du groupe ici


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def copy_board():
    """
    Copie un board Monday.com avec toutes les fonctionnalités :
    - Extraction de la colonne "Élément" → insured_name
    - Calcul automatique de Com, Boni, Sur-Com à partir de PA
    - Création séquentielle des groupes
    - Upload des items dans les bons groupes
    """

    print("="*80)
    print("COPIE DE BOARD MONDAY.COM")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"  Source Board ID:    {SOURCE_BOARD_ID}")
    print(f"  Target Board Name:  {TARGET_BOARD_NAME}")
    print(f"  Copy All Groups:    {COPY_ALL_GROUPS}")
    if not COPY_ALL_GROUPS:
        print(f"  Source Group ID:    {SOURCE_GROUP_ID}")
    print(f"\n✨ Fonctionnalités activées:")
    print(f"  ✓ Mapping 'Élément' → 'insured_name'")
    print(f"  ✓ Calcul automatique de Com, Boni, Sur-Com")
    print(f"  ✓ Création séquentielle des groupes")
    print(f"  ✓ Insertion correcte dans les groupes")
    print("="*80)

    # Créer la configuration
    config = create_monday_legacy_config(
        api_key=MONDAY_API_KEY,
        source_board_id=SOURCE_BOARD_ID,
        target_board_name=TARGET_BOARD_NAME,
        source_group_id=SOURCE_GROUP_ID if not COPY_ALL_GROUPS else None,
        month_group=None  # None = préserver la structure des groupes
    )

    # Lancer le pipeline
    print("\n🚀 Démarrage de la copie...\n")

    pipeline = InsuranceCommissionPipeline(config)
    success = pipeline.run()

    # Résultat
    if success:
        print("\n" + "="*80)
        print("✅ COPIE TERMINÉE AVEC SUCCÈS")
        print("="*80)
        print(f"\n📋 Board créé: {TARGET_BOARD_NAME}")
        print(f"\n✓ Vérifications à faire dans Monday.com:")
        print(f"  1. Les groupes existent et sont dans le bon ordre")
        print(f"  2. Les items sont dans leurs groupes respectifs (pas tous dans le même)")
        print(f"  3. La colonne 'insured_name' contient les noms (ancienne colonne Élément)")
        print(f"  4. Les colonnes Com, Boni, Sur-Com ont des valeurs calculées")
        print(f"  5. Toutes les autres colonnes sont correctement copiées")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("❌ ÉCHEC DE LA COPIE")
        print("="*80)
        print(f"\nVérifiez les erreurs ci-dessus pour diagnostiquer le problème.")
        return 1


# =============================================================================
# EXEMPLES D'UTILISATION
# =============================================================================

def example_copy_all_groups():
    """
    Exemple 1: Copier le board complet avec tous les groupes
    """
    global SOURCE_BOARD_ID, TARGET_BOARD_NAME, COPY_ALL_GROUPS, SOURCE_GROUP_ID

    SOURCE_BOARD_ID = 18283488594
    TARGET_BOARD_NAME = "Copie Complète - Tous les Groupes"
    COPY_ALL_GROUPS = True
    SOURCE_GROUP_ID = None

    return copy_board()


def example_copy_single_group():
    """
    Exemple 2: Copier seulement un groupe spécifique
    """
    global SOURCE_BOARD_ID, TARGET_BOARD_NAME, COPY_ALL_GROUPS, SOURCE_GROUP_ID

    SOURCE_BOARD_ID = 18283488594
    TARGET_BOARD_NAME = "Copie Partielle - Un Groupe"
    COPY_ALL_GROUPS = False
    SOURCE_GROUP_ID = "group_mkw99a2k"  # ID du groupe "Octobre 2025"

    return copy_board()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Fonction principale - choisissez l'exemple à exécuter
    """

    # Option 1: Utiliser la configuration par défaut (en haut du fichier)
    return copy_board()

    # Option 2: Utiliser un exemple prédéfini
    # return example_copy_all_groups()
    # return example_copy_single_group()


if __name__ == "__main__":
    sys.exit(main())
