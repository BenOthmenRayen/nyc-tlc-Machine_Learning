# Zones de New York : segmentation et prédiction de la part taxi / VTC (TLC 2024-2025)

Projet académique réalisé à **ESPRIT** dans le cadre du module Machine Learning : **objectif ML6** du projet d'entrepôt de données NYC TLC.

L'objectif 6 combine deux volets complémentaires sur les **263 zones de taxi de New York** :

1. **Volet non supervisé – segmentation des zones.** Regrouper les zones en segments homogènes selon la façon dont on s'y déplace : volume d'activité, part des taxis et des VTC, prix, durée et distance des courses, pourboires, temps d'attente, exposition au péage de congestion du centre de Manhattan (CBD). Chaque segment est nommé à partir de ses variables dominantes.
2. **Volet supervisé – prédiction de la part taxi / VTC.** Prédire, pour une zone et une tranche horaire, la part des courses assurées par les taxis (Yellow, Green) par rapport aux VTC (HVFHV), en utilisant notamment le segment de la zone comme variable explicative.

Les deux volets débouchent sur des **recommandations** par segment (offre de taxis, tarification, effets du péage CBD).

---

## Sommaire

- [Contexte](#contexte)
- [Données](#données)
- [Démarche](#démarche)
- [Règles de nettoyage](#règles-de-nettoyage)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation et exécution](#installation-et-exécution)
- [Avancement](#avancement)
- [Auteur](#auteur)

---

## Contexte

La **Taxi & Limousine Commission (TLC)** de New York publie chaque mois l'ensemble des courses de taxis et de VTC de la ville. Quatre services coexistent :

- **Yellow** : taxis jaunes, hélés dans la rue, surtout à Manhattan et aux aéroports ;
- **Green** : taxis verts (« Boro taxis »), autorisés hors du centre de Manhattan ;
- **FHV** : voitures avec chauffeur réservées auprès d'une base (limousines, black cars) ;
- **HVFHV** : grandes plateformes de VTC (Uber, Lyft…).

Le 5 janvier 2025, New York a introduit un **péage de congestion** pour les véhicules qui entrent dans le centre de Manhattan. Comparer février-mars 2024 à février-mars 2025 permet d'observer la même saison avant et après ce péage.

## Données

- **Source :** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Période :** février et mars 2024, février et mars 2025 (119 jours)
- **Volume :** environ 14,3 millions de courses Yellow, 209 000 Green, 6,4 millions FHV, et plusieurs dizaines de millions HVFHV
- **Référentiel :** `taxi_zone_lookup.csv` (265 codes de zone : 263 zones réelles, 264 = inconnue, 265 = hors NYC)

Les fichiers de données sont trop volumineux pour GitHub et **ne sont pas versionnés**. Ils doivent être téléchargés depuis le site de la TLC et placés localement (voir [Installation](#installation-et-exécution)).

## Démarche

1. **Extraction** : lecture des CSV bruts par blocs (avec reprise automatique en cas de coupure du Google Drive) et conservation des seules colonnes utiles à l'objectif.
2. **Renommage** : noms de colonnes courts et identiques pour les 4 services (`date_debut`, `zone_depart`, `distance`, `prix`, `pourboire`, `frais_cbd`…).
3. **Nettoyage table par table** : un notebook par service, une étape par cellule, avec une note expliquant chaque décision.
4. **Agrégation par zone** : chaque table nettoyée est résumée par zone de départ (nombre de courses, sommes, moyennes).
5. **Intégration** : les 4 résumés sont joints avec le référentiel des zones pour obtenir **une ligne par zone**.
6. **Feature engineering** : courses par jour, parts taxi / VTC, prix moyen et prix par mile, durée et distance moyennes, taux de pourboire, part de courses payant le péage CBD, temps d'attente VTC, croissance 2024 → 2025…
7. **Préparation** : imputation, transformation logarithmique, écrêtage et standardisation.
8. **Clustering** : K-Means (choix de k par silhouette, Davies-Bouldin, Calinski-Harabasz et méthode du coude avec KneeLocator), Classification Ascendante Hiérarchique (Ward) et DBSCAN.
9. **Interprétation** : comparaison des méthodes, profils des segments et nommage.
10. **Volet supervisé** : construction d'une table zone × tranche horaire, cible = part des courses taxi face aux VTC, entraînement et évaluation de modèles de régression, analyse des variables les plus influentes.
11. **Recommandations** : actions proposées pour chaque segment à partir des deux volets.

## Règles de nettoyage

Le principe est simple : **une course n'est supprimée que si elle ne peut pas être rattachée à une zone ou à la période**. Si seule une mesure est impossible, on vide cette valeur et on garde la course, pour qu'elle compte toujours dans l'activité de sa zone sans fausser les moyennes.

**Courses supprimées**

- doublons exacts ;
- date de début ou de fin manquante ;
- date de début hors des 4 mois étudiés ;
- zone de départ manquante ou hors de l'intervalle 1 à 263.

**Valeurs vidées (course conservée)**

- durée hors de 1 à 180 minutes ;
- distance nulle, négative ou supérieure à 100 miles ;
- prix nul, négatif ou supérieur à 500 $ (les prix négatifs sont des annulations, codes de paiement 3 et 4) ;
- pourboire négatif ou supérieur à 200 $ ;
- temps d'attente VTC négatif ou supérieur à 60 minutes ;
- paie du chauffeur nulle, négative ou supérieure à 500 $ ;
- zone d'arrivée inconnue.

**Valeurs corrigées**

- `frais_cbd` vide en 2024 → 0, car le péage n'existait pas encore ;
- péage négatif (annulation) → 0 ;
- montant de péage inhabituel → tarif officiel (0,75 $ pour les taxis, 1,50 $ pour les VTC) ;
- code de paiement manquant → 5 (« inconnu » dans le dictionnaire TLC).

## Structure du dépôt

```
.
├── 01_extraction_renommage.ipynb   extraction des colonnes utiles, renommage, sauvegarde
├── 02_yellow_clean.ipynb           nettoyage des taxis jaunes
├── 02_green_clean.ipynb            nettoyage des taxis verts
├── 02_fhv_clean.ipynb              nettoyage des FHV
├── 02_hvfhv_clean.ipynb            nettoyage des VTC (Uber, Lyft…)
├── requirements.txt                bibliothèques Python nécessaires
├── .gitignore                      exclut les données et les fichiers temporaires
└── README.md
```

Dossiers créés à l'exécution (non versionnés) :

- `row_data/` : extraits des CSV bruts (colonnes utiles seulement) ;
- `renamed_data/` : tables renommées ;
- `clean_data/` : tables nettoyées, compressées en `.csv.gz`.

## Installation et exécution

```bash
git clone https://github.com/BenOthmenRayen/nyc-tlc-Machine_Learning.git
cd nyc-tlc-Machine_Learning
pip install -r requirements.txt
```

1. Télécharger les fichiers Yellow, Green, FHV et HVFHV de février-mars 2024 et février-mars 2025, ainsi que `taxi_zone_lookup.csv`, depuis le site de la TLC.
2. Indiquer leurs emplacements dans la première cellule de `01_extraction_renommage.ipynb`.
3. Exécuter les notebooks dans l'ordre : `01`, puis les quatre notebooks `02_*_clean`.

Prévoir plusieurs Go d'espace disque libre et au moins 16 Go de mémoire vive pour la table HVFHV.

## Avancement

- [x] Extraction et renommage des 4 sources
- [x] Nettoyage Yellow, Green, FHV et HVFHV
- [ ] Agrégation par zone et intégration
- [ ] Feature engineering et préparation
- [ ] K-Means, CAH et DBSCAN
- [ ] Nommage des segments
- [ ] Volet supervisé : prédiction de la part taxi / VTC par zone et tranche horaire
- [ ] Recommandations par segment

## Auteur

**Rayen Ben Othmen** — ESPRIT, module Machine Learning (2026)

- GitHub : [@BenOthmenRayen](https://github.com/BenOthmenRayen) · [@Rayennbenothmen](https://github.com/Rayennbenothmen)

Données : © NYC Taxi & Limousine Commission, publiées en open data.
