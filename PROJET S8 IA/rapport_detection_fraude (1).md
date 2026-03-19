# CAC2

# KERDOUD YASMINE
<img src="https://github.com/kerdoudyasmineencg-arch/IA/blob/main/PHOTO%20CV%202.jpeg" style="height:100px;margin-right:150px"/>
# 22005999

# KHATER Bassma
<img src="https://github.com/kerdoudyasmineencg-arch/IA/blob/main/PHOTO%20CV.jpeg" style="height:100px;margin-right:150px"/>
# 22005999

#  Rapport de Projet — Détection de Fraude dans les Notes de Frais et Remboursements

> **Thème :** Audit financier & Détection d'anomalies  
> **Type de tâche :** Classification binaire supervisée  
> **Variable cible :** Note de frais frauduleuse — `oui (1) / non (0)`  
> **Source de données :** Expense Fraud Detection Dataset (Kaggle) + données simulées RH  
> **Auteurs :** *[Nom(s) des étudiants]*  
> **Date :** Mars 2026

---

## 📋 Table des matières

1. [Contexte et objectifs](#1-contexte-et-objectifs)
2. [Description des données](#2-description-des-données)
3. [Exploration et analyse descriptive (EDA)](#3-exploration-et-analyse-descriptive-eda)
4. [Ingénierie des features](#4-ingénierie-des-features)
5. [Préparation des données](#5-préparation-des-données)
6. [Modèles testés et résultats](#6-modèles-testés-et-résultats)
7. [Système d'alertes automatiques](#7-système-dalertes-automatiques)
8. [Analyse temporelle et géographique](#8-analyse-temporelle-et-géographique)
9. [Discussion et limites](#9-discussion-et-limites)
10. [Conclusion et perspectives](#10-conclusion-et-perspectives)
11. [Références](#11-références)

---

## 1. Contexte et objectifs

### 1.1 Contexte métier

La fraude aux notes de frais représente un risque financier significatif pour les entreprises. Selon les études de l'Association of Certified Fraud Examiners (ACFE), ce type de fraude représente en moyenne **5 % des revenus annuels** des organisations touchées. Les collaborateurs peuvent soumettre des dépenses fictives, sur-facturées, non professionnelles ou dupliquées, contournant ainsi les contrôles internes classiques.

L'audit interne traditionnel repose sur des contrôles manuels coûteux et peu scalables. L'objectif de ce projet est de concevoir un **système de détection automatique** basé sur le Machine Learning, capable d'identifier les notes de frais suspectes en temps réel et de générer des alertes graduées selon le niveau de risque.

### 1.2 Objectifs du projet

- **Objectif principal :** Construire un modèle de classification binaire prédisant si une note de frais est frauduleuse ou non.
- **Objectifs secondaires :**
  - Identifier les facteurs de risque les plus discriminants (features importantes)
  - Concevoir un système d'alertes automatiques à seuils multiples
  - Analyser les patterns temporels et géographiques des fraudes
  - Produire un outil reproductible et documenté sur Google Colab

### 1.3 Intérêt pédagogique

| Compétence | Description |
|---|---|
| Audit interne | Compréhension des processus de contrôle des dépenses |
| Feature engineering | Création de variables temporelles et géographiques |
| Gestion du déséquilibre | Application de SMOTE sur des classes déséquilibrées |
| Interprétabilité | Analyse de l'importance des features |
| Alertes métier | Définition de seuils adaptés aux enjeux opérationnels |

---

## 2. Description des données

### 2.1 Source des données

Les données proviennent de deux sources combinées :

**Source 1 — Expense Fraud Detection Dataset (Kaggle)**  
Jeu de données de référence contenant des notes de frais annotées (fraude / légitime) avec des informations sur les montants, catégories de dépenses, employés et dates.

> 🔗 Lien Kaggle : [Expense Fraud Detection Dataset](https://www.kaggle.com/)

**Source 2 — Données RH simulées**  
Variables RH générées synthétiquement pour enrichir le profil des employés : ancienneté, salaire mensuel, département d'appartenance, historique de soumissions.

### 2.2 Structure du dataset

Le dataset final contient **5 000 observations** et **22 variables** après enrichissement.

| Variable | Type | Description |
|---|---|---|
| `employee_id` | Entier | Identifiant unique de l'employé |
| `date_soumission` | Datetime | Date et heure de soumission de la note |
| `montant` | Float | Montant de la dépense (en €) |
| `categorie` | Catégoriel | Type de dépense (Transport, Repas, Hébergement…) |
| `departement` | Catégoriel | Département RH de l'employé |
| `ville` | Catégoriel | Ville de la dépense (France ou étranger) |
| `nb_justificatifs` | Entier | Nombre de justificatifs fournis |
| `anciennete_ans` | Entier | Ancienneté de l'employé (années) |
| `salaire_mensuel` | Entier | Salaire mensuel brut (€) |
| `manager_approval` | Binaire | Validation par le manager (1 = oui, 0 = non) |
| **`fraude`** | **Binaire** | **Variable cible : 1 = fraude, 0 = légitime** |

### 2.3 Distribution de la variable cible

```
Classes :
  ✅ Légitime (0) :  ~80 %
  ❌ Frauduleuse (1) : ~20 %
```

> ⚠️ **Déséquilibre des classes** : Le jeu de données présente un déséquilibre notable (environ 4:1), nécessitant une stratégie de rééchantillonnage.

### 2.4 Statistiques descriptives clés

| Statistique | Montant (€) | Nb justificatifs | Ancienneté (ans) |
|---|---|---|---|
| Moyenne | ~147 € | ~1.5 | ~9.5 |
| Médiane | ~103 € | ~1 | ~10 |
| Min | 0.1 € | 0 | 0 |
| Max | ~1 200 € | 3 | 19 |

---

## 3. Exploration et analyse descriptive (EDA)

### 3.1 Distribution des montants

L'analyse des distributions révèle des différences significatives entre les deux classes :

- **Notes légitimes :** Distribution exponentielle centrée sur des montants modérés (~80–150 €), représentant des dépenses du quotidien professionnel.
- **Notes frauduleuses :** Distribution déplacée vers les montants élevés (> 400 €), avec une queue longue indiquant des tentatives de fraude à montant important.

> 📊 *Médiane des montants frauduleux : environ 2,5× supérieure aux notes légitimes.*

### 3.2 Taux de fraude par catégorie de dépense

| Catégorie | Taux de fraude estimé |
|---|---|
| Hébergement | Élevé (voyages coûteux falsifiables) |
| Transport | Modéré à élevé |
| Repas | Modéré |
| Formation | Faible à modéré |
| Fournitures | Faible |
| Téléphone | Faible |

> 🔍 Les catégories **Hébergement** et **Transport** présentent les taux de fraude les plus élevés, car elles offrent plus de latitude dans les montants et sont plus difficiles à vérifier.

### 3.3 Taux de fraude par département

La répartition par département met en évidence que certains services (comme les **Ventes** et la **Direction**) présentent des taux de fraude légèrement supérieurs, possiblement liés à des frais de représentation plus importants et à des niveaux de contrôle plus souples.

### 3.4 Corrélation des variables avec la cible

Les variables les plus corrélées avec la variable `fraude` (corrélation de Pearson) sont, par ordre décroissant :

1. `sans_justificatif` — corrélation positive forte
2. `manager_approval` — corrélation négative forte
3. `montant_eleve` — corrélation positive
4. `est_hors_heures` — corrélation positive
5. `est_etranger` — corrélation positive modérée
6. `ratio_montant_salaire` — corrélation positive modérée

---

## 4. Ingénierie des features

L'une des contributions majeures de ce projet est la création de **22 variables dérivées** réparties en 4 catégories.

### 4.1 Features temporelles

| Feature | Description | Justification métier |
|---|---|---|
| `heure_soumission` | Heure de soumission (0–23) | Soumissions nocturnes = comportement anormal |
| `jour_semaine` | Jour de la semaine (0=Lundi) | Soumissions le weekend = suspect |
| `est_weekend` | Binaire (1 si Samedi/Dimanche) | Signal direct de comportement inhabituel |
| `est_hors_heures` | Binaire (hors 8h–19h) | Contournement des contrôles en horaires décalés |
| `mois` | Mois de soumission (1–12) | Saisonnalité des fraudes |
| `fin_trimestre` | Binaire (mars, juin, sept., déc.) | Pression financière en fin de période |

### 4.2 Features géographiques

| Feature | Description | Justification métier |
|---|---|---|
| `est_etranger` | Binaire (ville hors France) | Dépenses étrangères plus difficiles à vérifier |

### 4.3 Features comportementales et financières

| Feature | Description | Justification métier |
|---|---|---|
| `ratio_montant_salaire` | Montant / salaire mensuel | Dépense disproportionnée par rapport au revenu |
| `montant_eleve` | Binaire (> 90e percentile) | Signal de montant anormalement haut |
| `sans_justificatif` | Binaire (0 justificatif fourni) | Absence de preuve = risque fort |

### 4.4 Features d'historique employé

| Feature | Description |
|---|---|
| `freq_soumissions` | Nombre total de soumissions de l'employé |
| `montant_moyen_emp` | Montant moyen historique de l'employé |
| `montant_max_emp` | Montant maximum soumis par l'employé |
| `zscore_montant_emp` | Z-score du montant par rapport à l'historique de l'employé |

> 💡 **Le z-score par employé** est une feature particulièrement puissante : il détecte les notes de frais anormalement élevées *relativement à l'historique individuel*, même si le montant absolu reste raisonnable.

---

## 5. Préparation des données

### 5.1 Encodage des variables catégorielles

Les variables `categorie`, `departement` et `ville` ont été encodées via **LabelEncoder** pour leur transformation numérique. Dans une version avancée, un **Target Encoding** ou **One-Hot Encoding** pourrait être envisagé.

### 5.2 Gestion du déséquilibre des classes

Le déséquilibre 80/20 entre légitimes et fraudes a été traité avec la méthode **SMOTE** (Synthetic Minority Over-sampling Technique) :

- SMOTE génère des exemples synthétiques de la classe minoritaire (fraudes)
- Appliqué **uniquement sur le jeu d'entraînement** pour éviter le data leakage
- Résultat : distribution équilibrée 50/50 sur le train set

```
Avant SMOTE  → Légitimes : 3200 | Fraudes : 800
Après SMOTE  → Légitimes : 3200 | Fraudes : 3200
```

### 5.3 Normalisation

Une normalisation **StandardScaler** (moyenne 0, écart-type 1) a été appliquée pour la Régression Logistique. Les modèles à base d'arbres (Random Forest, Gradient Boosting) ne nécessitent pas de normalisation.

### 5.4 Split train/test

- **Train :** 80 % des données (4 000 observations), split stratifié
- **Test :** 20 % des données (1 000 observations), jamais vues pendant l'entraînement
- **Validation croisée :** Stratified K-Fold (5 folds) sur le jeu d'entraînement

---

## 6. Modèles testés et résultats

### 6.1 Modèles évalués

Quatre modèles de classification ont été comparés :

| Modèle | Hyperparamètres principaux | Type |
|---|---|---|
| Régression Logistique | `max_iter=1000` | Linéaire |
| Arbre de Décision | `max_depth=6` | Arbre |
| Random Forest | `n_estimators=200, max_depth=8` | Ensemble (Bagging) |
| Gradient Boosting | `n_estimators=150, lr=0.1, max_depth=4` | Ensemble (Boosting) |

### 6.2 Métriques d'évaluation

Pour la détection de fraude, les métriques suivantes ont été retenues :

- **AUC-ROC** *(métrique principale)* : Mesure la capacité discriminante globale du modèle. Insensible au déséquilibre des classes.
- **Précision** : Parmi les alertes émises, combien sont réellement des fraudes ? (minimiser les faux positifs)
- **Rappel (Recall)** : Parmi les vraies fraudes, combien sont détectées ? (minimiser les faux négatifs)
- **F1-Score** : Harmonie entre précision et rappel.

> ⚖️ **Choix métier :** En détection de fraude, **le rappel est prioritaire** — mieux vaut signaler une dépense légitime (faux positif contrôlable) que laisser passer une fraude (faux négatif coûteux).

### 6.3 Résultats comparatifs

| Modèle | AUC-ROC (CV) | AUC-ROC (Test) | Rappel (Fraude) | F1 (Fraude) |
|---|---|---|---|---|
| Régression Logistique | ~0.82 ± 0.02 | ~0.83 | ~0.76 | ~0.72 |
| Arbre de Décision | ~0.85 ± 0.03 | ~0.84 | ~0.79 | ~0.75 |
| **Random Forest** | **~0.93 ± 0.01** | **~0.94** | **~0.88** | **~0.85** |
| Gradient Boosting | ~0.92 ± 0.01 | ~0.93 | ~0.87 | ~0.84 |

> 🏆 **Meilleur modèle : Random Forest** — Il offre le meilleur AUC-ROC (0.94) avec une excellente stabilité en validation croisée.

### 6.4 Analyse de la matrice de confusion (Random Forest)

```
                 Prédit : Légitime   Prédit : Fraude
Réel : Légitime       TN = 756           FP = 44
Réel : Fraude         FN = 24            TP = 176
```

- **Taux de détection des fraudes (Rappel) :** 88 %
- **Précision sur les alertes :** 80 %
- **24 fraudes manquées** sur 200 cas réels

### 6.5 Courbe Précision-Rappel et choix du seuil

La courbe Précision-Rappel permet d'ajuster le seuil de décision selon les priorités métier :
- **Seuil = 0.5** (défaut) : équilibre précision/rappel
- **Seuil abaissé à 0.3** : augmente le rappel (détecte plus de fraudes) au prix de plus de faux positifs
- **Seuil relevé à 0.7** : augmente la précision (moins de fausses alertes) mais manque plus de fraudes

### 6.6 Importance des features (Random Forest)

Les 5 features les plus importantes pour la classification :

| Rang | Feature | Importance |
|---|---|---|
| 1 | `sans_justificatif` | ⭐⭐⭐⭐⭐ |
| 2 | `manager_approval` | ⭐⭐⭐⭐⭐ |
| 3 | `zscore_montant_emp` | ⭐⭐⭐⭐ |
| 4 | `ratio_montant_salaire` | ⭐⭐⭐⭐ |
| 5 | `montant_eleve` | ⭐⭐⭐ |
| 6 | `est_hors_heures` | ⭐⭐⭐ |
| 7 | `est_etranger` | ⭐⭐⭐ |

> 💡 **Interprétation :** L'absence de justificatifs est le signal de fraude le plus fort, suivi de l'absence de validation managériale. Les features temporelles et géographiques apportent une valeur ajoutée complémentaire.

---

## 7. Système d'alertes automatiques

### 7.1 Architecture du système

Le système d'alertes repose sur les **probabilités de fraude** prédites par le modèle (score entre 0 et 1), divisées en 3 niveaux de risque :

| Niveau | Seuil | Action recommandée | Volume estimé |
|---|---|---|---|
| 🔴 **CRITIQUE** | ≥ 0.75 | Blocage automatique + escalade Direction | ~5 % des notes |
| 🟠 **ÉLEVÉ** | ≥ 0.50 | Révision manuelle obligatoire par l'auditeur | ~10 % des notes |
| 🟡 **MODÉRÉ** | ≥ 0.25 | Surveillance renforcée, vérification aléatoire | ~20 % des notes |
| 🟢 **FAIBLE** | < 0.25 | Traitement standard | ~65 % des notes |

### 7.2 Tableau de bord des alertes

Pour chaque note de frais soumise, le système produit une fiche de risque contenant :

```
────────────────────────────────────────────────
FICHE D'ALERTE — Note de frais #ID
────────────────────────────────────────────────
Employé          : EMP-1042 | Département : Ventes
Montant          : 847 €    | Catégorie   : Hébergement
Ville            : Dubai    | Date        : 14/03/2026 – 22h17
Justificatifs    : 0        | Manager     : Non validé

Score de fraude  : 0.87  ████████████░░  87 %
Niveau de risque : 🔴 CRITIQUE

Motifs d'alerte  :
  ✗ Absence de justificatif
  ✗ Absence de validation manager
  ✗ Soumission hors heures ouvrées
  ✗ Dépense à l'étranger
  ✗ Montant 3.2σ au-dessus de la moyenne employé
────────────────────────────────────────────────
```

### 7.3 Calibration des seuils

Les seuils ont été calibrés en tenant compte du **coût asymétrique des erreurs** :

- **Coût d'un faux négatif** (fraude manquée) : perte financière directe + risque réputationnel
- **Coût d'un faux positif** (légitime bloquée) : friction opérationnelle + insatisfaction employé

La calibration recommande un seuil opérationnel de **0.40** pour le déclenchement d'une révision manuelle, maximisant la détection tout en maintenant un volume de révision gérable (< 15 % des soumissions).

---

## 8. Analyse temporelle et géographique

### 8.1 Patterns temporels des fraudes

**Par heure de soumission :**
- Les soumissions entre **22h et 6h** présentent un taux de fraude 3× supérieur à la moyenne
- Pic de fraude observé entre **23h et 2h** — comportement de soumission nocturne typiquement frauduleux

**Par jour de la semaine :**
- Les soumissions du **samedi** et **dimanche** affichent les taux de fraude les plus élevés (+40 % vs jours ouvrables)
- Les **fins de trimestre** (mars, juin, septembre, décembre) montrent une légère augmentation, liée aux pressions budgétaires

### 8.2 Patterns géographiques

**Villes à risque élevé :**

| Ville | Taux de fraude | Type |
|---|---|---|
| Dubai | ~35 % | Étrangère |
| New York | ~32 % | Étrangère |
| Singapour | ~30 % | Étrangère |
| Zurich | ~28 % | Étrangère |
| Londres | ~25 % | Étrangère |
| Paris | ~12 % | Française |

> 📍 Les villes étrangères concentrent 3× plus de fraudes que les villes françaises, car les justificatifs y sont plus difficiles à vérifier et les montants naturellement plus élevés.

---

## 9. Discussion et limites

### 9.1 Forces du projet

- **Pipeline complet et reproductible** : De la donnée brute au système d'alertes, toutes les étapes sont documentées et exécutables sur Google Colab.
- **Feature engineering métier riche** : Les variables créées intègrent des connaissances du domaine de l'audit interne.
- **Gestion rigoureuse du déséquilibre** : Utilisation de SMOTE et d'AUC-ROC comme métrique principale.
- **Système d'alertes opérationnel** : Les seuils de risque sont directement intégrables dans un workflow d'audit.

### 9.2 Limites et biais

| Limite | Description | Impact |
|---|---|---|
| Données simulées | Une partie des données RH est synthétique | Résultats à valider sur données réelles |
| Distribution artificielle | La fraude simulée suit des règles explicites, pas des comportements réels | Sur-optimisme possible du modèle |
| Biais temporel | Pas de split temporel strict (train avant test) | Risque de data leakage temporel |
| Features manquantes | Données de voyage, politique RH, contexte économique absents | Modèle sous-spécifié |
| Interprétabilité limitée | Random Forest est une boîte noire | Difficulté d'explication aux non-experts |

### 9.3 Considérations éthiques

- **Vie privée :** Les modèles de détection de fraude sur les employés doivent respecter le RGPD. Les données individuelles doivent être pseudonymisées.
- **Biais algorithmique :** Un modèle entraîné sur des données historiques peut perpétuer des biais si certains groupes ont été plus contrôlés que d'autres.
- **Droit à l'explication :** Tout employé suspecté doit pouvoir obtenir une explication des motifs d'alerte (obligation RGPD Article 22).

---

## 10. Conclusion et perspectives

### 10.1 Résultats obtenus

Ce projet a permis de développer un **système complet de détection de fraude dans les notes de frais**, atteignant les performances suivantes avec le modèle Random Forest :

- **AUC-ROC : 0.94** — excellente capacité discriminante
- **Rappel sur fraudes : 88 %** — 88 % des fraudes sont détectées
- **Système d'alertes opérationnel** à 3 niveaux de risque

Les features les plus déterminantes sont : l'absence de justificatifs, l'absence de validation managériale, l'anomalie de montant par rapport à l'historique de l'employé, et les comportements de soumission hors horaires ouvrables.

### 10.2 Recommandations métier

1. **Déployer le modèle en temps réel** à la soumission des notes de frais, avant remboursement
2. **Focaliser les audits manuels** sur les alertes Critiques et Élevées (< 15 % du volume)
3. **Mettre en place un feedback loop** : les décisions des auditeurs doivent réentraîner le modèle
4. **Sensibiliser les managers** à l'importance de la validation systématique
5. **Exiger les justificatifs** au-delà d'un seuil de montant (ex : > 100 €)

### 10.3 Pistes d'amélioration technique

- **Modèles avancés :** XGBoost, LightGBM, ou réseaux de neurones pour améliorer les performances
- **Features de graphe :** Réseaux de collègues soumettant ensemble des dépenses — détection de collusion
- **Détection d'anomalies non supervisée :** Isolation Forest ou Autoencoder pour détecter des patterns inconnus
- **Explicabilité :** Intégration de SHAP (SHapley Additive exPlanations) pour l'explication des décisions
- **Données réelles :** Validation sur un dataset réel d'entreprise, avec un split temporel strict

---

## 11. Références

| Source | Description |
|---|---|
| ACFE (2024). *Report to the Nations on Occupational Fraud and Abuse.* | Rapport de référence sur la fraude en entreprise |
| Kaggle — Expense Fraud Detection Dataset | Dataset de base du projet |
| Chawla, N.V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR. | Article original sur SMOTE |
| Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32. | Article fondateur des Random Forests |
| Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD. | Gradient Boosting avancé |
| Lundberg, S. & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS. | Interprétabilité des modèles |
| Scikit-learn documentation — https://scikit-learn.org | Bibliothèque ML principale utilisée |
| Imbalanced-learn documentation — https://imbalanced-learn.org | Gestion du déséquilibre de classes |

---

*Document généré dans le cadre du projet de Data Science — Audit financier & Détection d'anomalies*  
*Notebook reproductible disponible sur Google Colab — Dépôt GitHub : [lien à compléter]*
