# Apprentissage statistique supervisé en analyse de survie

## Présentation du projet

Ce projet étudie l’application de méthodes d’**apprentissage statistique supervisé** à des données de survie.

Contrairement à une régression classique, l’analyse de survie cherche à prédire le temps avant la survenue d’un événement, par exemple un décès, une rechute ou une défaillance. La difficulté principale vient de la présence de **données censurées** : pour certains individus, l’événement n’a pas encore été observé à la fin du suivi.

Pour chaque individu (i), les données observées sont de la forme :

[
Y_i=\min(T_i,C_i),
\qquad
\delta_i=\mathbf 1_{{T_i\leq C_i}},
]

où :

* (T_i) est le véritable temps avant l’événement ;
* (C_i) est le temps de censure ;
* (Y_i) est le temps observé ;
* (\delta_i) indique si l’événement a effectivement été observé.

L’objectif du projet est de comparer plusieurs modèles statistiques et algorithmes de machine learning en tenant compte des différentes dimensions de la qualité prédictive.

---

## Objectifs

Le projet vise à :

* explorer plusieurs jeux de données réelles de survie ;
* présenter les principales méthodes descriptives et non paramétriques ;
* ajuster des modèles statistiques classiques ;
* appliquer des méthodes modernes d’apprentissage supervisé ;
* vérifier les hypothèses du modèle de Cox ;
* comparer les modèles sur des données de test ;
* étudier à la fois la discrimination, la précision probabiliste et la calibration ;
* montrer pourquoi le C-index ne suffit pas à lui seul pour évaluer un modèle de survie.

---

## Jeux de données

L’étude est réalisée sur trois jeux de données biomédicales.

| Jeu de données | Nombre d’individus | Nombre d’événements | Taux d’événement | Nombre de covariables |
| -------------- | -----------------: | ------------------: | ---------------: | --------------------: |
| METABRIC       |              1 904 |               1 103 |           57,9 % |                     9 |
| GBSG           |              2 232 |               1 267 |           56,8 % |                     7 |
| SUPPORT        |              8 873 |               6 036 |           68,0 % |                    14 |

### METABRIC

METABRIC contient des données cliniques relatives à des patientes atteintes d’un cancer du sein.

### GBSG

GBSG contient des données de suivi de patientes atteintes d’un cancer du sein après traitement.

### SUPPORT

SUPPORT contient des données cliniques concernant des patients hospitalisés souffrant de pathologies graves.

L’utilisation de ces trois bases permet d’évaluer la stabilité des conclusions dans des contextes différents en matière de taille d’échantillon, de taux de censure et de complexité clinique.

---

## Étapes de l’analyse

### 1. Préparation des données

La première étape consiste à :

* identifier les variables temporelles et l’indicateur d’événement ;
* examiner les types des variables ;
* traiter les valeurs manquantes ;
* encoder les variables catégorielles ;
* vérifier les distributions et les valeurs atypiques ;
* préparer les données dans un format compatible avec les bibliothèques de survie.

Pour certains modèles, la cible est représentée par un tableau structuré :

```python
def to_structured(df):
    return np.array(
        list(zip(df["event"].astype(bool), df["duration"].astype(float))),
        dtype=[("event", bool), ("time", float)],
    )
```

---

### 2. Analyse exploratoire

L’analyse exploratoire étudie notamment :

* la distribution des durées observées ;
* la proportion d’événements et de censures ;
* la distribution des covariables ;
* les relations entre les covariables et le statut de l’événement ;
* les différences entre les trois jeux de données.

Cette étape permet d’identifier les difficultés susceptibles d’influencer les modèles : déséquilibre, forte censure, valeurs manquantes, non-linéarités ou hétérogénéité des populations.

---

### 3. Méthodes classiques d’analyse de survie

#### Estimateur de Kaplan-Meier

L’estimateur de Kaplan-Meier estime la fonction de survie :

[
S(t)=\mathbb P(T>t).
]

Son estimateur est :

[
\widehat S(t)
=============

\prod_{t_j\leq t}
\left(
1-\frac{d_j}{n_j}
\right),
]

où (d_j) est le nombre d’événements observés au temps (t_j) et (n_j) le nombre d’individus encore à risque juste avant ce temps.

Il permet de visualiser la probabilité estimée de survivre au-delà d’un instant donné.

#### Estimateur de Nelson-Aalen

Le risque cumulé est estimé par :

[
\widehat\Lambda(t)
==================

\sum_{t_j\leq t}
\frac{d_j}{n_j}.
]

Cet estimateur mesure l’accumulation du risque au cours du temps.

#### Test du log-rank

Le test du log-rank compare les fonctions de survie de plusieurs groupes.

L’hypothèse nulle est :

[
H_0:S_1(t)=S_2(t)
\quad
\text{pour tout }t.
]

Une petite p-value indique que les différences observées entre les courbes de survie sont difficilement compatibles avec l’hypothèse d’égalité des fonctions de survie.

---

## Modèles étudiés

### Modèle à risques proportionnels de Cox

Le modèle de Cox suppose que le risque instantané conditionnel s’écrit :

[
\lambda(t\mid x)
================

\lambda_0(t)\exp(\beta^\top x),
]

où :

* (\lambda_0(t)) est le risque de base ;
* (x) est le vecteur des covariables ;
* (\beta) est le vecteur des coefficients.

Pour une variable (x_j), la quantité

[
\exp(\beta_j)
]

est le hazard ratio associé à une augmentation d’une unité de cette variable, toutes choses égales par ailleurs.

Le modèle est estimé par maximisation de la vraisemblance partielle.

### Vérification de l’hypothèse des risques proportionnels

Le modèle de Cox suppose que les rapports de risques sont constants au cours du temps.

Cette hypothèse est étudiée à l’aide des **résidus de Schoenfeld**. Sous l’hypothèse des risques proportionnels, les résidus ne doivent pas présenter de dépendance systématique au temps.

Une p-value inférieure au seuil choisi suggère que l’effet de la covariable considérée varie au cours du temps.

Les principales solutions possibles sont :

* introduire une interaction entre la variable et le temps ;
* stratifier le modèle ;
* transformer la variable ;
* découper le temps en plusieurs intervalles ;
* utiliser un modèle autorisant explicitement des effets dépendant du temps.

---

### Modèle Weibull AFT

Le modèle Accelerated Failure Time agit directement sur le temps de survie :

[
\log T
======

\beta^\top x+\sigma\varepsilon.
]

Dans cette formulation, les covariables accélèrent ou ralentissent le temps avant l’événement.

Le facteur

[
\exp(\beta_j)
]

peut être interprété comme un facteur multiplicatif sur le temps de survie, selon la paramétrisation utilisée.

Le modèle Weibull permet également d’obtenir une fonction de survie complète et des probabilités de survie individuelles.

---

### Random Survival Forest

La Random Survival Forest est une adaptation des forêts aléatoires aux données censurées.

Chaque arbre est construit à partir :

* d’un échantillon bootstrap ;
* d’un sous-ensemble aléatoire de variables ;
* d’un critère de séparation adapté aux données de survie.

La forêt permet de représenter :

* des relations non linéaires ;
* des interactions entre variables ;
* des effets complexes difficiles à modéliser avec une spécification paramétrique.

Elle produit notamment des scores de risque et des fonctions de survie individuelles.

---

## Séparation entraînement-test

Les données sont séparées en un ensemble d’entraînement et un ensemble de test :

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=random_state,
    stratify=df["event"],
)
```

La stratification permet de conserver approximativement la même proportion d’événements dans les deux ensembles.

Les opérations de prétraitement doivent être apprises uniquement sur l’ensemble d’entraînement afin d’éviter une fuite d’information.

---

## Gestion de l’horizon temporel

Certaines métriques de survie nécessitent que les temps évalués restent dans la zone couverte par les données d’entraînement.

Les observations de test dont le temps dépasse le suivi maximal exploitable dans l’ensemble d’entraînement peuvent être retirées ou tronquées selon le protocole utilisé.

Cette précaution est particulièrement importante pour les métriques reposant sur l’estimation de la distribution de la censure. Elle évite d’effectuer des calculs dans des régions où cette estimation devient instable ou n’est plus définie.

---

## Évaluation des modèles

L’évaluation ne repose pas sur une métrique unique. Plusieurs aspects complémentaires sont étudiés.

### 1. C-index

Le C-index mesure la capacité du modèle à classer correctement les individus selon leur risque.

Il évalue approximativement la probabilité que, parmi deux individus comparables, celui qui subit l’événement le plus tôt reçoive le score de risque le plus élevé.

Une valeur proche de :

* (0{,}5) correspond à un classement proche du hasard ;
* (1) correspond à un classement parfait ;
* une valeur inférieure à (0{,}5) peut indiquer que l’orientation du score est inversée.

Le C-index mesure la **discrimination**, mais pas la qualité des probabilités prédites.

---

### 2. C-index d’Uno

Le C-index d’Uno corrige l’effet de la censure à l’aide d’une pondération par l’inverse de la probabilité de ne pas être censuré.

Il est généralement plus adapté que le C-index classique lorsque la censure est importante.

Son calcul doit être effectué sur un horizon temporel pour lequel l’estimation de la loi de censure reste suffisamment fiable.

---

### 3. AUC dépendante du temps

L’AUC dépendante du temps mesure la capacité du modèle à distinguer, à un horizon (t) :

* les individus ayant subi l’événement avant (t) ;
* les individus encore en vie ou sans événement à (t).

Elle permet d’observer si le pouvoir discriminant du modèle reste stable ou se dégrade au cours du temps.

Une AUC moyenne peut également être calculée sur une grille d’horizons.

---

### 4. Score de Brier

À l’horizon (t), le score de Brier compare l’état observé à la probabilité de survie prédite :

[
BS(t)
=====

\frac{1}{n}
\sum_{i=1}^{n}
w_i(t)
\left[
\mathbf 1_{{T_i>t}}
-------------------

\widehat S(t\mid X_i)
\right]^2,
]

où (w_i(t)) corrige la présence de censure.

Contrairement au C-index, le score de Brier utilise directement les probabilités de survie prédites.

Une valeur faible indique de bonnes prédictions probabilistes.

---

### 5. Integrated Brier Score

L’Integrated Brier Score résume le score de Brier sur un intervalle temporel :

[
IBS
===

\frac{1}{\tau_2-\tau_1}
\int_{\tau_1}^{\tau_2}
BS(t),dt.
]

Il évalue la qualité globale de la fonction de survie prédite.

Plus l’IBS est faible, meilleures sont les prédictions probabilistes du modèle sur l’intervalle considéré.

---

## Calibration

La calibration étudie l’accord entre :

* les probabilités de survie prédites par le modèle ;
* les fréquences de survie effectivement observées.

À un horizon (t), un modèle est bien calibré si les individus auxquels il attribue une probabilité de survie proche de (70,%) présentent effectivement une survie observée proche de (70,%).

Une courbe de calibration compare généralement :

[
\text{survie prédite}
\quad\text{et}\quad
\text{survie observée}.
]

La diagonale représente une calibration parfaite.

### Interprétation

* une courbe proche de la diagonale indique une bonne calibration ;
* une courbe située au-dessus ou au-dessous de la diagonale révèle une surestimation ou une sous-estimation systématique de la survie ;
* un modèle peut avoir un bon C-index tout en étant mal calibré.

La discrimination et la calibration répondent donc à deux questions différentes :

* **Discrimination :** le modèle classe-t-il correctement les individus ?
* **Calibration :** les probabilités prédites sont-elles quantitativement fiables ?

---

## Utilisation de la calibration pour corriger un modèle

La calibration peut être utilisée comme outil de diagnostic, mais également comme étape de correction après l’ajustement du modèle.

Une méthode générale consiste à apprendre, sur un ensemble de validation, une fonction reliant les probabilités prédites aux probabilités réellement observées :

[
\widehat S_{\mathrm{corrigée}}(t\mid x)
=======================================

g_t\left(
\widehat S_{\mathrm{initiale}}(t\mid x)
\right).
]

La fonction (g_t) peut être estimée à l’aide de méthodes telles que :

* une régression logistique ;
* une régression isotone ;
* une transformation des logits ;
* un recalibrage du risque de base ;
* une méthode de calibration spécifique aux fonctions de survie.

La correction doit être apprise sur un ensemble distinct de celui utilisé pour entraîner le modèle. Elle doit ensuite être évaluée sur des données de test indépendantes.

Dans ce projet, la calibration est principalement utilisée pour identifier les biais probabilistes des modèles. La correction effective des prédictions n’est pas appliquée, afin de conserver une comparaison directe entre les modèles initialement ajustés.

---

## Résultats du modèle de Cox

| Dataset  | Log-vraisemblance partielle | AIC partiel | C-index entraînement | Statistique LR |           p-value LR | Variables significatives |
| -------- | --------------------------: | ----------: | -------------------: | -------------: | -------------------: | -----------------------: |
| METABRIC |                   -7 382,63 |   14 783,27 |                0,639 |         312,67 | (5,33\times10^{-62}) |                        7 |
| GBSG     |                   -9 014,37 |   18 042,75 |                0,665 |         297,89 | (1,71\times10^{-60}) |                        5 |
| SUPPORT  |                  -51 338,66 |  102 705,31 |                0,572 |         478,43 | (3,44\times10^{-93}) |                        9 |

Les tests du rapport de vraisemblance sont très significatifs pour les trois jeux de données. Les covariables apportent donc collectivement une information prédictive par rapport à un modèle ne contenant aucun effet explicatif.

Le pouvoir discriminant reste toutefois modéré :

* GBSG présente le meilleur C-index d’entraînement ;
* METABRIC présente une discrimination intermédiaire ;
* SUPPORT semble être le jeu de données le plus difficile pour le modèle linéaire de Cox.

Ces résultats doivent être complétés par une évaluation sur l’ensemble de test et par l’étude de la calibration.

---

## Principes de comparaison

Les modèles sont comparés selon plusieurs axes.

| Dimension                 | Métriques ou outils                                   |
| ------------------------- | ----------------------------------------------------- |
| Discrimination globale    | C-index, C-index d’Uno                                |
| Discrimination temporelle | AUC dépendante du temps                               |
| Précision probabiliste    | Score de Brier                                        |
| Précision globale         | Integrated Brier Score                                |
| Calibration               | Courbes de calibration                                |
| Interprétabilité          | Coefficients, hazard ratios, importance des variables |
| Validité du modèle        | Test des risques proportionnels                       |
| Robustesse                | Comparaison sur plusieurs jeux de données             |

Un modèle n’est donc pas considéré comme supérieur uniquement parce qu’il possède le C-index le plus élevé.

---

## Bibliothèques principales

Le projet utilise notamment :

```text
numpy
pandas
matplotlib
scikit-learn
lifelines
scikit-survival
skrub
```

Selon les modèles ajoutés, d’autres bibliothèques peuvent être nécessaires.

---

## Installation

```bash
git clone <URL_DU_DEPOT>
cd <NOM_DU_DEPOT>
pip install -r requirements.txt
```

Le notebook peut ensuite être exécuté avec Jupyter :

```bash
jupyter notebook
```

ou importé dans Google Colab.

---

## Reproductibilité

Les séparations entraînement-test et les modèles stochastiques utilisent une graine aléatoire fixée :

```python
RANDOM_STATE = 42
```

Cette graine permet de reproduire les résultats obtenus avec le même environnement logiciel.

---

## Conclusion

Ce projet met en évidence la nécessité d’une évaluation multidimensionnelle des modèles de survie.

Le C-index reste une mesure importante de discrimination, mais il ne permet pas de savoir si :

* les probabilités de survie sont correctes ;
* les prédictions sont précises à différents horizons ;
* le modèle respecte ses hypothèses ;
* les résultats restent stables d’un jeu de données à l’autre.

L’utilisation conjointe du C-index, de l’AUC dépendante du temps, du score de Brier, de l’Integrated Brier Score et des courbes de calibration fournit une analyse plus complète et plus fiable des performances prédictives.

---

## Auteur

**Elhabib Touisse**

Projet réalisé dans le cadre d’un travail consacré à l’apprentissage statistique supervisé appliqué aux données de survie.
