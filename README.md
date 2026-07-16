# Apprentissage statistique supervisé en analyse de survie

Ce dépôt compare plusieurs méthodes classiques et modernes d'analyse de survie sur trois jeux de données. Pour chacun d'eux, nous utilisons uniquement la version mise à disposition par la librairie `pycox` (chargement via `pycox.datasets`), déjà nettoyée et prétraitée par les auteurs de la librairie, afin de garantir une comparaison reproductible entre les différentes méthodes.

### METABRIC

Molecular Taxonomy of Breast Cancer International Consortium. 1 904 individus, 9 covariables (expression génique et variables cliniques telles que l'âge, la taille de la tumeur, le statut des récepteurs hormonaux). L'événement d'intérêt est le décès du patient.

### GBSG

German Breast Cancer Study Group. 2 232 individus, 7 covariables cliniques (âge, statut ménopausique, taille et grade de la tumeur, nombre de ganglions positifs, traitement hormonal). L'événement d'intérêt est la récidive ou le décès.

### SUPPORT

Study to Understand Prognoses and Preferences for Outcomes and Risks of Treatments. 8 873 individus, 14 covariables physiologiques et démographiques (âge, comorbidités, mesures biologiques) collectées sur des patients hospitalisés en soins intensifs. L'événement d'intérêt est le décès du patient.

Chaque observation contient :

* des covariables (X) ;
* un temps observé `duration` ;
* un indicateur `event`, égal à 1 si l'événement est observé et à 0 en cas de censure.

## Objectif

L'objectif est de comparer les modèles au-delà du seul C-index, en étudiant :

* la discrimination ;
* la précision des probabilités de survie ;
* la calibration ;
* les hypothèses et limites propres à chaque méthode.

## Modèles étudiés

* méthodes non paramétriques : Kaplan–Meier et Nelson–Aalen ;
* modèle de Cox à risques proportionnels ;
* Random Survival Forest ;
* Survival Support Vector Machine ;
* Gradient Boosting Survival Analysis avec perte de Cox ;
* DeepSurv ;
* DeepHit en temps discret — en cours.

## Métriques utilisées

### C-index de Harrell

Le concordance index (C-index) de Harrell mesure la capacité du modèle à classer correctement les individus selon leur risque. Pour chaque paire de patients comparable (c'est-à-dire dont on peut établir avec certitude lequel a survécu le plus longtemps), on regarde si le modèle attribue bien un risque plus élevé au patient qui a l'événement le plus tôt :

$$
C = P(\hat{R}_i > \hat{R}_j \mid T_i < T_j)
$$

où $\hat{R}_i$ est le score de risque prédit pour l'individu $i$ et $T_i$ son temps d'événement observé. Une valeur de $1$ correspond à un classement parfait, une valeur proche de $0{,}5$ correspond à un classement proche du hasard. Sa limite principale est de ne pas gérer correctement les paires non comparables à cause de la censure, ce qui introduit un biais lorsque le taux de censure est élevé.

### C-index d'Uno

Version du C-index corrigée de la censure par pondération IPCW (Inverse Probability of Censoring Weighting). Chaque paire comparable est pondérée par l'inverse de la probabilité d'être encore non censuré au moment de l'événement, ce qui corrige le biais introduit par la censure dépendant du temps :

$$
C_{\text{Uno}} = \frac{\sum_{i \neq j} \delta_i \, \hat{G}(T_i)^{-2} \, \mathbb{1}\{T_i < T_j\} \, \mathbb{1}\{\hat{R}_i > \hat{R}_j\}}{\sum_{i \neq j} \delta_i \, \hat{G}(T_i)^{-2} \, \mathbb{1}\{T_i < T_j\}}
$$

où $\delta_i$ est l'indicateur d'événement et $\hat{G}$ est l'estimateur de Kaplan–Meier de la fonction de survie de la censure. Il s'agit de la métrique principale de discrimination utilisée dans les notebooks.

### Brier score

À un horizon de temps $t$ donné, le Brier score mesure l'erreur quadratique moyenne entre le statut de survie observé et la probabilité de survie prédite par le modèle, en pondérant les observations censurées par IPCW pour rester non biaisé :

$$
BS(t) = \frac{1}{n} \sum_{i=1}^{n} \Big[ \frac{\big(0 - \hat{S}(t \mid X_i)\big)^2 \, \mathbb{1}\{T_i \le t, \delta_i = 1\}}{\hat{G}(T_i)} + \frac{\big(1 - \hat{S}(t \mid X_i)\big)^2 \, \mathbb{1}\{T_i > t\}}{\hat{G}(t)} \Big]
$$

où $\hat{S}(t \mid X_i)$ est la probabilité de survie prédite au temps $t$ pour l'individu $i$. Une valeur faible indique de meilleures prédictions ; contrairement au C-index, le Brier score évalue la qualité calibrée des probabilités et pas seulement le classement.

### Integrated Brier Score

Résume le Brier score sur toute une période d'intérêt $[t_{\min}, t_{\max}]$ en une seule valeur, ce qui permet de comparer les modèles sans avoir à fixer un horizon arbitraire :

$$
IBS = \frac{1}{t_{\max}-t_{\min}} \int_{t_{\min}}^{t_{\max}} BS(t)\, dt
$$

### Calibration

Compare les probabilités de survie prédites par le modèle aux probabilités de survie réellement observées, estimées par Kaplan–Meier sur des groupes de patients (par exemple des quantiles de risque prédit). Concrètement on trace des courbes de calibration : pour être bien calibré, un modèle qui prédit une probabilité de survie de $80\%$ à un horizon donné doit correspondre, dans les faits, à un groupe de patients dont environ $80\%$ sont encore en vie à cet horizon. Une bonne discrimination (bon C-index) n'implique pas nécessairement une bonne calibration : un modèle peut très bien ordonner les patients par risque tout en étant systématiquement trop optimiste ou trop pessimiste sur les probabilités absolues.

### Estimateurs non paramétriques (Kaplan–Meier et Nelson–Aalen)

Utilisés en amont comme références descriptives plutôt que comme métriques de comparaison de modèles :

* **Kaplan–Meier** estime directement la fonction de survie $S(t) = P(T > t)$ à partir des données observées, sans hypothèse sur la forme du risque, et sert de référence pour les courbes de calibration.
* **Nelson–Aalen** estime la fonction de risque cumulé $\Lambda(t)$, dont dérive l'estimateur de Breslow utilisé pour obtenir des probabilités de survie à partir d'un modèle de Cox.

## Organisation des notebooks

```text
notebooks/
├── methodes_classiques_cox.ipynb
├── random_survival_forest.ipynb
├── survival_svm.ipynb
├── gradient_boosting_cox.ipynb
├── deepsurv.ipynb
└── deephit.ipynb
```

Les modèles utilisent des partitions train/test communes afin de permettre une comparaison équitable.
Les hyperparamètres des modèles les plus complexes sont sélectionnés sur une validation interne, sans utiliser le jeu de test.

## Bibliothèques principales

```bash
pip install numpy pandas matplotlib scikit-learn
pip install lifelines scikit-survival
pip install pycox torchtuples torch
```

## État du projet

* Cox et méthodes classiques : terminé ;
* Random Survival Forest : terminé ;
* Survival SVM : terminé ;
* Gradient Boosting Survival Analysis : terminé ;
* DeepSurv : terminé ;
* DeepHit : en cours ;
* comparaison finale des modèles : à venir.
