# Apprentissage statistique supervisé en analyse de survie

Ce dépôt compare plusieurs méthodes classiques et modernes d'analyse de survie sur trois jeux de données disponibles dans `pycox` :

* **METABRIC** : 1 904 individus, 9 covariables, données sur le cancer du sein ;
* **GBSG** : 2 232 individus, 7 covariables, données sur le cancer du sein ;
* **SUPPORT** : 8 873 individus, 14 covariables, données de patients hospitalisés gravement malades.

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

Mesure la capacité du modèle à classer correctement les individus selon leur risque.
Une valeur proche de $0{,}5$ correspond à un classement proche du hasard.

### C-index d'Uno

Version du C-index corrigée de la censure par pondération IPCW. Il s'agit de la métrique principale de discrimination utilisée dans les notebooks.

### Brier score

Mesure, à un horizon donné, l'erreur entre le statut de survie observé et la probabilité de survie prédite.
Une valeur faible indique de meilleures prédictions.

### Integrated Brier Score

Résume le Brier score sur une période :

$$
IBS = \frac{1}{t_{\max}-t_{\min}} \int_{t_{\min}}^{t_{\max}} BS(t)\, dt
$$

### Calibration

Compare les probabilités de survie prédites aux survies réellement observées par Kaplan–Meier.
Une bonne discrimination n'implique pas nécessairement une bonne calibration.

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
