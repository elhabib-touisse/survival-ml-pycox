# Apprentissage statistique en analyse de survie

Ce projet compare plusieurs méthodes classiques et modernes pour les données de survie censurées à droite.

## Données

Les expériences utilisent trois bases issues de `pycox` :

- METABRIC
- SUPPORT
- FLCHAIN

Chaque observation est représentée sous la forme :

\[
(X_i, Y_i, \Delta_i)
\]

où :

- \(X_i\) désigne les covariables ;
- \(Y_i\) désigne le temps observé ;
- \(\Delta_i\) désigne l'indicateur d'événement.

## Organisation du projet

1. Analyse exploratoire et méthodes classiques de survie
2. Random Survival Forests
3. Survival SVM
4. Gradient Boosting Machines
5. DeepSurv
6. Comparaison finale

## Méthodes classiques

- Kaplan-Meier
- Nelson-Aalen
- Modèle de Cox
- Groupes de risque
- Test log-rank

## Méthodes ML

- Random Survival Forests
- Survival Support Vector Machines
- Gradient Boosting Survival
- DeepSurv

## Métriques

- C-index
- Time-dependent AUC
- Integrated Brier Score
- Kaplan-Meier par groupes de risque
