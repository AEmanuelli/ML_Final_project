# Analyse de l'Impact des Hyperparamètres et de l'Architecture des Réseaux Neuronaux

## Introduction
Dans le cadre de ce projet, nous avons mené une analyse approfondie pour évaluer l'impact des hyperparamètres et de l'architecture des réseaux neuronaux sur les performances des modèles. Cette analyse s'est déroulée en deux parties : l'examen d'un réseau linéaire et d'un réseau non-linéaire. Nous avons exploré l'influence du learning rate, du nombre d'itérations et du nombre de neurones dans la couche cachée sur ces modèles.

## Analyse du Réseau Linéaire
Pour le réseau linéaire, nous avons découvert que la sélection judicieuse des hyperparamètres conduisait à une précision maximale du modèle. De plus, nous avons suivi de près la convergence de la perte et de l'exactitude pendant la phase d'entraînement. Nous avons également étudié la frontière de décision du modèle, ce qui nous a permis d'acquérir une compréhension approfondie de l'apprentissage du modèle linéaire.

## Analyse du Réseau Non-Linéaire
Dans le cas du réseau non-linéaire, nous avons abordé le problème du XOR en utilisant un réseau de neurones à une couche cachée. Nous avons déterminé les paramètres optimaux du réseau en employant une heatmap pour identifier le meilleur couple (nombre d'itérations, learning rate) pour un réseau de neurones à une couche cachée comprenant 30 neurones. Ensuite, nous avons examiné l'influence du nombre de neurones dans la couche cachée sur les performances du modèle, en maintenant les hyperparamètres optimaux précédemment déterminés. Nous avons également recalculé la heatmap, obtenant ainsi des résultats considérablement améliorés, tout en conservant le même learning rate optimal et en réduisant le nombre d'itérations minimales requises pour obtenir d'excellentes performances, ce qui a permis d'optimiser l'efficacité computationnelle du modèle.

## Encapsulation et Classification Multi-Classe
Dans la dernière partie de notre analyse, nous avons exploré les concepts d'encapsulation et de classification multi-classe. Nous avons optimisé un réseau de neurones pour la classification de plusieurs classes, déterminant le nombre optimal de couches nécessaires pour classifier n classes. De plus, nous avons comparé les performances des fonctions d'activation LogSoftmax et Softmax dans ce contexte.

## Conclusion
En résumé, cette analyse approfondie nous a offert une meilleure compréhension de l'impact des différents hyperparamètres et architectures sur les performances des réseaux neuronaux. Ces connaissances peuvent être mises à profit pour améliorer la conception et l'optimisation des modèles d'apprentissage automatique dans une multitude de domaines d'application.