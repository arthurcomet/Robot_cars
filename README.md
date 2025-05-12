# 🏎️ Robocar Racing Simulator

Bienvenue dans le **Robocar Racing Simulator**, un projet visant à initier au développement d'intelligences artificielles (IA) pour la conduite autonome, à travers une simulation réaliste de course automobile.

## 📌 Objectif

L'objectif principal est de développer une IA capable de piloter une voiture sur circuit de manière **rapide**, **précise** et **sûre**. Le projet met l'accent sur l'apprentissage supervisé à partir de données collectées lors de sessions de conduite humaine.

---

## 🧠 Composants du projet

Le projet est structuré en 4 modules principaux :

- **Client** : connecte l'IA au simulateur.
- **Gestion des entrées** : capture les commandes utilisateur (clavier/manette).
- **Collecteur de données** : enregistre les données nécessaires à l'entraînement.
- **Intelligence artificielle** : pilote le véhicule via les données apprises.

---

## 🚀 Lancer la simulation

1. Télécharger le simulateur via ce lien :
   [Télécharger le simulateur](https://epitechfr.sharepoint.com/:f:/s/CIME-Robocar/Ei8CSM13u6xNgNGS8du6PLkBgdEUL88lUdQ2UpZcN-ptCQ?e=XklBdA)

2. Exécuter le binaire `RacingSimulator`.

3. Dans l'interface, un menu en haut à gauche permet de changer de **vue** et de **circuit**.

4. Connecter ensuite votre client à l'adresse suivante :  
   `0.0.0.0:8085`

---

## 🔧 Protocole de communication client ↔ simulateur

| Commande                     | Description                                  |
|-----------------------------|----------------------------------------------|
| `SET_SPEED:[-1.0 à 1.0]`    | Définit la vitesse                           |
| `SET_STEERING:[-1.0 à 1.0]` | Définit l'angle du volant                    |
| `SET_NUMBER_RAY:[1 à 50]`   | Nombre de rayons de détection                |
| `GET_SPEED`                 | Retourne la vitesse actuelle                 |
| `GET_STEERING`              | Retourne l'angle de direction                |
| `GET_POSITION`              | Retourne la position de la voiture           |
| `GET_INFOS_RAYCAST`         | Retourne les données des capteurs            |

- Toutes les commandes doivent être entourées par `;`.  
- Plusieurs commandes peuvent être envoyées en une fois :  
  Exemple : `;SET_SPEED:0.5;SET_STEERING:-0.2;`

---

## 🎮 Système de gestion des entrées

- Capture les entrées clavier ou manette.
- Transmet les commandes en temps réel au serveur.
- Permet une conduite manuelle pour générer des données d'entraînement.

---

## 🧾 Collecte de données

- Enregistre :
  - Positions
  - Commandes
  - Informations capteurs
- Ces données servent à entraîner l'IA via apprentissage supervisé.
- Analyse exploratoire (EDA) recommandée pour valider la qualité des données.

---

## 🧠 Intelligence Artificielle

- **Modèle supervisé uniquement** (pas de reinforcement learning à ce stade).
- S'entraîne sur les données collectées.
- Objectif : suivre la piste efficacement **sans sortir** des limites.

> ✅ Une voiture lente mais stable est préférée à une rapide mais imprécise.

---

## 📈 Bonnes pratiques

- Priorisez :
  - Un **grand dataset propre** plutôt qu'un modèle complexe.
  - L'évaluation via **métriques objectives**.
- Visualiser la conduite ne suffit pas, il faut des **tests chiffrés**.

---

## 🛠️ Technologies

- **Langage** : Python
- **Communication réseau** : protocole texte sur port 8085
- **Environnement** : Simulation graphique dédiée

---

## 📄 Version

**v1.0.0**
