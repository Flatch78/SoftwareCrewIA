# 🧾 **SoftwareCrewIA**


### **Le copilote IA qui réconcilie les équipes agiles**

## 🎯 **Présentation du projet**

Ce projet a pour objectif de créer un système capable de **générer automatiquement un titre et une description d'une User Story**, à partir d’un texte d’entrée.

Il repose sur un modèle entraîné (fine tuning) localement, puis utilisés via une API backend et une simple interface web.

L'ensemble du projet est pensé pour fonctionner **en local**, via **Docker**, sous la forme d’un **monorepo** contenant :

* le backend (API + modèle entraînement)
* le frontend (client web)

---

# 🏗️ **Architecture du projet**

## 📦 Monorepo

Le dépôt contient deux applications :

```
/frontend     → Interface web
/backend      → API + Entraînement + Utilisation des modèles
```

Le backend regroupe :

* 📚 **l’entraînement des modèles**
* 🧠 **l’inférence (utilisation des modèles entraînés)**
* 🔌 **l’API permettant l’accès au modèle**
* 🐳 **la configuration Docker pour une utilisation locale**

Le frontend est indépendant, mais peut être lancé dans le même monorepo.

---

# 🤖 **Objectifs ML**

Le modèle devra être capable de :

1. **Analyser une entrée utilisateur**
2. **Générer un titre de User Story** (ex : "En tant que client, je peux…")
3. **Générer une description claire et structurée**, par exemple :

   * Contexte
   * Besoin
   * Critères d’acceptation

Le projet pourra inclure :

* du pré-processing
* de la vectorisation
* un modèle complexe NLP (ex : GPT, Bert, etc.)

L’objectif n’est **pas** de reproduire des modèles géants, mais de construire une **pipeline fonctionnelle et reproductible**.

---

# 🧰 **Mise en place**

### ✔ Le projet contient :

* **un module d’entraînement**
  `backend/app/train/`
* **une API FastAPI**
  `backend/app/api/`
* **un frontend web** (Python - streamlit)
  `frontend/`
* **des conteneurs Docker**
  `docker-compose.yml` `backend/Dockerfile` `frontend/Dockerfile`

### ✔ La première version :

* fonctionne **exclusivement en local**
* n’est **pas déployée sur un serveur**
* utilise Docker pour simplifier l'exécution

---

# 🚀 **Démarrage rapide (local)**

```bash
docker-compose up --build
```

Lancement typique :

* API disponible sur : `http://localhost:4242`
* Frontend disponible sur : `http://localhost:4241`

---

# 🧪 **Flux général du projet**

1. 📄 **Import _déjà disponible dans `backend/app/data/raw`_ et préparation des données**
2. 🏋️ **Entraînement du modèle**
3. 💾 **Enregistrement du modèle** (joblib)
4. 🔌 **API pour servir le modèle**
5. 🖥️ **Frontend pour utilisation via une interface simple**

---

# 🔍 **To-do / Prochaines étapes**

* [x] Déterminer l’outil Git (GitHub / Bitbucket)
* [ ] Choisir le type de modèle (classique NLP ou petit transformer local)
* [ ] Construire le pipeline d’entraînement
* [x] Créer l’API backend
* [x] Développer l’interface frontend
* [x] Créer les images Docker
* [ ] Rédiger la documentation technique

---
