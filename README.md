from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np

class RobotPrediction:
    def __init__(self, equipe_a, equipe_b):
        self.equipe_a = equipe_a
        self.equipe_b = equipe_b
        self.scores_passes = []

    def entrer_scores_passes(self):
        nb_matchs = int(input("Entrez le nombre de matchs passés entre les équipes : "))
        for i in range(nb_matchs):
            score_a = int(input(f"Score de {self.equipe_a} dans le match {i + 1}: "))
            score_b = int(input(f"Score de {self.equipe_b} dans le match {i + 1}: "))
            self.scores_passes.append((score_a, score_b))

    def entrainer_modele_lineaire(self):
        if not self.scores_passes:
            return "Pas assez de données pour entraîner le modèle."

        X = np.array([score[0] for score in self.scores_passes]).reshape(-1, 1)
        y = np.array([score[1] for score in self.scores_passes])

        model = LinearRegression()
        model.fit(X, y)

        return model

    def entrainer_modele_knn(self):
        if not self.scores_passes:
            return "Pas assez de données pour entraîner le modèle."

        X = np.array([score[0] for score in self.scores_passes]).reshape(-1, 1)
        y = np.array([score[1] for score in self.scores_passes])

        model = KNeighborsRegressor(n_neighbors=3)  # Vous pouvez ajuster le nombre de voisins selon vos besoins
        model.fit(X, y)

        return model

    def faire_prediction(self, modele, score_a):
        score_b_pred = modele.predict(np.array(score_a).reshape(-1, 1))
        return int(score_b_pred[0])

    def afficher_resultat(self, score_a_predire):
        modele_lineaire = self.entrainer_modele_lineaire()
        modele_knn = self.entrainer_modele_knn()

        score_b_pred_lineaire = self.faire_prediction(modele_lineaire, score_a_predire)
        score_b_pred_knn = self.faire_prediction(modele_knn, score_a_predire)

        print(f"Prédiction (Modèle Linéaire): {self.equipe_a} {score_a_predire} - {self.equipe_b} {score_b_pred_lineaire}")
        print(f"Prédiction (Modèle KNN): {self.equipe_a} {score_a_predire} - {self.equipe_b} {score_b_pred_knn}")

# Exemple d'utilisation
equipe_a = input("Nom de l'équipe A : ")
equipe_b = input("Nom de l'équipe B : ")

robot = RobotPrediction(equipe_a, equipe_b)
robot.entrer_scores_passes()

# Entrer le score de l'équipe A pour faire une prédiction
score_a_predire = int(input(f"Entrez le score de {equipe_a} pour faire une prédiction : "))
robot.afficher_resultat(score_a_predire)

