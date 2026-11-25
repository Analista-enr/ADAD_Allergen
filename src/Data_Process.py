from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

"""
Cette classe à pour objectif de nettoyer les données : 

Pour chaque chips :
- groupement par meme chip
- suppression des colonnes avec + de 20% de valeurs manquantes
- mapping des variables qualitatives soient par map directe soit par encodage one-hot
- suppréssion des variables géographiques (une analyse plus poussée pourra être effectuée à ce moment la)
- calcul d’une première matrice de corrélation (spearman)
- Calcul d’une seconde matrice qui ne retient que les variables avec une corrélation avec + 0.5
- Passage dans l’IA pour analyser les résultats médicaux.
- Faire gaffe aux 9 dans l’âge et “Blood_Month_sample”

"""
PATH = r"data.csv"
TTINT = ["Treatment_of_athsma", "Age_of_onsets", "Treatment_of_atopic_dematitis"]


class Vulcain:
    def __init__(self):
        self.df = pd.read_csv(
            "data.csv",
            sep=";",
            encoding="utf-8-sig",  # -sig supprime le BOM proprement
            decimal=",",
            low_memory=False,
        )
        self.COLUMNS = list(self.df.columns)
        self.chip_dict, self.LEN = self.chips_df()
        self.CHIPS = list(self.df.Chip_Type.unique())

        self.pipeline()

    def chips_df(self):
        chip_dict = {}
        LEN = defaultdict(int)
        for chip in self.df.Chip_Type:
            chip_dict[chip] = self.df[self.df.Chip_Type == chip]
            LEN[chip] = len(chip_dict[chip])
        return chip_dict, LEN

    def cleaning(self, chip: str):
        df = self.chip_dict[chip]
        N = self.LEN[chip]
        for col in TTINT:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        df = self.one_hot(
            df, "General_cofactors"
        )  # C'est une liste de numéro de cofacteur qui sont présent
        # Nettoyage
        df = df[df.columns.difference(["Age", "Blood_Month_sample"])].replace(9, np.nan)
        dico_nan = {col: round(df[col].isna().sum() / N * 100, 2) for col in df.columns}
        COL_TO_SUPP = {col for col in df.columns if dico_nan[col] > 20}
        df.drop(COL_TO_SUPP, axis=1, inplace=True)
        # La y'a une question de comment on remplace les NaN
        STR_COL = df.select_dtypes(include=["object", "string"]).columns.tolist()
        df.drop(STR_COL, axis=1, inplace=True)
        return df

    def one_hot(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        # 1. Nettoyer les espaces éventuels autour des virgules puis faire le one-hot encoding
        one_hot = (
            df[col].str.replace(r"\s*,\s*", ",", regex=True).str.get_dummies(sep=",")
        )
        # 2. (Optionnel) Trier les colonnes dans l'ordre numérique : '0', '1', '2', ..., '10'
        one_hot = one_hot.reindex(sorted(one_hot.columns, key=int), axis=1)
        # 3. (Optionnel) Convertir les 0/1 en True/False (booléens) et ajouter un préfixe
        one_hot = one_hot.add_prefix("cofacteur_")
        # 4. Rattacher au DataFrame principal et supprimer l'ancienne colonne
        df = pd.concat([df.drop(columns=[col]), one_hot], axis=1)
        return df

    def pipeline(self):
        for chip in self.CHIPS:
            self.chip_dict[chip] = self.cleaning(chip)

    ### Plotting des figures

    def plot_matrix(self, df: pd.DataFrame):
        # 1. Sélection des colonnes numériques
        df_num = df.select_dtypes(include=np.number)

        # 2. Matrice de corrélation
        corr = df_num.corr(method="spearman")  # ou "spearman" (cf. remarques)

        # 3. Heatmap hiérarchisée (clustering des colonnes pour faire émerger les familles)
        # plt.figure(figsize=(14, 12))
        corr_clean = corr.dropna(axis=0, how="any").dropna(axis=1, how="any")

        # 2. Clustermap sur la matrice propre
        sns.clustermap(
            corr_clean,
            method="average",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            figsize=(15, 15),
        )
        plt.show()
        return corr_clean

    def detect_corr(self, M: pd.DataFrame, seuil: float = 0.3) -> dict:
        dicorr = defaultdict(list)
        for col in M.columns:
            dicorr[col] = [
                idx for idx in M.index if idx != col and abs(M.loc[idx, col]) > seuil
            ]
        return dict(dicorr)

    def plot_distributions(self, dfs, columns, path="distributions.pdf", min_n=10):
        """dfs : dict {nom_puce: DataFrame}. Un graphique par grandeur, une courbe par puce."""
        sns.set_theme(style="whitegrid", context="notebook", palette="Set2")
        with PdfPages(path) as pdf:
            for col in columns:
                # valeurs numériques uniquement, NaN et texte ignorés
                data = {
                    chip: pd.to_numeric(df[col], errors="coerce").dropna()
                    for chip, df in dfs.items()
                    if col in df.columns
                }
                data = {c: s for c, s in data.items() if len(s) >= min_n}
                if not data:
                    continue

                d = pd.concat(
                    [
                        pd.DataFrame(
                            {"x": np.log10(1 + s), "Puce": f"{c} (n={len(s)})"}
                        )
                        for c, s in data.items()
                    ]
                )

                fig, ax = plt.subplots(figsize=(7, 4))
                sns.kdeplot(
                    data=d,
                    x="x",
                    hue="Puce",
                    fill=True,
                    alpha=0.25,
                    common_norm=False,
                    cut=0,
                    linewidth=2,
                    ax=ax,
                )
                ax.set_title(col, fontweight="bold")
                ax.set_xlabel("log10(1 + valeur)")
                ax.set_ylabel("Densité")
                sns.despine()
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
