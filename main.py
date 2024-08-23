import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objs as go
from sklearn.datasets import make_blobs
from kmeans import KMeans




st.markdown("[by Achraf Hajji](https://www.linkedin.com/in/achraf-hajji-a271ab241/)")

# Interface Streamlit
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #f0f2f6;  /* Couleur de fond de la page */
    }
    .sidebar .sidebar-content {
        background-color: #1f1f1f;  /* Couleur de fond de la barre latérale */
        color: #ffffff;  /* Couleur du texte dans la barre latérale */
    }
    .stButton button {
        color: #ffffff;  /* Couleur du texte des boutons */
        background-color: #007bff;  /* Couleur de fond des boutons */
        border-radius: 5px;  /* Bordures arrondies des boutons */
    }
    .stTextInput input, .stNumberInput input {
        background-color: #ffffff;  /* Couleur de fond des champs de saisie */
        color: #000000;  /* Couleur du texte dans les champs de saisie */
    }
    .stSelectbox select {
        background-color: #ffffff;  /* Couleur de fond des listes déroulantes */
        color: #000000;  /* Couleur du texte dans les listes déroulantes */
    }
    h1 {
        color: #007bff;  /* Couleur des titres */
        font-size: 2.5em;
    }
    h2 {
        color: #0056b3;  /* Couleur des sous-titres */
        font-size: 2em;
    }
    h3 {
        color: #003d7a;  /* Couleur des sous-sous-titres */
        font-size: 1.5em;
    }
    p, li {
        color: #333333;  /* Couleur du texte normal */
        font-size: 1.2em;
    }
    .stImage img {
        border-radius: 10px;  /* Bordures arrondies des images */
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);  /* Ombre portée pour les images */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Création des onglets
tab1, tab2, tab3 = st.tabs(["Présentation", "Clustering K-Means", "Importer vos données"])

# Onglet Présentation
with tab1:
    st.title("Présentation de l'algorithme K-Means")
    st.header("Introduction")
    st.write("""
    K-Means est un algorithme d'apprentissage non supervisé couramment utilisé pour regrouper des points de données similaires en clusters.
    Il s'efforce de minimiser la distance moyenne entre les points géométriques en partitionnant itérativement les ensembles de données
    en un nombre fixe de sous-groupes non chevauchants (les clusters) où chaque point appartient au cluster avec le centre de cluster moyen le plus proche.
    """)
    st.header("Pourquoi K-Means ?")
    st.write("""
    K-Means est utilisé pour découvrir des groupes qui n'ont pas été explicitement étiquetés dans les données. Voici quelques applications courantes :
    - **Segmentation des clients** : Les clients peuvent être regroupés pour mieux adapter les produits et offres.
    - **Clustering de texte, documents ou résultats de recherche** : Grouper pour trouver des sujets dans les textes.
    - **Regroupement d'images ou compression d'images** : Regrouper les images similaires ou les couleurs.
    - **Détection d'anomalies** : Identifier les outliers des clusters.
    - **Apprentissage semi-supervisé** : Combiner les clusters avec un petit ensemble de données étiquetées pour obtenir des résultats plus précieux.
    """)
    st.header("Comment fonctionne K-Means ?")
    st.write("""
    L'algorithme K-Means identifie un certain nombre de centroïdes dans un ensemble de données, un centroïde étant la moyenne arithmétique
    de tous les points de données appartenant à un cluster particulier. L'algorithme attribue ensuite chaque point de données au cluster le plus proche
    en essayant de garder les clusters aussi petits que possible.
    """)
    st.subheader("Étapes pratiques :")
    st.write("1. **Initialisation** : L'algorithme commence par initialiser les coordonnées des centres de clusters à 'K'. Le nombre K est une variable d'entrée et les positions peuvent également être données en entrée.")
    st.image("images/10.jpg")
    st.write("2. **Assignation** : Avec chaque passage de l'algorithme, chaque point est assigné à son centre de cluster le plus proche.")
    st.image("images/11.jpg")
    st.write("3. **Mise à jour** : Les centres de cluster sont ensuite mis à jour pour être les 'centres' de tous les points qui y sont assignés à ce passage. Cela se fait en recalculant les centres de cluster comme la moyenne des points dans chaque cluster respectif.")
    st.image("images/12.jpg")
    st.write("4. **Répétition** : L'algorithme se répète jusqu'à ce qu'il y ait un changement minimal des centres de cluster par rapport à la dernière itération.")
    st.header("Limitations de K-Means")
    st.write("""
    - **Forme des clusters** : K-Means est très efficace pour capturer la structure et faire des inférences de données si les clusters ont une forme sphérique uniforme. Mais si les clusters ont des formes géométriques plus complexes, l'algorithme fait un mauvais travail de clustering des données.
    - **Nombre de clusters** : K-Means n'apprend pas lui-même le nombre de clusters à partir des données, mais cette information doit être prédéfinie.
    - **Clusters chevauchants** : K-Means ne peut pas déterminer comment assigner les points de données où le chevauchement se produit.
    """)
    st.header("Conclusion")
    st.write("""
    K-Means est un algorithme simple et efficace pour les tâches de clustering simples, mais il peut nécessiter des algorithmes plus sophistiqués pour les structures de données complexes.
    """)

# Onglet Clustering K-Means
with tab2:
    st.title("K-Means Clustering")

    n_clusters = st.number_input("Entrez le nombre de clusters :", min_value=1, max_value=10, value=1)
    max_iter = st.number_input("Entrez le nombre de répétitions :", min_value=100, max_value=500, value=100)
    distance_metric = st.selectbox("Choisissez la métrique de distance :", ("euclidean", "manhattan"))

    X, _ = make_blobs(n_samples=300, centers=n_clusters, random_state=42, cluster_std=1.0)

    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, distance_metric=distance_metric)
    centroids, labels = kmeans.fit(X)

    # Graphique 2D avec Plotly
    fig_2d = go.Figure()

    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow']

    for i in range(n_clusters):
        fig_2d.add_trace(go.Scatter(
            x=X[labels == i + 1, 0],
            y=X[labels == i + 1, 1],
            mode='markers',
            marker=dict(color=colors[i % len(colors)], size=10),
            name=f'Cluster {i + 1}'
        ))

    fig_2d.add_trace(go.Scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode='markers',
        marker=dict(color='red', size=12, symbol='cross'),
        name='Centroids'
    ))

    fig_2d.update_layout(title='K-Means Clustering 2D',
                          xaxis_title='Feature 1',
                          yaxis_title='Feature 2')

    st.plotly_chart(fig_2d)

    # Graphique 3D avec Plotly
    fig_3d = go.Figure()

    fig_3d.add_trace(go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2] if X.shape[1] > 2 else np.zeros(X.shape[0]),
        mode='markers',
        marker=dict(size=5, color=labels),
        name='Points'
    ))

    fig_3d.add_trace(go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2] if X.shape[1] > 2 else np.zeros(n_clusters),
        mode='markers',
        marker=dict(size=10, color='red', symbol='x'),
        name='Centroids'
    ))

    fig_3d.update_layout(title='K-Means Clustering 3D',
                         scene=dict(
                             xaxis_title='Feature 1',
                             yaxis_title='Feature 2',
                             zaxis_title='Feature 3' if X.shape[1] > 2 else 'Z'),
                         showlegend=True)

    st.plotly_chart(fig_3d)

# Onglet Importer vos données
with tab3:
    st.title("Importer vos données")

    uploaded_file = st.file_uploader("Importer un fichier Excel contenant les données", key='1', type=['xlsx', 'xls'])

    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        data.set_index(pd.Index(np.arange(1, len(data) + 1)), inplace=True)

        st.subheader("Les données importées :")
        st.write(data)

        x_axis_2d = st.selectbox("Sélectionnez l'axe pour X (2D) :", data.columns)
        y_axis_2d = st.selectbox("Sélectionnez l'axe pour Y (2D) :", data.columns)
        X_2d = data[[x_axis_2d, y_axis_2d]].values

        x_axis_3d = st.selectbox("Sélectionnez l'axe pour X (3D) :", data.columns)
        y_axis_3d = st.selectbox("Sélectionnez l'axe pour Y (3D) :", data.columns)
        z_axis_3d = st.selectbox("Sélectionnez l'axe pour Z (3D) :", data.columns)
        X_3d = data[[x_axis_3d, y_axis_3d, z_axis_3d]].values

        k_value = st.number_input("Entrez le nombre de clusters (k) :", min_value=1, value=2)
        max_iter = st.number_input("Entrez le nombre maximal d'itération :", min_value=100, value=100)
        distance_metric = st.selectbox("Sélectionnez le type de distance :", ['euclidean', 'manhattan'])

        kmeans_2d = KMeans(n_clusters=k_value, max_iter=max_iter, distance_metric=distance_metric)
        centroids_2d, labels_2d = kmeans_2d.fit(X_2d)

        kmeans_3d = KMeans(n_clusters=k_value, max_iter=max_iter, distance_metric=distance_metric)
        centroids_3d, labels_3d = kmeans_3d.fit(X_3d)

        centroids_df = pd.DataFrame(centroids_2d, columns=[x_axis_2d, y_axis_2d])
        centroids_df.index = np.arange(1, k_value + 1)

        st.subheader("Centroids finaux (2D) :")
        st.write(centroids_df)

        fig_2d, ax_2d = plt.subplots()
        scatter_2d = ax_2d.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_2d)
        legend_2d = ax_2d.legend(*scatter_2d.legend_elements(), title="Clusters")
        ax_2d.add_artist(legend_2d)
        ax_2d.scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker='*', c='red', label='Centroids')
        ax_2d.set_xlabel(x_axis_2d)
        ax_2d.set_ylabel(y_axis_2d)
        ax_2d.set_title('K-Means Clustering 2D')
        st.pyplot(fig_2d)

        # Graphique 3D avec Plotly
        fig_3d = go.Figure()

        fig_3d.add_trace(go.Scatter3d(
            x=X_3d[:, 0],
            y=X_3d[:, 1],
            z=X_3d[:, 2],
            mode='markers',
            marker=dict(size=5, color=labels_3d),
            name='Points'
        ))

        fig_3d.add_trace(go.Scatter3d(
            x=centroids_3d[:, 0],
            y=centroids_3d[:, 1],
            z=centroids_3d[:, 2],
            mode='markers',
            marker=dict(size=10, color='red', symbol='x'),
            name='Centroids'
        ))

        fig_3d.update_layout(title='K-Means Clustering 3D',
                             scene=dict(
                                 xaxis_title=x_axis_3d,
                                 yaxis_title=y_axis_3d,
                                 zaxis_title=z_axis_3d),
                             showlegend=True)

        st.plotly_chart(fig_3d)
