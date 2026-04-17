import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.title("Análisis de ECOBICI 🚲")

# =========================
# SUBIR VARIOS ARCHIVOS
# =========================
uploaded_files = st.file_uploader(
    "Sube tus archivos CSV (enero, febrero, marzo)",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:

    dfs = []

    for file in uploaded_files:
        temp = pd.read_csv(file)

        # Guardar nombre del archivo (mes)
        temp['source_file'] = file.name

        dfs.append(temp)

    # Unir todos los datos
    df = pd.concat(dfs, ignore_index=True)

    st.subheader("Datos combinados")
    st.write(df.head())

    # =========================
    # LIMPIEZA
    # =========================
    df['Fecha_Retiro'] = df['Fecha_Retiro'].astype(str).str.strip()
    df['Hora_Retiro'] = df['Hora_Retiro'].astype(str).str.strip()

    df['datetime'] = pd.to_datetime(
        df['Fecha_Retiro'] + ' ' + df['Hora_Retiro'],
        dayfirst=True,
        errors='coerce'
    )

    df = df.dropna(subset=['datetime'])

    # Variables
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day_name()
    df['month'] = df['datetime'].dt.month

    # =========================
    # VIAJES POR HORA
    # =========================
    st.subheader("Viajes por hora")

    trips_hour = df.groupby('hour').size()

    fig1, ax1 = plt.subplots()
    ax1.plot(trips_hour.index, trips_hour.values)
    ax1.set_title("Viajes por hora")
    ax1.set_xlabel("Hora")
    ax1.set_ylabel("Número de viajes")

    st.pyplot(fig1)

    # =========================
    # VIAJES POR DÍA
    # =========================
    st.subheader("Viajes por día")

    order_days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    trips_day = df['day'].value_counts().reindex(order_days)

    fig2, ax2 = plt.subplots()
    sns.barplot(x=trips_day.index, y=trips_day.values, ax=ax2)
    ax2.set_title("Viajes por día")
    ax2.tick_params(axis='x', rotation=45)

    st.pyplot(fig2)

    # =========================
    # HEATMAP
    # =========================
    st.subheader("Heatmap día vs hora")

    heatmap = df.groupby(['day','hour']).size().unstack(fill_value=0)
    heatmap = heatmap.reindex(order_days)

    fig3, ax3 = plt.subplots()
    sns.heatmap(heatmap, cmap="coolwarm", ax=ax3)
    ax3.set_title("Uso por día y hora")

    st.pyplot(fig3)

    # =========================
    # MÉTODO DEL CODO
    # =========================
    st.subheader("Método del codo")

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(heatmap)

    inertia = []
    k_values = range(1, 8)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)
        inertia.append(kmeans.inertia_)

    fig4, ax4 = plt.subplots()
    ax4.plot(k_values, inertia, marker='o')
    ax4.set_title("Elbow Method")
    ax4.set_xlabel("k")
    ax4.set_ylabel("Inercia")

    st.pyplot(fig4)

    # =========================
    # K-MEANS FINAL
    # =========================
    st.subheader("Clusters con PCA")

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    fig5, ax5 = plt.subplots()
    scatter = ax5.scatter(data_pca[:,0], data_pca[:,1], c=clusters)
    ax5.set_title("Clusters (PCA)")

    st.pyplot(fig5)
