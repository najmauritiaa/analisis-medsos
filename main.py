import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import re

st.set_page_config(page_title="Follower Network Analyzer", layout="wide")
st.title("📊 Follower Network Analysis")

uploaded_file = st.file_uploader("Upload file scraped_followers (.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        # Load and clean data
        df = pd.read_csv(uploaded_file)
        if 'followee' in df.columns and 'follower' in df.columns:
            # Bersihkan data
            df = df[~df['follower'].astype(str).str.contains("www.instagram.com", na=False)]
            df = df[~df['follower'].astype(str).str.contains(r'\?xmt=', na=False)]

            st.success("✅ Data berhasil dibersihkan dan dimuat.")

            # Build directed graph
            G = nx.from_pandas_edgelist(df, source='followee', target='follower', create_using=nx.DiGraph())

            # Hitung centrality metrics
            st.subheader("🎯 Centrality Metrics")
            degree = nx.degree_centrality(G)
            closeness = nx.closeness_centrality(G)
            betweenness = nx.betweenness_centrality(G)

            centrality_df = pd.DataFrame({
                "Node": list(degree.keys()),
                "Degree": list(degree.values()),
                "Closeness": [closeness[n] for n in degree.keys()],
                "Betweenness": [betweenness[n] for n in degree.keys()]
            })
            centrality_df["Average"] = centrality_df[["Degree", "Closeness", "Betweenness"]].mean(axis=1)
            centrality_df = centrality_df.sort_values(by="Average", ascending=False)

            st.dataframe(centrality_df, use_container_width=True)

            # Deteksi komunitas dengan Louvain
            st.subheader("🧩 Community Detection (Louvain)")
            G_undirected = G.to_undirected()
            partition = community_louvain.best_partition(G_undirected)
            community_df = pd.DataFrame(list(partition.items()), columns=["Node", "Community"])

            st.write(f"Jumlah komunitas terdeteksi: {len(set(partition.values()))}")
            st.dataframe(community_df.value_counts("Community").reset_index(name="Jumlah Anggota"), use_container_width=True)

            # Visualisasi jaringan dengan komunitas
            st.subheader("🕸️ Network Graph")
            pos = nx.spring_layout(G_undirected, seed=42)
            cmap = plt.get_cmap("viridis")
            num_comms = max(partition.values()) + 1
            colors = [cmap(partition[node] / num_comms) for node in G_undirected.nodes()]

            plt.figure(figsize=(12, 8))
            nx.draw_networkx_nodes(G_undirected, pos, node_color=colors, node_size=50, alpha=0.8)
            nx.draw_networkx_edges(G_undirected, pos, alpha=0.5)
            plt.title("Community Detection using Louvain Algorithm")
            plt.axis("off")
            st.pyplot(plt)

        else:
            st.error("❌ Kolom harus terdiri dari 'followee' dan 'follower'.")

    except Exception as e:
        st.error(f"Terjadi error saat memproses file: {e}")
else:
    st.info("Silakan upload file CSV berisi data scraped follower.")
