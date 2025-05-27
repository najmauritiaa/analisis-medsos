import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="Follower Network Analyzer", layout="wide")
st.title("📊 Follower Network Analysis")

uploaded_file = st.file_uploader("Upload file scraped_followers (.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if 'followee' in df.columns and 'follower' in df.columns:
            st.success("✅ Format file sesuai.")

            # Build Directed Graph
            G = nx.DiGraph()
            edges = list(zip(df['follower'], df['followee']))
            G.add_edges_from(edges)

            st.subheader("🎯 Centrality Metrics")

            degree = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G)
            closeness = nx.closeness_centrality(G)
            try:
                eigenvector = nx.eigenvector_centrality(G)
            except nx.NetworkXException:
                eigenvector = {node: 0 for node in G.nodes()}

            centrality_df = pd.DataFrame({
                "Node": list(G.nodes()),
                "Degree": [degree[n] for n in G.nodes()],
                "Betweenness": [betweenness[n] for n in G.nodes()],
                "Closeness": [closeness[n] for n in G.nodes()],
                "Eigenvector": [eigenvector[n] for n in G.nodes()]
            }).sort_values(by="Degree", ascending=False)

            st.dataframe(centrality_df, use_container_width=True)

            st.subheader("🕸️ Network Graph")
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=8)
            st.pyplot(plt)

        else:
            st.error("❌ Kolom harus terdiri dari 'followee' dan 'follower'.")

    except Exception as e:
        st.error(f"Terjadi error saat memproses file: {e}")
else:
    st.info("Silakan upload file CSV berisi data scraped follower.")
