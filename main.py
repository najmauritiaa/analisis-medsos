import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load Data
st.title("Instagram Influencer & Community Detection Tool")

st.markdown("Paste the Instagram profile URL or username below:")
username = st.text_input("Instagram Username / URL", "")

if st.button("Analyze"):
    st.success(f"Analysis started for: {username}")

    # Step 1: Load scraped followers (simulasi dari file)
    try:
        followers_df = pd.read_csv("scraped_followers_1.csv")
        centrality_df = pd.read_csv("combined_centrality.csv")
    except FileNotFoundError:
        st.error("File data tidak ditemukan.")
        st.stop()

    st.subheader("Centrality Metrics")
    st.dataframe(centrality_df.head(10))

    # Step 2: Create Graph
    st.subheader("Network Graph Visualization")
    G = nx.Graph()

    for _, row in followers_df.iterrows():
        G.add_edge(row['source'], row['target'])

    # Plot the graph
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, k=0.15)
    nx.draw(G, pos, with_labels=False, node_size=20, edge_color='gray')
    st.pyplot(plt)

    # Step 3: Show Top Influencers
    st.subheader("Top 10 Influencers")
    top_influencers = centrality_df.sort_values("composite_centrality", ascending=False).head(10)
    st.table(top_influencers[['username', 'composite_centrality']])

    # Step 4: Clustering (dummy example using degree)
    st.subheader("Community Detection with K-Means")
    degrees = dict(G.degree())
    df_deg = pd.DataFrame(list(degrees.items()), columns=["username", "degree"])

    km = KMeans(n_clusters=3, random_state=42).fit(df_deg[['degree']])
    df_deg['cluster'] = km.labels_

    st.write(df_deg.head(10))

    # Future: Color-code graph by cluster here using community detection
