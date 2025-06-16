import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import community as community_louvain
from streamlit_agraph import agraph, Node, Edge, Config
import numpy as np
import re
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="Advanced Follower Network Analyzer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .community-stats {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚀 Advanced Follower Network Analysis</h1>', unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.header("⚙️ Analysis Configuration")
show_labels = st.sidebar.checkbox("Show node labels in graph", value=False)
node_size_metric = st.sidebar.selectbox(
    "Node size based on:",
    ["Degree Centrality", "Betweenness Centrality", "Closeness Centrality", "PageRank"]
)
filter_isolated = st.sidebar.checkbox("Filter isolated nodes", value=True)
min_edges = st.sidebar.slider("Minimum edges for visualization", 1, 10, 2)

uploaded_file = st.file_uploader("📁 Upload your scraped_followers (.csv)", type=["csv"])

def clean_data(df):
    """Enhanced data cleaning function"""
    original_size = len(df)
    
    # Remove Instagram URLs and malformed entries
    df = df[~df['follower'].astype(str).str.contains("www.instagram.com", na=False)]
    df = df[~df['follower'].astype(str).str.contains(r'\?xmt=', na=False)]
    df = df[~df['follower'].astype(str).str.contains("http", na=False)]
    
    # Remove null values
    df = df.dropna(subset=['follower', 'followee'])
    
    # Remove self-loops
    df = df[df['follower'] != df['followee']]
    
    # Remove duplicate edges
    df = df.drop_duplicates()
    
    cleaned_size = len(df)
    
    return df, original_size, cleaned_size

def calculate_advanced_metrics(G):
    """Calculate comprehensive network metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['degree'] = nx.degree_centrality(G)
    metrics['closeness'] = nx.closeness_centrality(G)
    metrics['betweenness'] = nx.betweenness_centrality(G)
    
    # Advanced metrics
    metrics['pagerank'] = nx.pagerank(G)
    metrics['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000)
    
    # In-degree and out-degree for directed graphs
    if G.is_directed():
        metrics['in_degree'] = dict(G.in_degree())
        metrics['out_degree'] = dict(G.out_degree())
    
    return metrics

def create_centrality_dataframe(metrics, G):
    """Create comprehensive centrality DataFrame"""
    nodes = list(G.nodes())
    
    df_data = {
        'Node': nodes,
        'Degree_Centrality': [metrics['degree'].get(node, 0) for node in nodes],
        'Closeness_Centrality': [metrics['closeness'].get(node, 0) for node in nodes],
        'Betweenness_Centrality': [metrics['betweenness'].get(node, 0) for node in nodes],
        'PageRank': [metrics['pagerank'].get(node, 0) for node in nodes],
        'Eigenvector_Centrality': [metrics['eigenvector'].get(node, 0) for node in nodes]
    }
    
    if G.is_directed():
        df_data['In_Degree'] = [metrics['in_degree'].get(node, 0) for node in nodes]
        df_data['Out_Degree'] = [metrics['out_degree'].get(node, 0) for node in nodes]
    
    df = pd.DataFrame(df_data)
    
    # Calculate composite score
    centrality_cols = ['Degree_Centrality', 'Closeness_Centrality', 'Betweenness_Centrality', 'PageRank']
    df['Composite_Score'] = df[centrality_cols].mean(axis=1)
    df = df.sort_values('Composite_Score', ascending=False)
    
    return df

def create_interactive_network_plotly(G, partition, centrality_df):
    """Create interactive network visualization using Plotly"""
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_info.append(f"{edge[0]} → {edge[1]}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Prepare node traces
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node info
        degree = G.degree(node)
        community = partition.get(node, 0)
        centrality_score = centrality_df[centrality_df['Node'] == node]['Composite_Score'].iloc[0] if len(centrality_df[centrality_df['Node'] == node]) > 0 else 0
        
        node_text.append(f"User: {node}<br>Degree: {degree}<br>Community: {community}<br>Centrality: {centrality_score:.3f}")
        node_colors.append(community)
        
        # Size based on selected metric
        if node_size_metric == "Degree Centrality":
            size_value = centrality_df[centrality_df['Node'] == node]['Degree_Centrality'].iloc[0] if len(centrality_df[centrality_df['Node'] == node]) > 0 else 0
        elif node_size_metric == "Betweenness Centrality":
            size_value = centrality_df[centrality_df['Node'] == node]['Betweenness_Centrality'].iloc[0] if len(centrality_df[centrality_df['Node'] == node]) > 0 else 0
        elif node_size_metric == "Closeness Centrality":
            size_value = centrality_df[centrality_df['Node'] == node]['Closeness_Centrality'].iloc[0] if len(centrality_df[centrality_df['Node'] == node]) > 0 else 0
        else:  # PageRank
            size_value = centrality_df[centrality_df['Node'] == node]['PageRank'].iloc[0] if len(centrality_df[centrality_df['Node'] == node]) > 0 else 0
        
        node_sizes.append(max(10, size_value * 100))
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text' if show_labels else 'markers',
        text=[node for node in G.nodes()] if show_labels else None,
        textposition="middle center",
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Community",
                thickness=15,
                len=0.5
            ),
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=dict(
                           text=f'Interactive Network Graph (Node size: {node_size_metric})',
                           x=0.5,
                           font=dict(size=20)
                       ),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Hover over nodes for details. Zoom and pan to explore.",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor='left', yanchor='bottom',
                           font=dict(color="#888", size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='white'
                   ))
    
    return fig

def create_agraph_visualization(G, partition):
    """Create interactive network using streamlit-agraph"""
    # Limit nodes for better performance
    if len(G.nodes()) > 100:
        # Get top nodes by degree
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:100]
        subgraph_nodes = [node for node, _ in top_nodes]
        G_sub = G.subgraph(subgraph_nodes)
    else:
        G_sub = G
    
    nodes = []
    edges = []
    
    # Color map for communities
    communities = set(partition.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF']
    community_colors = {comm: colors[i % len(colors)] for i, comm in enumerate(communities)}
    
    # Add nodes
    for node in G_sub.nodes():
        degree = G_sub.degree(node)
        community = partition.get(node, 0)
        color = community_colors.get(community, '#95A5A6')
        
        nodes.append(Node(
            id=str(node), 
            label=str(node) if show_labels else "",
            size=max(10, degree * 3),
            color=color,
            title=f"User: {node}\nDegree: {degree}\nCommunity: {community}"
        ))
    
    # Add edges
    for edge in G_sub.edges():
        edges.append(Edge(
            source=str(edge[0]), 
            target=str(edge[1]),
            color='#95A5A6'
        ))
    
    config = Config(
        width=800,
        height=600,
        directed=G.is_directed(),
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False
    )
    
    return agraph(nodes=nodes, edges=edges, config=config)

if uploaded_file is not None:
    try:
        # Load and clean data
        df = pd.read_csv(uploaded_file)
        
        if 'followee' in df.columns and 'follower' in df.columns:
            df_clean, original_size, cleaned_size = clean_data(df)
            
            # Data quality metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Original Records", f"{original_size:,}")
            with col2:
                st.metric("✅ Clean Records", f"{cleaned_size:,}")
            with col3:
                st.metric("🗑️ Removed Records", f"{original_size - cleaned_size:,}")
            with col4:
                st.metric("📈 Data Quality", f"{(cleaned_size/original_size)*100:.1f}%")
            
            # Build graph
            G = nx.from_pandas_edgelist(df_clean, source='followee', target='follower', create_using=nx.DiGraph())
            
            # Filter isolated nodes if requested
            if filter_isolated:
                isolated_nodes = list(nx.isolates(G))
                G.remove_nodes_from(isolated_nodes)
            if min_edges > 1:
                low_degree_nodes = [node for node, degree in dict(G.degree()).items() if degree < min_edges]
                G.remove_nodes_from(low_degree_nodes)
            # Basic network statistics
            st.markdown("---")
            st.subheader("🔍 Network Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("👥 Total Nodes", len(G.nodes()))
            with col2:
                st.metric("🔗 Total Edges", len(G.edges()))
            with col3:
                st.metric("📊 Density", f"{nx.density(G):.4f}")
            with col4:
                if len(G.nodes()) > 0:
                    avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
                    st.metric("📈 Avg Degree", f"{avg_degree:.2f}")
            
            # Calculate metrics
            with st.spinner("🔄 Calculating network metrics..."):
                metrics = calculate_advanced_metrics(G)
                centrality_df = create_centrality_dataframe(metrics, G)
            
            # Centrality Analysis
            st.markdown("---")
            st.subheader("🎯 Centrality Analysis")
            
            # Top influencers
            st.markdown("### 🏆 Top Influencers")
            top_n = st.slider("Number of top influencers to show:", 5, 20, 10)
            st.dataframe(centrality_df.head(top_n), use_container_width=True)
            
            # Centrality correlation heatmap
            st.markdown("### 📊 Centrality Metrics Correlation")
            centrality_cols = ['Degree_Centrality', 'Closeness_Centrality', 'Betweenness_Centrality', 'PageRank', 'Eigenvector_Centrality']
            corr_matrix = centrality_df[centrality_cols].corr()
            
            fig_heatmap = px.imshow(
                corr_matrix,
                title="Centrality Metrics Correlation Matrix",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            fig_heatmap.update_layout(width=800, height=600)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Centrality distribution plots
            st.markdown("### 📈 Centrality Distributions")
            
            fig_dist = make_subplots(
                rows=2, cols=3,
                subplot_titles=('Degree', 'Closeness', 'Betweenness', 'PageRank', 'Eigenvector', 'Composite'),
                specs=[[{"secondary_y": False}]*3, [{"secondary_y": False}]*3]
            )
            
            for i, col in enumerate(['Degree_Centrality', 'Closeness_Centrality', 'Betweenness_Centrality', 'PageRank', 'Eigenvector_Centrality', 'Composite_Score']):
                row = (i // 3) + 1
                col_pos = (i % 3) + 1
                
                fig_dist.add_trace(
                    go.Histogram(x=centrality_df[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig_dist.update_layout(height=600, title_text="Centrality Distributions")
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Community Detection
            st.markdown("---")
            st.subheader("🧩 Community Detection Analysis")
            
            with st.spinner("🔄 Detecting communities..."):
                G_undirected = G.to_undirected()
                partition = community_louvain.best_partition(G_undirected)
                modularity = community_louvain.modularity(partition, G_undirected)
            
            # Community statistics
            community_sizes = pd.Series(partition).value_counts().sort_index()
            num_communities = len(community_sizes)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏘️ Communities Found", num_communities)
            with col2:
                st.metric("📊 Modularity Score", f"{modularity:.3f}")
            with col3:
                st.metric("👥 Largest Community", f"{community_sizes.max()} members")
            
            # Community size distribution
            fig_comm = px.bar(
                x=community_sizes.index,
                y=community_sizes.values,
                title="Community Size Distribution",
                labels={'x': 'Community ID', 'y': 'Number of Members'}
            )
            fig_comm.update_layout(showlegend=False)
            st.plotly_chart(fig_comm, use_container_width=True)
            
            # Network Visualizations
            st.markdown("---")
            st.subheader("🕸️ Network Visualizations")
            
            viz_option = st.radio(
                "Choose visualization type:",
                ["Interactive Plotly Graph", "Force-Directed Graph (AGraph)", "Static Community Plot"]
            )
            
            if viz_option == "Interactive Plotly Graph":
                st.markdown("### 🚀 Interactive Network (Plotly)")
                fig_network = create_interactive_network_plotly(G_undirected, partition, centrality_df)
                st.plotly_chart(fig_network, use_container_width=True)
                
            elif viz_option == "Force-Directed Graph (AGraph)":
                st.markdown("### 🌐 Force-Directed Interactive Graph")
                st.info("This graph is fully interactive - you can drag nodes, zoom, and explore!")
                create_agraph_visualization(G_undirected, partition)
                
            else:  # Static Community Plot
                st.markdown("### 🎨 Static Community Visualization")
                fig, ax = plt.subplots(figsize=(15, 10))
                pos = nx.spring_layout(G_undirected, seed=42, k=1, iterations=50)
                
                # Draw nodes colored by community
                communities = set(partition.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
                
                for i, community in enumerate(communities):
                    nodes_in_community = [node for node, comm in partition.items() if comm == community]
                    nx.draw_networkx_nodes(
                        G_undirected, pos,
                        nodelist=nodes_in_community,
                        node_color=[colors[i]],
                        node_size=[centrality_df[centrality_df['Node'] == node]['Composite_Score'].iloc[0] * 500 + 50 for node in nodes_in_community],
                        alpha=0.8,
                        label=f'Community {community}'
                    )
                
                nx.draw_networkx_edges(G_undirected, pos, alpha=0.3, edge_color='gray')
                
                if show_labels and len(G_undirected.nodes()) < 50:
                    nx.draw_networkx_labels(G_undirected, pos, font_size=8)
                
                plt.title("Network Graph with Community Detection", size=16, pad=20)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.axis('off')
                st.pyplot(fig, bbox_inches='tight')
            
            # Export Options
            st.markdown("---")
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Download Centrality Data"):
                    csv = centrality_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="centrality_analysis.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("🏘️ Download Community Data"):
                    community_df = pd.DataFrame(list(partition.items()), columns=['Node', 'Community'])
                    csv = community_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="community_detection.csv",
                        mime="text/csv"
                    )
            
        else:
            st.error("❌ CSV file must contain 'followee' and 'follower' columns.")
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)
else:
    st.info("👆 Please upload a CSV file containing scraped follower data to begin analysis.")
    
    # Sample data format
    st.markdown("---")
    st.subheader("📋 Expected Data Format")
    sample_data = pd.DataFrame({
        'followee': ['user1', 'user2', 'user3'],
        'follower': ['follower1', 'follower2', 'follower3']
    })
    st.dataframe(sample_data)
    st.caption("Your CSV should have 'followee' and 'follower' columns with user relationships.")
