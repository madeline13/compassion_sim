k = 10

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import seaborn as sns
# Clustering imports
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="Mutual Aid in Disaster Simulation", layout="wide")
st.title("Mutual Aid and Its Effects on a Disaster-Affected Population")

# Sidebar controls for population and simulation
n_large = st.sidebar.checkbox("Enable Large Population (10,000+)", value=False)
if n_large:
    N = st.sidebar.slider("Population Size (N)", min_value=1000, max_value=50000, value=10000, step=1000)
else:
    N = st.sidebar.slider("Population Size (N)", min_value=100, max_value=2000, value=1000, step=100)
n_steps = st.sidebar.slider("Simulation Steps", min_value=10, max_value=100, value=20, step=5)
harmful_fraction = st.sidebar.slider("Fraction Harmful Members", min_value=0.0, max_value=0.5, value=0.15, step=0.01)

np.random.seed(42)  # Ensures reproducibility for large populations
needs = [
    'food', 'water', 'shelter', 'medical', 'sanitation', 'psychological', 'education', 'security', 'transport', 'communication'
]

# Generate population data
ids = np.arange(N)
ages = np.random.randint(0, 90, N)
vuln_choices = np.random.choice(['low', 'medium', 'high'], size=N, p=[0.3, 0.5, 0.2])
locations = np.random.choice(['camp', 'urban', 'rural'], size=N)
needs_list = [np.random.choice(needs, size=np.random.randint(1, 4), replace=False).tolist() for _ in range(N)]
df = pd.DataFrame({
    'id': ids,
    'age': ages,
    'vulnerability': vuln_choices,
    'location': locations,
    'needs': needs_list
})

# Initial needs: granular by vulnerability
person_needs = np.ones((N, len(needs)))
for i in range(N):
    vuln = df.loc[i, 'vulnerability']
    for j, need in enumerate(needs):
        if need in df.loc[i, 'needs']:
            if vuln == 'high':
                person_needs[i, j] = np.random.choice([0, 0.25])
            elif vuln == 'medium':
                person_needs[i, j] = np.random.choice([0.25, 0.5])
            else:
                person_needs[i, j] = np.random.choice([0.5, 0.75])

# Mutual aid matrix: can help if not highly vulnerable and shares location or need
can_help_matrix = np.zeros((N, N), dtype=bool)
vuln_mask = df['vulnerability'] != 'high'
for i in range(N):
    if not vuln_mask[i]:
        continue
    loc_i = df.loc[i, 'location']
    needs_i = set(df.loc[i, 'needs'])
    for j in range(N):
        if i == j:
            continue
        shared_location = loc_i == df.loc[j, 'location']
        shared_need = bool(needs_i & set(df.loc[j, 'needs']))
        if shared_location or shared_need:
            can_help_matrix[i, j] = True

# Harmful members
harmful_indices = np.random.choice(N, int(N * harmful_fraction), replace=False)

# Simulation: semi-random walks + mutual aid
walk_history = np.zeros((N, n_steps, len(needs)))
walk_history[:, 0, :] = person_needs
for step in range(1, n_steps):
    for i in range(N):
        random_step = np.random.normal(loc=0.05, scale=0.02, size=len(needs))
        walk_history[i, step, :] = np.clip(walk_history[i, step-1, :] + random_step, 0, 1)


# --- PCA + KMeans Clustering: Find 10 maximally different groups ---
# Use initial needs as features for clustering (or use final_needs if you want post-aid clusters)
features_for_clustering = person_needs
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_for_clustering)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(features_2d)
df['cluster'] = cluster_labels

# Model finite resources for each need (e.g., food, water)
finite_resources = np.random.randint(N//2, N, size=len(needs))
st.sidebar.write("Finite Tangible Resources (per need):")
st.sidebar.write(dict(zip(needs, finite_resources)))

def fractal_aid(i, depth, max_depth, visited, resource_state):
    if depth > max_depth or i in visited:
        return np.zeros(len(needs)), resource_state
    visited.add(i)
    helpers = np.where(can_help_matrix[:, i])[0]
    total_aid = np.zeros(len(needs))
    for h in helpers:
        aid_strength = np.random.uniform(0.05, 0.2, len(needs))
        if h in harmful_indices:
            harm_score = np.mean(walk_history[h, -1, :])
            needs_mask = np.array([need in df.loc[h, 'needs'] for need in needs])
            resource_curve = np.log1p(resource_state[needs_mask]) / np.log1p(finite_resources[needs_mask])
            base_aid = -np.log1p(1 + harm_score) * aid_strength[needs_mask] * resource_curve
            resource_state[needs_mask] = np.maximum(resource_state[needs_mask] - 1, 0)
        else:
            helper_score = np.mean(walk_history[h, -1, :])
            needs_mask = np.array([need in df.loc[h, 'needs'] for need in needs])
            resource_curve = np.log1p(resource_state[needs_mask]) / np.log1p(finite_resources[needs_mask])
            base_aid = np.log1p(1 + helper_score) * aid_strength[needs_mask] * resource_curve
            resource_state[needs_mask] = np.maximum(resource_state[needs_mask] - 1, 0)
        # Recursive branching
        branch_aid, resource_state = fractal_aid(h, depth+1, max_depth, visited.copy(), resource_state)
        # Logarithmic payoff increase from leaf to stem
        branch_depth = max_depth - depth + 1
        if branch_depth > 1:
            payoff_multiplier = np.log1p(branch_depth)
        else:
            payoff_multiplier = 1.0
        total_aid[needs_mask] += base_aid * payoff_multiplier
        total_aid += 0.5 * branch_aid
    return total_aid, resource_state

max_branch_depth = st.sidebar.slider("Mutual Aid Branching Depth", min_value=1, max_value=4, value=2, step=1)
aid_weights = np.zeros((N, len(needs)))
global_resource_state = finite_resources.copy()
# Ensure k is defined here for cluster-based aid
if N > 2000:
    st.warning("Fractal branching mutual aid is disabled for N > 2000 for performance. No aid is computed.")
    # No aid is computed for large N
else:
    # Compute fractal aid only for cluster representatives, then assign to all cluster members
    progress = st.progress(0, text="Calculating fractal mutual aid (by cluster)...")
    cluster_representatives = []
    for c in range(k):
        members = df[df['cluster'] == c].index.values
        if len(members) == 0:
            continue
        # Pick the first member as representative
        rep = members[0]
        cluster_representatives.append(rep)
        resource_state = global_resource_state.copy()
        aid, resource_state = fractal_aid(rep, 1, max_branch_depth, set(), resource_state)
        # Assign this aid to all cluster members
        aid_weights[members, :] = aid
        global_resource_state = resource_state
        progress.progress((c+1)/k, text=f"Calculating fractal mutual aid... {c+1}/{k} clusters")
    progress.empty()


# Final needs after mutual aid
final_needs = np.clip(walk_history[:, -1, :] + aid_weights, 0, 1)
avg_final_needs = np.mean(final_needs, axis=0)

# --- PCA + KMeans Clustering: Find 10 maximally different groups ---
st.subheader("Clustering: 10 Maximally Different Groups (PCA + KMeans)")
# Use final_needs as features for clustering
features_for_clustering = final_needs
# Reduce to 2D for visualization
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_for_clustering)
# KMeans clustering
k = 10
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(features_2d)
df['cluster'] = cluster_labels

# Show cluster sizes
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
st.write(f"Cluster sizes: {cluster_counts.to_dict()}")

# Visualize clusters in 2D PCA space
import matplotlib.pyplot as plt
fig3, ax3 = plt.subplots(figsize=(8, 6))
for i in range(k):
    idx = cluster_labels == i
    ax3.scatter(features_2d[idx, 0], features_2d[idx, 1], label=f"Cluster {i}", alpha=0.6)
ax3.set_xlabel("PCA 1")
ax3.set_ylabel("PCA 2")
ax3.set_title("Population Clusters in PCA Space")
ax3.legend()
st.pyplot(fig3)
plt.close(fig3)

# Show a sample of individuals from each cluster
st.write("Sample individuals from each cluster:")
sample_per_cluster = 3
sampled_rows = []
for i in range(k):
    members = df[df['cluster'] == i]
    if not members.empty:
        sampled_rows.append(members.sample(min(sample_per_cluster, len(members)), random_state=42))
if sampled_rows:
    st.dataframe(pd.concat(sampled_rows)[['id', 'age', 'vulnerability', 'location', 'needs', 'cluster']])

st.subheader("Average Final Needs Met per Type After Mutual Aid")
st.bar_chart(pd.Series(avg_final_needs, index=needs))

# Visualize fractal branching structure for a sample of individuals
st.subheader("Mutual Aid Fractal Branching Structure (Sample)")
import networkx as nx
sample_branch_size = st.sidebar.slider("Branching Visualization Sample Size", min_value=1, max_value=10, value=3, step=1)
sample_nodes = np.random.choice(N, sample_branch_size, replace=False)
max_depth_vis = st.sidebar.slider("Branching Visualization Depth", min_value=1, max_value=3, value=2, step=1)


# --- Filtered Branch Graph Visualization: Only 20% Strongest and 20% Weakest Connections ---
def build_branch_graph_filtered(root, depth, max_depth, visited, G, edge_strengths, quantile_low=0.05, quantile_high=0.95):
    if depth > max_depth or root in visited:
        return
    visited.add(root)
    helpers = np.where(can_help_matrix[:, root])[0]
    # Calculate connection strengths (e.g., by aid potential or similarity)
    strengths = []
    for h in helpers:
        # Example: use cosine similarity of needs as connection strength
        v1 = person_needs[root]
        v2 = person_needs[h]
        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        strengths.append((h, sim))
    if strengths:
        sims = np.array([s[1] for s in strengths])
        low = np.quantile(sims, quantile_low)
        high = np.quantile(sims, quantile_high)
        for h, sim in strengths:
            if sim <= low or sim >= high:
                G.add_edge(root, h, weight=sim)
                edge_strengths.append(sim)
                build_branch_graph_filtered(h, depth+1, max_depth, visited.copy(), G, edge_strengths, quantile_low, quantile_high)

for node in sample_nodes:
    G = nx.DiGraph()
    edge_strengths = []
    build_branch_graph_filtered(node, 1, max_depth_vis, set(), G, edge_strengths, quantile_low=0.05, quantile_high=0.95)
    fig, ax = plt.subplots(figsize=(8, 4))
    pos = nx.spring_layout(G, seed=42)
    # Draw edges with color mapped to strength
    edges = G.edges(data=True)
    weights = [d.get('weight', 0.5) for (_, _, d) in edges]
    norm = plt.Normalize(min(weights) if weights else 0, max(weights) if weights else 1)
    edge_colors = plt.cm.coolwarm(norm(weights)) if weights else 'gray'
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=100, arrows=True, edge_color=edge_colors, width=2)
    ax.set_title(f"Branching Structure (Top/Bottom 5%) for Individual {node}")
    st.pyplot(fig)
    plt.close(fig)

# Show final needs for a sample of individuals
st.subheader("Final Needs for Sample Individuals")
if N <= 5000:
    sample_size = st.sidebar.slider("Sample Size to Display", min_value=5, max_value=50, value=10, step=1)
    sample_indices = np.random.choice(N, sample_size, replace=False)
    sample_df = pd.DataFrame(final_needs[sample_indices], columns=needs)
    sample_df['Individual'] = sample_indices
    sample_df = sample_df.set_index('Individual')
    st.dataframe(sample_df.style.background_gradient(cmap="YlGnBu"))
else:
    st.info("Sample table disabled for large populations (N > 5000) for performance.")

# Heatmap: overall population state after mutual aid
st.subheader("Population Needs State After Mutual Aid")
if N <= 5000:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(final_needs, cmap="YlGnBu", cbar_kws={'label': 'Need Met (0=unmet, 1=met)'}, xticklabels=needs, yticklabels=False, ax=ax)
    ax.set_xlabel("Needs")
    ax.set_ylabel("Individuals")
    ax.set_title("Population Needs State After Mutual Aid")
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Heatmap disabled for large populations (N > 5000) for performance.")

# Evolution of needs over time
st.subheader("Evolution of Population Needs Over Time")
avg_needs_over_time = np.mean(walk_history, axis=0)
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.heatmap(avg_needs_over_time.T, cmap="YlGnBu", cbar_kws={'label': 'Average Need Met'}, xticklabels=[f"Step {i}" for i in range(n_steps)], yticklabels=needs, ax=ax2)
ax2.set_xlabel("Simulation Step")
ax2.set_ylabel("Needs")
ax2.set_title("Average Population Needs Over Time")
st.pyplot(fig2)
plt.close(fig2)

## --- PCA + KMeans Clustering: Find 10 maximally different groups ---
st.subheader("Clustering: 10 Maximally Different Groups (PCA + KMeans)")
# Use final_needs as features for clustering
features_for_clustering = final_needs
# Reduce to 2D for visualization
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_for_clustering)
# KMeans clustering
k = 10
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(features_2d)
df['cluster'] = cluster_labels

# Show cluster sizes
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
st.write(f"Cluster sizes: {cluster_counts.to_dict()}")

# Visualize clusters in 2D PCA space
fig3, ax3 = plt.subplots(figsize=(8, 6))
for i in range(k):
    idx = cluster_labels == i
    ax3.scatter(features_2d[idx, 0], features_2d[idx, 1], label=f"Cluster {i}", alpha=0.6)
ax3.set_xlabel("PCA 1")
ax3.set_ylabel("PCA 2")
ax3.set_title("Population Clusters in PCA Space")
ax3.legend()
st.pyplot(fig3)
plt.close(fig3)

# Show a sample of individuals from each cluster
st.write("Sample individuals from each cluster:")
sample_per_cluster = 3
sampled_rows = []
for i in range(k):
    members = df[df['cluster'] == i]
    if not members.empty:
        sampled_rows.append(members.sample(min(sample_per_cluster, len(members)), random_state=42))
if sampled_rows:
    st.dataframe(pd.concat(sampled_rows)[['id', 'age', 'vulnerability', 'location', 'needs', 'cluster']])

# Effects of mutual aid over time
st.subheader("Effects of Mutual Aid Over Time")
mutual_aid_effect = []
for step in range(n_steps):
    cumulative_aid = aid_weights * (step / n_steps)
    needs_with_aid = np.clip(walk_history[:, step, :] + cumulative_aid, 0, 1)
    avg_with_aid = np.mean(needs_with_aid, axis=0)
    mutual_aid_effect.append(avg_with_aid)
mutual_aid_effect = np.array(mutual_aid_effect)
st.line_chart(pd.DataFrame(mutual_aid_effect, columns=needs), use_container_width=True)

st.success("Simulation complete. Adjust parameters in the sidebar and rerun for new results.")
