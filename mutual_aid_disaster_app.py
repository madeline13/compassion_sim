k = 10
import streamlit as st
from io import StringIO
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

# --- Unsupervised Learning: Data Upload and Clustering-based Outcome Prediction ---
st.sidebar.header("Unsupervised ML: Predict Outcomes")
ml_mode = st.sidebar.checkbox("Enable ML Outcome Prediction", value=False)

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



@st.cache_data(show_spinner=False)
def generate_population(N, needs):
    ids = np.arange(N)
    ages = np.random.randint(0, 90, N)
    vulnerability = np.clip(np.random.normal(loc=0.5, scale=0.2, size=N), 0, 1)
    locations = np.random.choice(['camp', 'urban', 'rural'], size=N)
    # Convert needs_list to tuples for hashability
    needs_list = [tuple(np.random.choice(needs, size=np.random.randint(1, 4), replace=False)) for _ in range(N)]
    df = pd.DataFrame({
        'id': ids,
        'age': ages,
        'vulnerability': vulnerability,
        'location': locations,
        'needs': needs_list
    })
    return df

df = generate_population(N, needs)


@st.cache_data(show_spinner=False)
def generate_person_needs(df, needs):
    N = len(df)
    person_needs = np.ones((N, len(needs)))
    for i in range(N):
        vuln = df.loc[i, 'vulnerability']  # Now a float between 0 and 1
        for j, need in enumerate(needs):
            if need in df.loc[i, 'needs']:
                # More vulnerable = lower initial need met
                low = 0.1 + 0.4 * (1 - vuln)
                high = 0.6 + 0.3 * (1 - vuln)
                person_needs[i, j] = np.random.uniform(low, high)
    return person_needs

person_needs = generate_person_needs(df, needs)

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


@st.cache_data(show_spinner=False)
def simulate_walk_history(person_needs, n_steps, needs):
    N = person_needs.shape[0]
    walk_history = np.zeros((N, n_steps, len(needs)))
    walk_history[:, 0, :] = person_needs
    # Define need hierarchy indices
    level1 = [needs.index(n) for n in ['food', 'water', 'shelter']]
    level2 = [needs.index(n) for n in ['medical', 'sanitation', 'security', 'transport', 'communication']]
    level3 = [needs.index(n) for n in ['psychological', 'education']]
    threshold = 0.5
    for step in range(1, n_steps):
        # Vectorized random walk for all individuals
        random_steps = np.random.normal(loc=0.05, scale=0.02, size=(N, len(needs)))
        next_step = np.clip(walk_history[:, step-1, :] + random_steps, 0, 1)
        # Enforce hierarchy: cap higher-level needs if lower-level needs are unmet
        min_level1 = np.min(next_step[:, level1], axis=1)
        min_level2 = np.min(next_step[:, level2], axis=1)
        # Cap level2 and level3 by level1
        for idx in level2:
            next_step[:, idx] = np.where(min_level1 < threshold, np.minimum(next_step[:, idx], min_level1), next_step[:, idx])
        for idx in level3:
            next_step[:, idx] = np.where(min_level1 < threshold, np.minimum(next_step[:, idx], min_level1), next_step[:, idx])
            next_step[:, idx] = np.where(min_level2 < threshold, np.minimum(next_step[:, idx], min_level2), next_step[:, idx])
        walk_history[:, step, :] = next_step
    return walk_history

walk_history = simulate_walk_history(person_needs, n_steps, needs)


# --- PCA + KMeans Clustering: Find 10 maximally different groups ---
# Use initial needs as features for clustering (or use final_needs if you want post-aid clusters)
features_for_clustering = person_needs
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_for_clustering)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(features_2d)
df['cluster'] = cluster_labels


# Resource density and scarcity sliders
st.sidebar.subheader("Resource Environment")
resource_density = st.sidebar.slider("Resource Density (mean)", min_value=0.1, max_value=2.0, value=1.0, step=0.05, help="Controls the average amount of resources available per need.")
resource_scarcity = st.sidebar.slider("Resource Scarcity (spread)", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="Controls the variability (scarcity) of resources across needs.")

# Model finite resources for each need (e.g., food, water) based on sliders
min_res = int(N * 0.1 * resource_density * (1 - resource_scarcity))
max_res = int(N * 1.0 * resource_density * (1 + resource_scarcity))
if min_res < 1: min_res = 1
finite_resources = np.random.randint(min_res, max_res+1, size=len(needs))

# --- Enhanced Finite Tangible Resources Sidebar Section ---
st.sidebar.markdown("---")
st.sidebar.subheader(":package: Finite Tangible Resources")
st.sidebar.caption("These are the available resources for each need in the simulation. Adjust density and scarcity above to see how resources change.")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
resource_dict = dict(zip(needs, finite_resources))

# Color bar chart for resources
fig_res, ax_res = plt.subplots(figsize=(2.5, 2.5))
colors = list(mcolors.TABLEAU_COLORS.values())
ax_res.barh(list(resource_dict.keys()), list(resource_dict.values()), color=colors[:len(resource_dict)])
ax_res.set_xlabel("Units")
ax_res.set_title("Resources per Need", fontsize=10)
plt.tight_layout()
st.sidebar.pyplot(fig_res)
plt.close(fig_res)

# Pretty table with tooltips
import streamlit as st
from collections import OrderedDict
st.sidebar.markdown("**Resource Levels:**")
for i, (need, val) in enumerate(resource_dict.items()):
    st.sidebar.markdown(f"- <span title='Units available for {need}' style='color:{colors[i%len(colors)]};font-weight:bold'>{need.capitalize()}</span>: <b>{val}</b>", unsafe_allow_html=True)
st.sidebar.markdown("---")

def fractal_aid(i, depth, max_depth, visited, resource_state, reciprocity_boost=None):
    if reciprocity_boost is None:
        reciprocity_boost = np.zeros(len(needs))
    if depth > max_depth or i in visited:
        return np.zeros(len(needs)), resource_state, reciprocity_boost
    visited.add(i)
    helpers = np.where(can_help_matrix[:, i])[0]
    total_aid = np.zeros(len(needs))
    received_aid = np.zeros(len(needs))  # Track how much aid this node receives
    for h in helpers:
        # --- Reciprocity: helpers who received more aid in the past are more likely to help ---
        base_aid_strength = np.random.uniform(0.05, 0.2, len(needs))
        # Boost aid strength if this helper received aid in the past (reciprocity)
        boost = reciprocity_boost if reciprocity_boost is not None else np.zeros(len(needs))
        aid_strength = base_aid_strength + 0.2 * boost  # 0.2 is a tunable feedback factor
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

            # --- Resource/Need Trading Logic ---
            helper_needs = walk_history[h, -1, :]
            recipient_needs = walk_history[i, -1, :]
            surplus_indices = np.where(helper_needs < 0.3)[0]
            deficit_indices = np.where(recipient_needs > 0.7)[0]
            for s in surplus_indices:
                for d in deficit_indices:
                    if s != d and resource_state[s] > 0 and resource_state[d] > 0:
                        trade_amt = min(0.1, resource_state[s], resource_state[d])
                        resource_state[s] -= trade_amt
                        resource_state[d] -= trade_amt
                        total_aid[d] += trade_amt * 0.5
                        total_aid[s] += trade_amt * 0.5

        # Recursive branching, pass updated reciprocity_boost
        branch_aid, resource_state, branch_reciprocity = fractal_aid(h, depth+1, max_depth, visited.copy(), resource_state, reciprocity_boost)
        # Logarithmic payoff increase from leaf to stem
        branch_depth = max_depth - depth + 1
        if branch_depth > 1:
            payoff_multiplier = np.log1p(branch_depth)
        else:
            payoff_multiplier = 1.0
        total_aid[needs_mask] += base_aid * payoff_multiplier
        total_aid += 0.5 * branch_aid
        received_aid[needs_mask] += base_aid * payoff_multiplier  # Track aid received for reciprocity
    # Update reciprocity_boost for this node: more received aid = more likely to help others
    new_reciprocity_boost = reciprocity_boost + 0.5 * received_aid  # 0.5 is a tunable feedback factor
    return total_aid, resource_state, new_reciprocity_boost

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
        aid, resource_state, _ = fractal_aid(rep, 1, max_branch_depth, set(), resource_state)
        # Assign this aid to all cluster members
        aid_weights[members, :] = aid
        global_resource_state = resource_state
        progress.progress((c+1)/k, text=f"Calculating fractal mutual aid... {c+1}/{k} clusters")
    progress.empty()


# Final needs after mutual aid
final_needs = np.clip(walk_history[:, -1, :] + aid_weights, 0, 1)
# Enforce hierarchy on final_needs
level1 = [needs.index(n) for n in ['food', 'water', 'shelter']]
level2 = [needs.index(n) for n in ['medical', 'sanitation', 'security', 'transport', 'communication']]
level3 = [needs.index(n) for n in ['psychological', 'education']]
threshold = 0.5
min_level1 = np.min(final_needs[:, level1], axis=1)
min_level2 = np.min(final_needs[:, level2], axis=1)
for idx in level2:
    final_needs[:, idx] = np.where(min_level1 < threshold, np.minimum(final_needs[:, idx], min_level1), final_needs[:, idx])
for idx in level3:
    final_needs[:, idx] = np.where(min_level1 < threshold, np.minimum(final_needs[:, idx], min_level1), final_needs[:, idx])
    final_needs[:, idx] = np.where(min_level2 < threshold, np.minimum(final_needs[:, idx], min_level2), final_needs[:, idx])
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
#########################
# ML OUTCOME PREDICTION #
#########################
if ml_mode:
    st.subheader("Upload Data for Unsupervised Learning (Clustering)")
    uploaded_file = st.file_uploader("Upload CSV (features + optional outcome column)", type=["csv"])
    if 'use_sim' not in st.session_state:
        st.session_state['use_sim'] = False
    st.info("Or use the current simulation output for ML clustering:")
    if st.button("Use Simulation Output for ML Clustering"):
        st.session_state['use_sim'] = True
    data = None
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state['use_sim'] = False
        st.write("Preview of uploaded data:", data.head())
    elif st.session_state['use_sim']:
        # Use simulation output: combine df and final_needs
        sim_data = df.copy().reset_index(drop=True)
        for i, n in enumerate(needs):
            sim_data[n] = final_needs[:, i]
        data = sim_data
        st.write("Preview of simulation output as ML data:", data.head())
    if data is not None:
        # Only allow numeric columns for default feature selection
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("No numeric columns found in data. Please upload a file or run a simulation with numeric features.")
        else:
            feature_cols = st.multiselect(
                "Select features for clustering (numeric only)",
                data.columns.tolist(),
                default=numeric_cols
            )
            # Warn if any selected features are not numeric
            non_numeric_selected = [col for col in feature_cols if col not in numeric_cols]
            if non_numeric_selected:
                st.warning(f"The following selected features are not numeric and will be excluded: {non_numeric_selected}")
                feature_cols = [col for col in feature_cols if col in numeric_cols]
            n_clusters = st.slider("Number of clusters (k)", min_value=2, max_value=20, value=5)
            if st.button("Run Clustering"):
                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA
                from sklearn.cluster import KMeans
                X = data[feature_cols].values
                X_scaled = StandardScaler().fit_transform(X)
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                data['PredictedCluster'] = cluster_labels
                st.write("Cluster assignments:", data[['PredictedCluster']].value_counts().sort_index())
                # Visualize clusters
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 6))
                for i in range(n_clusters):
                    idx = cluster_labels == i
                    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Cluster {i}", alpha=0.6)
                ax.set_xlabel("PCA 1")
                ax.set_ylabel("PCA 2")
                ax.set_title("Clusters in Data (PCA)")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
                # If outcome column exists, show cluster-outcome relationship
                outcome_col = st.selectbox("Optional: Select outcome column to compare with clusters", [None] + data.columns.tolist())
                if outcome_col and outcome_col in data.columns:
                    st.write("Outcome distribution by cluster:")
                    st.dataframe(data.groupby('PredictedCluster')[outcome_col].value_counts().unstack(fill_value=0))
    st.markdown("---")
    st.info("If you see a TypeError about loading a JS module, try a hard refresh in your browser. If that fails, stop Streamlit, clear the .streamlit and __pycache__ folders, and relaunch the app.")


# Visualize fractal branching structure for a sample of individuals (limit for large N)
if N <= 2000:
    st.subheader("Mutual Aid Fractal Branching Structure (Sample)")
    import networkx as nx
    sample_branch_size = st.sidebar.slider("Branching Visualization Sample Size", min_value=1, max_value=10, value=3, step=1)
    sample_nodes = np.random.choice(N, sample_branch_size, replace=False)
    max_depth_vis = st.sidebar.slider("Branching Visualization Depth", min_value=1, max_value=3, value=2, step=1)

    def build_branch_graph_filtered(root, depth, max_depth, visited, G, edge_strengths, quantile_low=0.05, quantile_high=0.95):
        if depth > max_depth or root in visited:
            return
        visited.add(root)
        helpers = np.where(can_help_matrix[:, root])[0]
        strengths = []
        for h in helpers:
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
        edges = G.edges(data=True)
        weights = [d.get('weight', 0.5) for (_, _, d) in edges]
        norm = plt.Normalize(min(weights) if weights else 0, max(weights) if weights else 1)
        edge_colors = plt.cm.coolwarm(norm(weights)) if weights else 'gray'
        nx.draw(G, pos, ax=ax, with_labels=True, node_size=100, arrows=True, edge_color=edge_colors, width=2)
        ax.set_title(f"Branching Structure (Top/Bottom 5%) for Individual {node}")
        st.pyplot(fig)
        plt.close(fig)
else:
    st.info("Branching visualization disabled for large populations (N > 2000) for performance.")

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


# Evolution of needs over time (mean, min, max)
st.subheader("Evolution of Population Needs Over Time (Mean)")
avg_needs_over_time = np.mean(walk_history, axis=0)
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.heatmap(avg_needs_over_time.T, cmap="YlGnBu", cbar_kws={'label': 'Average Need Met'}, xticklabels=[f"Step {i}" for i in range(n_steps)], yticklabels=needs, ax=ax2)
ax2.set_xlabel("Simulation Step")
ax2.set_ylabel("Needs")
ax2.set_title("Average Population Needs Over Time")
st.pyplot(fig2)
plt.close(fig2)

st.subheader("Evolution of Population Needs Over Time (Min)")
min_needs_over_time = np.min(walk_history, axis=0)
fig_min, ax_min = plt.subplots(figsize=(12, 6))
sns.heatmap(min_needs_over_time.T, cmap="YlOrRd", cbar_kws={'label': 'Min Need Met'}, xticklabels=[f"Step {i}" for i in range(n_steps)], yticklabels=needs, ax=ax_min)
ax_min.set_xlabel("Simulation Step")
ax_min.set_ylabel("Needs")
ax_min.set_title("Minimum Population Needs Over Time")
st.pyplot(fig_min)
plt.close(fig_min)

st.subheader("Evolution of Population Needs Over Time (Max)")
max_needs_over_time = np.max(walk_history, axis=0)
fig_max, ax_max = plt.subplots(figsize=(12, 6))
sns.heatmap(max_needs_over_time.T, cmap="YlGn", cbar_kws={'label': 'Max Need Met'}, xticklabels=[f"Step {i}" for i in range(n_steps)], yticklabels=needs, ax=ax_max)
ax_max.set_xlabel("Simulation Step")
ax_max.set_ylabel("Needs")
ax_max.set_title("Maximum Population Needs Over Time")
st.pyplot(fig_max)
plt.close(fig_max)

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
