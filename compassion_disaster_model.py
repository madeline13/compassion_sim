import numpy as np
import pandas as pd
import streamlit as st

# Parameters for population and needs
total_population = 1000
np.random.seed(42)

# Define possible needs
needs = [
    'food', 'water', 'shelter', 'medical', 'sanitation', 'psychological', 'education', 'security', 'transport', 'communication'
]

# Generate population data
population_data = []
for i in range(total_population):
    person = {
        'id': i,
        'age': np.random.randint(0, 90),
        'vulnerability': np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2]),
        'location': np.random.choice(['camp', 'urban', 'rural']),
        'needs': np.random.choice(needs, size=np.random.randint(1, 4), replace=False).tolist()
    }
    population_data.append(person)

df = pd.DataFrame(population_data)

# Aggregate needs
need_counts = {}
for need in needs:
    need_counts[need] = df['needs'].apply(lambda x: need in x).sum()

# Model: Predict most compassionate response
# Simple model: prioritize high vulnerability, most common needs
compassion_scores = {}
for need in needs:
    high_vuln = df[(df['vulnerability'] == 'high') & (df['needs'].apply(lambda x: need in x))].shape[0]
    score = 0.7 * high_vuln + 0.3 * need_counts[need]
    compassion_scores[need] = score

# Sort needs by compassion score
sorted_needs = sorted(compassion_scores.items(), key=lambda x: x[1], reverse=True)

# Streamlit UI
st.title("Humanitarian Disaster Needs & Compassionate Response Model")
st.write("Population size:", total_population)
st.write("Needs distribution:")
st.bar_chart(pd.Series(need_counts))
st.write("Compassionate response prioritization:")
st.table(pd.DataFrame(sorted_needs, columns=['Need', 'Compassion Score']))

# Show sample population data
st.write("Sample population data:")
st.dataframe(df.sample(10))

# Model: How members can help one another
st.header("Mutual Aid Potential Among Population")
# Define help criteria: can help if not highly vulnerable and shares location or need
can_help_matrix = np.zeros((total_population, total_population), dtype=int)
for i in range(total_population):
    for j in range(total_population):
        if i == j:
            continue
        # Helper must not be highly vulnerable
        if df.loc[i, 'vulnerability'] == 'high':
            continue
        # Can help if shares location or at least one need
        shared_location = df.loc[i, 'location'] == df.loc[j, 'location']
        shared_need = bool(set(df.loc[i, 'needs']) & set(df.loc[j, 'needs']))
        if shared_location or shared_need:
            can_help_matrix[i, j] = 1

# Summarize help potential
total_possible_helps = np.sum(can_help_matrix)
avg_helps_per_person = total_possible_helps / total_population
st.write(f"Total possible help connections: {total_possible_helps}")
st.write(f"Average helps per person: {avg_helps_per_person:.2f}")

# Show a sample of help connections
sample_idx = np.random.choice(total_population, 5, replace=False)
help_samples = {}
for idx in sample_idx:
    helpers = np.where(can_help_matrix[:, idx] == 1)[0].tolist()
    help_samples[f"Person {idx}"] = helpers
st.write("Sample help connections (who can help whom):")
st.json(help_samples)

# --- Semi-Random Walks in Needs Space & Mutual Aid ---
st.header("Population Progress: Semi-Random Walks in Needs Space")
# Each person starts with a vector of needs (0=unmet, 0.25=low, 0.5=medium, 0.75=high, 1=met)
person_needs = np.zeros((total_population, len(needs)))
for i in range(total_population):
    for j, need in enumerate(needs):
        if need in df.loc[i, 'needs']:
            # More granular: vulnerability affects initial need level
            vuln = df.loc[i, 'vulnerability']
            if vuln == 'high':
                person_needs[i, j] = np.random.choice([0, 0.25])  # high vulnerability, need mostly unmet
            elif vuln == 'medium':
                person_needs[i, j] = np.random.choice([0.25, 0.5])  # medium vulnerability, need partially met
            else:
                person_needs[i, j] = np.random.choice([0.5, 0.75])  # low vulnerability, need mostly met
        else:
            person_needs[i, j] = 1.0  # need not present, considered met

# Simulate semi-random walks toward optimal (all needs met)
n_steps = 20
walk_history = np.zeros((total_population, n_steps, len(needs)))
walk_history[:, 0, :] = person_needs
for step in range(1, n_steps):
    for i in range(total_population):
        # Random improvement for each need (small step toward 1.0)
        random_step = np.random.normal(loc=0.05, scale=0.02, size=len(needs))
        walk_history[i, step, :] = np.clip(walk_history[i, step-1, :] + random_step, 0, 1)

# Model mutual aid: weighted contributions from helpers
st.header("Mutual Aid: Weighted Contributions to Needs")
# For each person, sum weighted help from others who can help them
aid_weights = np.zeros((total_population, len(needs)))

# Randomly select a fraction of population as 'harmful' (e.g., 15%)
harmful_fraction = 0.15
harmful_indices = np.random.choice(total_population, int(total_population * harmful_fraction), replace=False)

for i in range(total_population):
    helpers = np.where(can_help_matrix[:, i] == 1)[0]
    for h in helpers:
        # Assign a random aid strength for each helper and need (between 0.05 and 0.2)
        aid_strength = np.random.uniform(0.05, 0.2, len(needs))
        # Harmful members reduce needs (negative aid)
        if h in harmful_indices:
            harm_score = np.mean(walk_history[h, -1, :])
            for j, need in enumerate(needs):
                if need in df.loc[h, 'needs']:
                    # Logarithmic diminishing harm, with random strength
                    aid_weights[i, j] -= np.log1p(1 + harm_score) * aid_strength[j]
        else:
            # Helpful members contribute positively
            helper_score = np.mean(walk_history[h, -1, :])
            for j, need in enumerate(needs):
                if need in df.loc[h, 'needs']:
                    # Logarithmic diminishing returns for aid, with random strength
                    aid_weights[i, j] += np.log1p(1 + helper_score) * aid_strength[j]

# Apply mutual aid to final needs state
final_needs = np.clip(walk_history[:, -1, :] + aid_weights, 0, 1)
avg_final_needs = np.mean(final_needs, axis=0)
st.write("Average final needs met per type after mutual aid:", dict(zip(needs, avg_final_needs)))

# Visualize impact of harmful vs helpful members
avg_harmful = np.mean(final_needs[harmful_indices], axis=0) if len(harmful_indices) > 0 else np.zeros(len(needs))
avg_helpful = np.mean(final_needs[[i for i in range(total_population) if i not in harmful_indices]], axis=0)
st.write("Average needs met for harmful members:", dict(zip(needs, avg_harmful)))
st.write("Average needs met for helpful members:", dict(zip(needs, avg_helpful)))

# --- Visualization: Overall Needs Met After Mutual Aid ---
st.header("Overall Needs Met After Mutual Aid")
st.write(f"Average number of helps per person: {avg_helps_per_person:.2f}")
overall_needs_series = pd.Series(avg_final_needs, index=needs)
st.bar_chart(overall_needs_series)

# Visualize progress for a sample of individuals

# Visualize overall population state as a heatmap
st.header("Overall Population State After Mutual Aid")
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(final_needs, cmap="YlGnBu", cbar_kws={'label': 'Need Met (0=unmet, 1=met)'}, xticklabels=needs, yticklabels=False, ax=ax)
ax.set_xlabel("Needs")
ax.set_ylabel("Individuals")
ax.set_title("Population Needs State After Mutual Aid")
st.pyplot(fig)
plt.close(fig)

# Visualize evolution of needs over time (average per need at each step)
st.header("Evolution of Population Needs Over Time")
avg_needs_over_time = np.mean(walk_history, axis=0)  # shape: (n_steps, len(needs))
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.heatmap(avg_needs_over_time.T, cmap="YlGnBu", cbar_kws={'label': 'Average Need Met'}, xticklabels=[f"Step {i}" for i in range(n_steps)], yticklabels=needs, ax=ax2)
ax2.set_xlabel("Simulation Step")
ax2.set_ylabel("Needs")
ax2.set_title("Average Population Needs Over Time")
st.pyplot(fig2)
plt.close(fig2)

# Visualize effects of mutual aid over time
st.header("Effects of Mutual Aid Over Time")
# For each step, estimate the average effect of mutual aid by comparing walk_history (no aid) vs. walk_history + cumulative aid
mutual_aid_effect = []
for step in range(n_steps):
    # Estimate cumulative aid up to this step (scale aid_weights by step/n_steps)
    cumulative_aid = aid_weights * (step / n_steps)
    needs_with_aid = np.clip(walk_history[:, step, :] + cumulative_aid, 0, 1)
    avg_with_aid = np.mean(needs_with_aid, axis=0)
    mutual_aid_effect.append(avg_with_aid)
mutual_aid_effect = np.array(mutual_aid_effect)  # shape: (n_steps, len(needs))
st.line_chart(pd.DataFrame(mutual_aid_effect, columns=needs), use_container_width=True)

# --- Optimal Needs Vector & Finite Resource Allocation ---

# --- No Outside Aid: Only Mutual Aid ---
st.header("No Outside Aid: Only Mutual Aid Available")
st.write("In this scenario, all progress toward meeting needs comes from within the population. There are no external resources or interventions.")
