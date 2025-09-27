import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
import math

st.set_page_config(page_title="Compassion Simulation", layout="wide")
st.title("Compassion Simulation: Game of Life with Stochastic Elements")

# Sidebar controls
N = st.sidebar.slider("Population (N)", min_value=100, max_value=1000, value=300, step=5)
n_iter = st.sidebar.slider("Iterations", min_value=500, max_value=5000, value=2000, step=100)

# Initial steps
x_step = np.array([random.choice([0,1,2]), random.choice([0,1,2])])
y_step = np.array([random.choice([0,1,2]), random.choice([0,1,2])])

helped = []
harmed = []
blocked = []

class Target():
    def __init__(self, pace = .1, capability = .1, propensity = 0.01, index = None):
        self.pace = random.triangular(.1,.5)
        self.capability = random.triangular(0.1, 1.0, 0.1)
        self.propensity = random.triangular(-0.1, 1.0, 0.1)
    def walking(self, x_step, y_step):
        if self.pace < 3:
            x_step = random.triangular(-1, 0.1, -0.6)
            y_step = random.triangular(-1, 0.1, -0.6)
        elif self.pace == 3:
            x_step = random.triangular(-0.1, 0.04, 0.2)
            y_step = random.triangular(-0.1, 0.04, 0.2)
        else:
            x_step = random.triangular(0.4, 0.1, 0.6)
            y_step = random.triangular(0.4, 0.1, 0.6)
        return (x_step, y_step)

class Helper(Target):
    def __init__(self):
        Target.__init__(self, pace = random.triangular(.1,.5), capability = 0, propensity = 0)
        self.pace = random.triangular(.6,1)
        self.capability = random.triangular(0,.5)
        self.propensity = random.triangular(-0.1, 1.0, 0.45)
    def helping(self, x_step, y_step):
        helped.append(Target().walking(x_step + ((self.propensity) + (self.pace)), y_step + ((self.propensity) + (self.pace))))

class Hinderer(Target):
    def __init__(self):
        Target.__init__(self, pace = random.triangular(.1,.5), capability = 0, propensity = 0)
        self.pace = random.triangular(.1,.5)
        self.capability = random.triangular(0,.5)
        self.propensity = random.triangular(-1.0, 0.1, -0.45)
    def hindering(self, x_step, y_step):
        harmed.append(Target().walking(x_step - ((self.propensity) + (self.pace)), y_step - ((self.propensity) + (self.pace))))

class Obstacle():
    def __init__(self, strength = 0.1):
        self.strength = random.triangular(-0.9, -0.1, -0.3)
    def blocking(self, x_step, y_step):
        blocked.append(Target().walking(x_step - self.strength, y_step - self.strength))

# Run simulation
for n in range(n_iter):
    Helper().helping(x_step, y_step)
    Hinderer().hindering(x_step, y_step)
    Obstacle().blocking(x_step, y_step)

# Cosine similarity calculations
help_cossim = []
for i, j in enumerate(helped):
    help_cossim.append(np.dot(i, j) / ((np.linalg.norm(i)) * (np.linalg.norm(j))))
harm_cossim = []
for i, j in enumerate(harmed):
    harm_cossim.append(np.dot(i, j) / ((np.linalg.norm(i)) * (np.linalg.norm(j))))

# PCA and KMeans clustering
try:
    a_trainrange = help_cossim[1:100]
    b_trainrange = harm_cossim[1:100]
    pca = PCA(2)
    df = pca.fit_transform(a_trainrange)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(df)
    label = kmeans.fit_predict(df)
except Exception as e:
    st.error(f"PCA/KMeans error: {e}")
    df = np.zeros((99,2))
    label = np.zeros(99)

# Visualizations
st.subheader("Agent Position Scatterplots")
help_df = pd.DataFrame(helped, columns=["x_step", "y_step"])
harm_df = pd.DataFrame(harmed, columns=["x_step", "y_step"])
fig1, ax1 = plt.subplots()
sns.scatterplot(data=help_df, x="x_step", y="y_step", ax=ax1)
ax1.set_title("Helped Agents")
st.pyplot(fig1)
fig2, ax2 = plt.subplots()
sns.scatterplot(data=harm_df, x="x_step", y="y_step", ax=ax2)
ax2.set_title("Harmed Agents")
st.pyplot(fig2)

st.subheader("Clustering Results (PCA)")
fig3, ax3 = plt.subplots()
for i in np.unique(label):
    ax3.scatter(df[label == i, 0], df[label == i, 1], label=f"Cluster {i}")
ax3.legend()
ax3.set_title("PCA Clusters of Helped Agents")
st.pyplot(fig3)

# Noise introduction and effects
try:
    clean_signal = pd.DataFrame(df, columns=["x_step", "y_step"], dtype=float)
    result = map(lambda x, y: abs(x + y), a_trainrange, b_trainrange)
    d = list(result)
    mu, sigma = 0, d
    noise = np.random.normal(mu, sigma, [99, 2])
    signal = clean_signal + noise
    signal_array = np.array(signal)
    signal_df = pd.DataFrame(signal_array, columns=["x_step", "y_step"], dtype=float)
except Exception as e:
    st.error(f"Noise generation error: {e}")
    signal_df = pd.DataFrame(np.zeros((99,2)), columns=["x_step", "y_step"])

st.subheader("Signal with Noise")
fig4, ax4 = plt.subplots()
sns.stripplot(data=signal_df, ax=ax4)
ax4.set_title("Signal Distribution with Noise")
st.pyplot(fig4)

# Resource allocation
try:
    effects_df = pd.DataFrame(signal_array, columns=["x_step_w_effect", "y_step_w_effect"], dtype=float)
    df1 = pd.concat([clean_signal, effects_df], axis=1)
    iteredx = np.where(df1['x_step_w_effect'] < 0, abs(df1['x_step'])*10, (df1['x_step_w_effect']) + abs(df1['x_step']))
    iteredy = np.where(df1['y_step_w_effect'] < 0, abs(df1['y_step'])*10, (df1['y_step_w_effect']) + abs(df1['y_step']))
except Exception as e:
    st.error(f"Resource allocation error: {e}")
    iteredx = np.zeros(99)
    iteredy = np.zeros(99)

st.subheader("Resource Allocation Stripplots")
fig5, ax5 = plt.subplots()
sns.stripplot(iteredx, ax=ax5)
ax5.set_title("Resource Allocation X")
st.pyplot(fig5)
fig6, ax6 = plt.subplots()
sns.stripplot(iteredy, ax=ax6)
ax6.set_title("Resource Allocation Y")
st.pyplot(fig6)

st.success("Simulation complete. Adjust parameters in the sidebar and rerun for new results.")
