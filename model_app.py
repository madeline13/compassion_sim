import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import random
import math
import matplotlib.cm as cm

st.set_page_config(page_title="Trained Model Interface", layout="wide")
st.title("Trained Model: Agent Clustering and Prediction")

# Sidebar controls for test data
n_test = st.sidebar.slider("Test Sample Size", min_value=50, max_value=500, value=200, step=10)

# Simulate trained model (from previous simulation)
def get_trained_model_data():
    # For demonstration, use random data similar to the training process
    np.random.seed(42)
    X_train = np.random.normal(loc=0, scale=1, size=(100, 2))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X_pca)
    labels = kmeans.predict(X_pca)
    return pca, kmeans, X_pca, labels

pca, kmeans, X_pca, labels = get_trained_model_data()

st.subheader("Training Data Clusters (PCA)")
fig, ax = plt.subplots()
for i in np.unique(labels):
    ax.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f"Cluster {i}")
ax.legend()
ax.set_title("Training Data Clusters")
st.pyplot(fig)

# Test new agent positions
st.subheader("Test New Agent Positions")
X_test = np.random.normal(loc=0, scale=1, size=(n_test, 2))
X_test_pca = pca.transform(X_test)
test_labels = kmeans.predict(X_test_pca)

fig2, ax2 = plt.subplots()
for i in np.unique(test_labels):
    ax2.scatter(X_test_pca[test_labels == i, 0], X_test_pca[test_labels == i, 1], label=f"Cluster {i}")
ax2.legend()
ax2.set_title("Test Data Clusters")
st.pyplot(fig2)

st.write("Test data points are assigned to clusters using the trained PCA and KMeans model.")

# Interactive prediction for a single agent
st.subheader("Predict Cluster for a Single Agent")
x_input = st.number_input("x position", value=0.0)
y_input = st.number_input("y position", value=0.0)
input_point = np.array([[x_input, y_input]])
input_pca = pca.transform(input_point)
pred_label = kmeans.predict(input_pca)[0]
st.write(f"Predicted cluster: {pred_label}")
st.write("PCA transformed coordinates:", input_pca)
st.subheader("Introduce Noise from Helper, Hinderer")
st.markdown("Adjust noise parameters and view the resulting signal.")

# Interface controls for noise
sample_size = st.number_input("Sample size for noise", min_value=10, max_value=200, value=99, step=1)
mu = st.number_input("Noise mean (mu)", value=0.0)

clean_signal = pd.DataFrame(X_pca, columns=["x_step", "y_step"], dtype=float)
a_trainrange = X_pca[:sample_size, 0]
b_trainrange = X_pca[:sample_size, 1]
result = map(lambda x, y: abs(x + y), a_trainrange, b_trainrange)
d = list(result)
sigma_source = st.selectbox("Noise sigma source", options=["from data", "custom"], index=0)
if sigma_source == "custom":
    sigma_val = st.number_input("Custom sigma value", min_value=0.0, value=1.0)
    sigma = sigma_val
else:
    # d is shape (sample_size,); expand to (sample_size, 2) for broadcasting
    sigma = np.tile(np.array(d).reshape(-1, 1), (1, 2))

try:
    noise = np.random.normal(mu, sigma, [sample_size, 2])
    signal = clean_signal.iloc[:sample_size] + noise
    signal_array = np.array(signal)
    signal_df = pd.DataFrame(signal_array, columns=["x_step", "y_step"], dtype=float)
except Exception as e:
    st.error(f"Noise generation error: {e}")
    signal_df = pd.DataFrame(np.zeros((sample_size,2)), columns=["x_step", "y_step"])

st.write("Noisy signal DataFrame:")
st.dataframe(signal_df)

fig3, ax3 = plt.subplots()
import seaborn as sns
sns.stripplot(data=signal_df, ax=ax3)
ax3.set_title("Signal Distribution with Noise")
st.pyplot(fig3)

# --- Interface for difference between clean and helped/harmed ---
st.subheader("Compare Clean Signal with Helped/Harmed Effects")
effects_df = pd.DataFrame(signal_array, columns=["x_step_w_effect", "y_step_w_effect"], dtype=float)
st.write("Effects DataFrame (helped/harmed):")
st.dataframe(effects_df)
df1 = pd.concat([clean_signal.iloc[:sample_size], effects_df], axis=1)
st.write("Comparison DataFrame (clean vs. effects):")
st.dataframe(df1)
fig6, ax6 = plt.subplots()
import seaborn as sns
sns.stripplot(data=df1, ax=ax6)
ax6.set_title("Comparison: Clean vs. Helped/Harmed Effects")
st.pyplot(fig6)

# --- Altruist resource negotiation interface ---
st.subheader("Altruist Resource Negotiation Simulation")
st.markdown("Adjust hyperparameters and simulate resource negotiation among agents.")

# Hyperparameter controls
speed_help_init = st.number_input("Initial speed_help", min_value=0, max_value=10, value=0)
y_weight = st.slider("Y weight", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
x_weight = st.slider("X weight", min_value=0.0, max_value=1.0, value=0.85, step=0.01)

# Prepare iteredx and iteredy from signal_df for negotiation
iteredx = signal_df["x_step"].values
iteredy = signal_df["y_step"].values

# Diagnostics for negotiation arrays
st.write("Negotiation array diagnostics:")
st.write(f"iteredx shape: {iteredx.shape}, sample: {iteredx[:5]}")
st.write(f"iteredy shape: {iteredy.shape}, sample: {iteredy[:5]}")
st.write(f"logcurvex will be computed from iteredx, negot_x")
st.write(f"logcurvey will be computed from iteredy, negot_x")

# Compute negotiation variables and show diagnostics
negot_x_raw = np.max(iteredx) - np.max(iteredy)
negot_x = max(negot_x_raw, 1e-3)
logcurvex = iteredx + np.log2(negot_x)
logcurvey = iteredy + np.log2(negot_x)
iteredx_negotiate = np.where(iteredx < np.max(iteredx), np.max(iteredx), iteredx + np.log2(negot_x))
iteredy_negotiate = np.where(iteredy < np.max(iteredy), np.max(iteredy), iteredy + np.log2(negot_x))
st.write(f"logcurvex shape: {logcurvex.shape}, sample: {logcurvex[:5]}")
st.write(f"logcurvey shape: {logcurvey.shape}, sample: {logcurvey[:5]}")
st.write(f"iteredx_negotiate shape: {iteredx_negotiate.shape}, sample: {iteredx_negotiate[:5]}")
st.write(f"iteredy_negotiate shape: {iteredy_negotiate.shape}, sample: {iteredy_negotiate[:5]}")

# Simulate negotiation
optx_list = []
opty_list = []
speed_help = speed_help_init
while speed_help < 11:
    speed_help += 1
    optx = np.where(logcurvex > (logcurvex/speed_help), iteredx_negotiate - (logcurvex/speed_help), iteredx + logcurvex/speed_help)
    optx_list.append(optx)
speed_help = speed_help_init
while speed_help < 11:
    speed_help += 1
    opty = np.where(logcurvey > (logcurvey/speed_help), iteredy_negotiate - (logcurvey/speed_help), iteredy + logcurvey/speed_help)
    opty_list.append(opty)
st.write(f"optx_list length: {len(optx_list)}; sample: {optx_list[0][:5] if optx_list else 'empty'}")
st.write(f"opty_list length: {len(opty_list)}; sample: {opty_list[0][:5] if opty_list else 'empty'}")

# Compute negotiation variables

# Ensure negot_x is always positive and above a small threshold
negot_x_raw = np.max(iteredx) - np.max(iteredy)
negot_x = max(negot_x_raw, 1e-3)
logcurvex = iteredx + np.log2(negot_x)
logcurvey = iteredy + np.log2(negot_x)
iteredx_negotiate = np.where(iteredx < np.max(iteredx), np.max(iteredx), iteredx + np.log2(negot_x))
iteredy_negotiate = np.where(iteredy < np.max(iteredy), np.max(iteredy), iteredy + np.log2(negot_x))

# Simulate negotiation
optx_list = []
opty_list = []
speed_help = speed_help_init
while speed_help < 11:
    speed_help += 1
speed_help = speed_help_init
while speed_help < 11:
    speed_help += 1
    opty = np.where(logcurvey > (logcurvey/speed_help), iteredy_negotiate - (logcurvey/speed_help), iteredy + logcurvey/speed_help)
    opty_list.append(opty)

st.write("Negotiated X allocations (optx):")
st.dataframe(np.array(optx_list))
st.write("Negotiated Y allocations (opty):")
# Stripplot for each optx and opty step
st.dataframe(np.array(opty_list))

# Stripplot for iteredx and optx
fig9, ax9 = plt.subplots()
sns.stripplot(data=iteredx, ax=ax9)
ax9.set_title("iteredx stripplot")
st.pyplot(fig9)

fig10, ax10 = plt.subplots()

# Stripplot for iteredy and opty
fig11, ax11 = plt.subplots()
sns.stripplot(data=iteredy, ax=ax11)
ax11.set_title("iteredy stripplot")

# Slider to select speed_help value for stripplot

if optx_list and opty_list:
    min_slider = speed_help_init + 1
    max_slider = speed_help_init + len(optx_list)
    selected_speed_help = st.slider("Select speed_help value for stripplot", min_value=min_slider, max_value=max_slider, value=min_slider)
    idx = selected_speed_help - min_slider
    if 0 <= idx < len(optx_list):
        optx_selected = optx_list[idx]
        opty_selected = opty_list[idx]
        fig_optx, ax_optx = plt.subplots()
        sns.stripplot(data=optx_selected, ax=ax_optx)
        ax_optx.set_title(f"optx stripplot (speed_help={selected_speed_help})")
        st.pyplot(fig_optx)
        fig_opty, ax_opty = plt.subplots()
        sns.stripplot(data=opty_selected, ax=ax_opty)
        ax_opty.set_title(f"opty stripplot (speed_help={selected_speed_help})")
        st.pyplot(fig_opty)
    else:
        st.warning("Selected speed_help value is out of range.")
else:
    st.warning("No negotiation data available for stripplot.")
