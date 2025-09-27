# libraries
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import math

# Streamlit for interactive app
import streamlit as st

x_step = np.array([random.choice([0,1,2]), random.choice([0,1,2])])
def run_simulation(N=300, n_iter=2000, help_amount=1, hinder_amount=1,
                  helper_pace=0.8, helper_capability=0.25, helper_propensity=0.45,
                  hinderer_pace=0.3, hinderer_capability=0.25, hinderer_propensity=-0.45):
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
                x_val = float(random.triangular(-1, 0.1, -0.6))
                y_val = float(random.triangular(-1, 0.1, -0.6))
            elif self.pace == 3:
                x_val = float(random.triangular(-0.1, 0.04, 0.2))
                y_val = float(random.triangular(-0.1, 0.04, 0.2))
            else:
                x_val = float(random.triangular(0.4, 0.1, 0.6))
                y_val = float(random.triangular(0.4, 0.1, 0.6))
            return (x_val, y_val)

    class Helper(Target):
        def __init__(self):
            Target.__init__(Target, pace = helper_pace, capability = helper_capability, propensity = helper_propensity)
            self.pace = helper_pace
            self.capability = helper_capability
            self.propensity = helper_propensity
        def helping(self, x_step, y_step):
            for _ in range(help_amount):
                helped.append(Target().walking(x_step + ((self.propensity) + (self.pace)), y_step + ((self.propensity) + (self.pace))))

    class Hinderer(Target):
        def __init__(self):
            Target.__init__(Target, pace = hinderer_pace, capability = hinderer_capability, propensity = hinderer_propensity)
            self.pace = hinderer_pace
            self.capability = hinderer_capability
            self.propensity = hinderer_propensity
        def hindering(self, x_step, y_step):
            for _ in range(hinder_amount):
                harmed.append(Target().walking(x_step - ((self.propensity) + (self.pace)), y_step - ((self.propensity) + (self.pace))))

    class Obstacle():
        def __init__(self, strength = 0.1):
            self.strength = random.triangular(-0.9, -0.1, -0.3)
        def blocking(self, x_step, y_step):
            blocked.append(Target().walking(x_step - self.strength, y_step - self.strength))

    for n in range(n_iter):
        for _ in range(N):
            x_step = np.array([random.choice([0,1,2]), random.choice([0,1,2])])
            y_step = np.array([random.choice([0,1,2]), random.choice([0,1,2])])
            Helper().helping(x_step, y_step)
            Hinderer().hindering(x_step, y_step)
            Obstacle().blocking(x_step, y_step)

    # Cosine similarity calculations
    help_cossim = []
    for idx in range(len(helped)-1):
        i = helped[idx]
        j = helped[idx+1]
        i = np.array(i).flatten()
        j = np.array(j).flatten()
        if i.shape != (2,) or j.shape != (2,):
            continue
        norm_i = linalg.norm(i)
        norm_j = linalg.norm(j)
        if norm_i == 0 or norm_j == 0:
            help_cossim.append([0, 0])
            blocked.append([0, 0])
        else:
            val = np.dot(i, j) / (norm_i * norm_j)
            help_cossim.append([val, val])
            blocked.append([val, val])

    harm_cossim = []
    for idx in range(len(harmed)-1):
        i = harmed[idx]
        j = harmed[idx+1]
        i = np.array(i).flatten()
        j = np.array(j).flatten()
        if i.shape != (2,) or j.shape != (2,):
            continue
        norm_i = linalg.norm(i)
        norm_j = linalg.norm(j)
        if norm_i == 0 or norm_j == 0:
            harm_cossim.append([0, 0])
            blocked.append([0, 0])
        else:
            val = np.dot(i, j) / (norm_i * norm_j)
            harm_cossim.append([val, val])
            blocked.append([val, val])

    # DataFrames for visualization (use raw agent positions)
    help_flat = [np.array(pt).flatten() for pt in helped if np.array(pt).flatten().shape == (2,)]
    harm_flat = [np.array(pt).flatten() for pt in harmed if np.array(pt).flatten().shape == (2,)]
    help_df = pd.DataFrame(help_flat, columns = ['x_step', 'y_step'])
    harm_df = pd.DataFrame(harm_flat, columns = ['x_step', 'y_step'])

    # Clustering
    a_trainrange = np.array(help_cossim)
    if a_trainrange.size == 0:
        a_trainrange = np.zeros((1,2))
    elif a_trainrange.ndim == 1:
        a_trainrange = a_trainrange.reshape(-1, 2)
    elif a_trainrange.ndim > 2:
        a_trainrange = a_trainrange.reshape(a_trainrange.shape[0], -1)
    X = a_trainrange
    if X.shape[0] >= 2 and X.shape[1] >= 2:
        pca = PCA(2)
        df = pca.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(df)
        label = kmeans.fit_predict(df)
    else:
        # Not enough samples/features for PCA, fill with zeros
        df = np.zeros((X.shape[0], 2))
        label = np.zeros(X.shape[0])

    b_trainrange = np.array(harm_cossim)
    if b_trainrange.size == 0:
        b_trainrange = np.zeros((1,2))
    elif b_trainrange.ndim == 1:
        b_trainrange = b_trainrange.reshape(-1, 2)
    elif b_trainrange.ndim > 2:
        b_trainrange = b_trainrange.reshape(b_trainrange.shape[0], -1)
    X2 = b_trainrange
    if X2.shape[0] >= 2 and X2.shape[1] >= 2:
        pca2 = PCA(2)
        df2 = pca2.fit_transform(X2)
        kmeans2 = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(df2)
        label2 = kmeans2.fit_predict(df2)
    else:
        # Not enough samples/features for PCA, fill with zeros
        df2 = np.zeros((X2.shape[0], 2))
        label2 = np.zeros(X2.shape[0])

    return {
        'help_df': help_df,
        'harm_df': harm_df,
        'df': df,
        'label': label,
        'df2': df2,
        'label2': label2
    }





def main():
    # Move histogram plotting after iteredx and iteredy are defined
    st.title("Compassion Simulation: Game of Life with RL and Clustering")
    st.write("This app demonstrates the simulation and clustering of agent behaviors in a stochastic Game of Life.")

    N = st.slider("Population Size (N)", min_value=100, max_value=10000, value=1000, step=100)
    n_iter = st.slider("Iterations", min_value=100, max_value=20000, value=2000, step=100)

    help_amount = st.slider("Helper: Amount of Helping", min_value=1, max_value=10, value=1)
    hinder_amount = st.slider("Hinderer: Amount of Hindering", min_value=1, max_value=10, value=1)

    st.markdown("### Helper Parameters")
    helper_pace = st.slider("Helper Pace", min_value=1, max_value=10, value=5, step=1)
    helper_capability = st.slider("Helper Capability", min_value=1, max_value=10, value=5, step=1)
    helper_propensity = st.slider("Helper Propensity", min_value=1, max_value=10, value=5, step=1)

    st.markdown("### Hinderer Parameters")
    hinderer_pace = st.slider("Hinderer Pace", min_value=1, max_value=10, value=5, step=1)
    hinderer_capability = st.slider("Hinderer Capability", min_value=1, max_value=10, value=5, step=1)
    hinderer_propensity = st.slider("Hinderer Propensity", min_value=1, max_value=10, value=5, step=1)

    if st.button("Run Simulation", key="run_simulation_button"):
        results = run_simulation(
            int(help_amount), int(hinder_amount),
            int(helper_pace), int(hinderer_pace),
            int(helper_capability), int(hinderer_capability),
            int(helper_propensity), int(hinderer_propensity)
        )
        st.success("Simulation complete!")

        # DataFrame and visualizations (now inside the button block)
        df1 = pd.DataFrame(results['df'], columns=['x_step_w_effect', 'y_step_w_effect'])
        st.dataframe(df1)
        st.subheader("Stripplot: Effects Distribution")
        fig6, ax6 = plt.subplots()
        sns.stripplot(data=df1, ax=ax6)
        st.pyplot(fig6)
        plt.close(fig6)

        # Effects and resource allocation
        mask_x = df1['x_step_w_effect'] < 0
        iteredx = np.where(mask_x, abs(df1['x_step_w_effect'])*10, (df1['x_step_w_effect']) + abs(df1['x_step_w_effect']))
        fig7, ax7 = plt.subplots()
        sns.stripplot(data=iteredx, ax=ax7)
        st.pyplot(fig7)
        plt.close(fig7)

        mask_y = df1['y_step_w_effect'] < 0
        iteredy = np.where(mask_y, abs(df1['y_step_w_effect'])*10, (df1['y_step_w_effect']) + abs(df1['y_step_w_effect']))
        fig8, ax8 = plt.subplots()
        sns.stripplot(data=iteredy, ax=ax8)
        st.pyplot(fig8)
        plt.close(fig8)

        negot_x = max(iteredx) - max(iteredy)
        if negot_x > 0:
            log_val = math.log2(negot_x)
        else:
            log_val = 0
        logcurvex = iteredx + log_val
        logcurvey = iteredy + log_val

        fig9, ax9 = plt.subplots()
        sns.stripplot(logcurvex, ax=ax9)
        st.pyplot(fig9)
        plt.close(fig9)

        mask_iteredx = iteredx < max(iteredx)
        iteredx_negotiate = np.where(mask_iteredx, max(iteredx), iteredx + log_val)
        mask_iteredy = iteredy < max(iteredy)
        iteredy_negotiate = np.where(mask_iteredy, max(iteredy), iteredy + log_val)

        speed_help = 0
        y_weight = 0.75
        x_weight = 0.85
        optx_array = []
        opty_array = []
        while (speed_help < int(11)):
            speed_help = speed_help + 1
            divisor_x = logcurvex / speed_help
            mask_x_opt = logcurvex > divisor_x
            optx = np.where(mask_x_opt, iteredx_negotiate - divisor_x, iteredx_negotiate + divisor_x)
            optx_array.extend(optx.tolist())
        optx_array = np.array(optx_array)
        st.write("Optimized x allocations (array):", optx_array)

        speed_help = 0
        while (speed_help < int(11)):
            speed_help = speed_help + 1
            divisor_y = logcurvey / speed_help
            mask_y_opt = logcurvey > divisor_y
            opty = np.where(mask_y_opt, iteredy_negotiate - divisor_y, iteredy_negotiate + divisor_y)
            opty_array.extend(opty.tolist())
        opty_array = np.array(opty_array)
        st.write("Optimized y allocations (array):", opty_array)

        st.subheader("Histogram: Optimized x allocations")
        fig_optx, ax_optx = plt.subplots()
        ax_optx.hist(optx_array, bins=int(30), color='mediumslateblue', edgecolor='black')
        ax_optx.set_xlabel('Optimized x allocation')
        ax_optx.set_ylabel('Frequency')
        st.pyplot(fig_optx)
        plt.close(fig_optx)

        st.subheader("Histogram: Optimized y allocations")
        fig_opty, ax_opty = plt.subplots()
        ax_opty.hist(opty_array, bins=int(30), color='mediumseagreen', edgecolor='black')
        ax_opty.set_xlabel('Optimized y allocation')
        ax_opty.set_ylabel('Frequency')
        st.pyplot(fig_opty)
        plt.close(fig_opty)



if __name__ == "__main__":
    main()

