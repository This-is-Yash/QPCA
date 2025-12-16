import numpy as np
import pandas as pd
import streamlit as st
from qutip import Qobj
from qutip.visualization import matrix_histogram
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ======== PAGE TITLE ========
st.title("⚛️ Quantum Principal Component Analysis (QPCA)")
st.markdown(
    "This web app demonstrates **Quantum PCA (QPCA)** using classical simulation with density matrices."
)

# ======== SIDEBAR CONTROLS ========
st.sidebar.header("⚙️ Controls")
use_demo = st.sidebar.checkbox("Use demo dataset", True)
show_density_plot = st.sidebar.checkbox("Show density matrix visualization", True)
show_steps = st.sidebar.checkbox("Show step-by-step computation", True)
show_scree = st.sidebar.checkbox("Show scree plot", True)
show_bloch = st.sidebar.checkbox("Show Bloch sphere visualization per PC", True)

# ======== DATA INPUT =========
data = None
if use_demo:
    st.subheader("📊 Demo Dataset")
    data = np.array([[3.0, 1.0, 3.0],
                     [4.0, 2.0, 4.0],
                     [4.0, 6.0, 7.0],
                     [1.0, 5.0, 7.0]])
    df = pd.DataFrame(data)
    edited_df = st.data_editor(df, num_rows="dynamic")
    data = edited_df.values
else:
    st.subheader("📥 Upload CSV Dataset")
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            st.error("No numeric columns found in CSV!")
            st.stop()
        edited_df = st.data_editor(numeric_df, num_rows="dynamic")
        data = edited_df.values

if data is None:
    st.stop()

# ======== QPCA COMPUTATION =========
st.header("🧮 QPCA Computation")
N, M = data.shape
rho = Qobj(np.zeros((M, M), dtype=complex), dims=[[M], [M]])

for i in range(N):
    x = data[i]
    norm = np.linalg.norm(x)
    if norm == 0:
        st.error("Zero-norm vector encountered")
        st.stop()
    x_norm = x / norm
    ket = Qobj(x_norm, dims=[[M], [1]])
    rho += (ket * ket.dag()) / N
    if show_steps:
        st.write(f"Sample {i+1} contribution (trace = {((ket*ket.dag())/N).tr():.4f})")

st.markdown("**Final Density Matrix (ρ):**")
st.write(np.real_if_close(rho.full()))
st.write(f"Trace(ρ) = {rho.tr():.4f}")

if show_density_plot:
    fig, ax = matrix_histogram(rho)
    ax.set_title("Density Matrix (ρ)")
    st.pyplot(fig)

# ======== EIGENVALUES & PRINCIPAL COMPONENTS =========
eigenvalues, eigenvectors = rho.eigenstates()
order = np.argsort(eigenvalues)[::-1]
eigenvalues = np.real_if_close(eigenvalues[order])
eigenvectors = [eigenvectors[i] for i in order]

st.subheader("Eigenvalues")
st.write(eigenvalues)

default_k = min(3, len(eigenvalues))
k = st.slider("Select number of principal components", 1, len(eigenvalues), value=default_k)

st.subheader("Principal Components")
pcs = []
for i in range(k):
    pc_vec = np.real_if_close(eigenvectors[i].full().flatten())
    pcs.append(pc_vec)
    st.markdown(f"**PC {i+1} — Variance: {eigenvalues[i]:.4f}**")
    st.write(pc_vec)

# ======== INTERACTIVE BLOCH SPHERE FUNCTION =========
def plot_bloch_sphere_interactive(vectors, pc_labels=None):
    fig = go.Figure()

    # Sphere mesh
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.2,
                             colorscale=[[0, 'lightblue'], [1, 'lightblue']], showscale=False))

    # Axes
    axes = np.eye(3)
    axes_labels = ['X', 'Y', 'Z']
    for i in range(3):
        fig.add_trace(go.Scatter3d(x=[0, axes[i,0]], y=[0, axes[i,1]], z=[0, axes[i,2]],
                                   mode='lines+text', line=dict(color='black', width=4),
                                   text=[None, axes_labels[i]], textposition='top center'))

    # Vectors
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink']
    for i, vec in enumerate(vectors):
        fig.add_trace(go.Scatter3d(
            x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
            mode='lines+markers+text',
            marker=dict(size=4, color=colors[i % len(colors)]),
            line=dict(width=5, color=colors[i % len(colors)]),
            text=[None, pc_labels[i] if pc_labels else f"PC{i+1}"],
            textposition='top center'
        ))

    fig.update_layout(scene=dict(
        xaxis=dict(range=[-1,1], showbackground=False),
        yaxis=dict(range=[-1,1], showbackground=False),
        zaxis=dict(range=[-1,1], showbackground=False),
        aspectmode='cube'
    ),
    width=700, height=700, margin=dict(r=20, l=20, b=20, t=20))
    return fig

# ======== BLOCH SPHERE VISUALIZATION (INTERACTIVE) =========
if show_bloch:
    st.subheader("Bloch Sphere Visualization (Interactive)")
    for i, pc in enumerate(pcs):
        vecs = [pc[j*3:(j+1)*3] for j in range(len(pc)//3)]
        labels = [f"PC {i+1}"] * len(vecs)
        fig_bloch = plot_bloch_sphere_interactive(vecs, labels)
        st.plotly_chart(fig_bloch, use_container_width=True)

# ======== SCREE PLOT =========
if show_scree:
    st.subheader("Scree Plot")
    fig2, ax2 = plt.subplots()
    ax2.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Eigenvalue")
    ax2.set_title("Scree Plot")
    st.pyplot(fig2)

# ======== EXPORT =========
st.subheader("Export PCA Results")
export_df = pd.DataFrame(pcs, columns=[f"Feature {i+1}" for i in range(M)])
csv_file = export_df.to_csv(index=False)
st.download_button("Download CSV", csv_file, "qpca_results.csv")
st.download_button("Download LaTeX", export_df.to_latex(index=False), "qpca_results.tex")
