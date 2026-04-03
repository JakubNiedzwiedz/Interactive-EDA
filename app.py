import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
 
st.set_page_config(page_title="EDA Explorer", layout="wide")
st.title("Eksploracyjna Analiza Danych")

# Kilka zestawów wbudowanych danych
 
BUILTIN = {
    "Taksówki NYC (taxis)": "taxis",
    "Kwiaty Irysów (iris)": "iris",
    "Pingwiny (penguins)": "penguins",
    "Diamenty (diamonds)": "diamonds",
    "Titanic": "titanic",
}
 
@st.cache_data
def load_builtin(name):
    return sns.load_dataset(name)
 
@st.cache_data
def load_file(data, ext):
    buf = io.BytesIO(data)
    if ext == "csv":
        return pd.read_csv(buf)
    elif ext in ("xls", "xlsx"):
        return pd.read_excel(buf)
    elif ext == "json":
        return pd.read_json(buf)

# Sidebar

st.sidebar.header("Dane")
source = st.sidebar.radio("Źródło", ["Wbudowany zbiór", "Wgraj plik"])
 
df = None
if source == "Wbudowany zbiór":
    choice = st.sidebar.selectbox("Zbiór danych", list(BUILTIN.keys()))
    df = load_builtin(BUILTIN[choice])
else:
    file = st.sidebar.file_uploader("Plik CSV / Excel / JSON", type=["csv", "xlsx", "xls", "json"])
    if file:
        ext = file.name.rsplit(".", 1)[-1].lower()
        df = load_file(file.read(), ext)
 
if df is None:
    st.info("Wybierz lub wgraj zbiór danych.")
    st.stop()

# Filtry kategoryczne

cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
 
st.sidebar.header("Filtry")
for col in cat_cols[:3]:
    vals = sorted(df[col].dropna().unique().tolist())
    if len(vals) <= 20:
        sel = st.sidebar.multiselect(col, vals, default=vals)
        df = df[df[col].isin(sel)]

# zawartośc 

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Przegląd", "Statystyki", "Rozkłady", "Relacje", "Korelacje", "Wartości odstające"
])

#1. Przegląd

with tab1:
    st.subheader("Podgląd danych")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wiersze", len(df))
    c2.metric("Kolumny", df.shape[1])
    c3.metric("Brakujące wartości", int(df.isnull().sum().sum()))
    c4.metric("Duplikaty", int(df.duplicated().sum()))

    st.subheader("Typy danych")
    st.dataframe(df.dtypes.rename("typ").reset_index().rename(columns={"index": "kolumna"}),
                 use_container_width=True, hide_index=True)

# 2. Statystyki

with tab2:
    st.subheader("Statystyki opisowe — zmienne numeryczne")
    if num_cols:
        st.dataframe(df[num_cols].describe(), use_container_width=True)
    else:
        st.info("Brak kolumn numerycznych.")

    st.subheader("Statystyki opisowe — zmienne kategoryczne")
    if cat_cols:
        st.dataframe(df[cat_cols].describe(), use_container_width=True)
    else:
        st.info("Brak kolumn kategorycznych.")

    st.subheader("Brakujące wartości")
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if miss.empty:
        st.success("Brak wartości brakujących.")
    else:
        st.dataframe(miss.rename("braki"), use_container_width=True)

# 3. Rozkłady

with tab3:
    st.subheader("Histogram zmiennej numerycznej")
    if num_cols:
        col = st.selectbox("Zmienna", num_cols, key="hist_col")
        bins = st.slider("Liczba przedziałów", 5, 100, 30)
        kde  = st.checkbox("Pokaż KDE", True)
 
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.histplot(df[col].dropna(), bins=bins, kde=kde, ax=ax)
        ax.set_title(f"Histogram: {col}")
        ax.set_xlabel(col)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
 
        data = df[col].dropna()
        skew = data.skew()
        kurt = data.kurt()
        st.write(f"Skośność: **{skew:.3f}** | Kurtoza: **{kurt:.3f}**")
        if len(data) >= 8:
            _, p = stats.shapiro(data[:5000])
            result = "Normalny (p > 0.05)" if p > 0.05 else "Nie-normalny (p ≤ 0.05)"
            st.write(f"Test Shapiro-Wilk: p = {p:.4f} — {result}")
    else:
        st.info("Brak kolumn numerycznych.")
 
    st.subheader("Wykres słupkowy zmiennej kategorycznej")
    if cat_cols:
        col_cat = st.selectbox("Zmienna kategoryczna", cat_cols, key="bar_col")
        top_n = st.slider("Top N kategorii", 3, 30, 10)
 
        vc = df[col_cat].value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.barh(vc.index.astype(str)[::-1], vc.values[::-1])
        ax.set_title(f"Rozkład: {col_cat}")
        ax.set_xlabel("Liczba")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
    else:
        st.info("Brak kolumn kategorycznych.")

# 4. Relacje

with tab4:
    st.subheader("Scatter plot")
    if len(num_cols) >= 2:
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("Oś X", num_cols, key="sc_x")
        y = c2.selectbox("Oś Y", num_cols, index=1, key="sc_y")
        hue_opt = ["Brak"] + cat_cols
        hue = c3.selectbox("Kolor (hue)", hue_opt, key="sc_hue")
        reg = st.checkbox("Linia regresji", True)
 
        fig, ax = plt.subplots(figsize=(7, 4))
        h = hue if hue != "Brak" else None
        if reg and h is None:
            sns.regplot(data=df, x=x, y=y, ax=ax, scatter_kws={"alpha": 0.5})
        else:
            sns.scatterplot(data=df, x=x, y=y, hue=h, alpha=0.5, ax=ax)
        ax.set_title(f"{y} vs {x}")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
    else:
        st.info("Potrzebne co najmniej 2 kolumny numeryczne.")
 
    st.subheader("Boxplot")
    if num_cols and cat_cols:
        c1, c2, c3 = st.columns(3)
        y_bx = c1.selectbox("Zmienna numeryczna", num_cols, key="bx_y")
        x_bx = c2.selectbox("Zmienna kategoryczna", cat_cols, key="bx_x")
        btype = c3.radio("Typ", ["Box", "Violin"], horizontal=True)
 
        top = df[x_bx].value_counts().head(10).index
        df_bx = df[df[x_bx].isin(top)]
 
        fig, ax = plt.subplots(figsize=(7, 4))
        if btype == "Box":
            sns.boxplot(data=df_bx, x=x_bx, y=y_bx, ax=ax)
        else:
            sns.violinplot(data=df_bx, x=x_bx, y=y_bx, ax=ax)
        ax.set_title(f"{y_bx} według {x_bx}")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
    else:
        st.info("Potrzebna co najmniej 1 kolumna numeryczna i 1 kategoryczna.")

# 5. Korelacje

with tab5:
    st.subheader("Macierz korelacji")
    if len(num_cols) >= 2:
        method = st.selectbox("Metoda", ["pearson", "spearman", "kendall"])
        corr = df[num_cols].corr(method=method)
 
        n = len(num_cols)
        fig, ax = plt.subplots(figsize=(n * 0.9 + 1.5, n * 0.8 + 1))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    mask=np.triu(np.ones_like(corr, dtype=bool)),
                    linewidths=0.5, annot_kws={"size": 9}, ax=ax)
        ax.set_title(f"Korelacje ({method})")
        ax.tick_params(labelsize=9)
        plt.tight_layout()
 
        col_plot, _ = st.columns([1, 1])
        with col_plot:
            st.pyplot(fig, use_container_width=True)
    else:
        st.info("Potrzebne co najmniej 2 kolumny numeryczne.")


# 6. Wartości odstające
with tab6:
    st.subheader("Wykrywanie wartości odstających")
    if not num_cols:
        st.info("Brak kolumn numerycznych.")
    else:
        col_out = st.selectbox("Zmienna", num_cols, key="out_col")
 
        data = df[col_out].dropna()
 
        z = np.abs(stats.zscore(data))
        outliers = data[z > 3]
        lo = data.mean() - 3 * data.std()
        hi = data.mean() + 3 * data.std()
 
        st.write(f"Wykryto **{len(outliers)}** wartości odstających "
                 f"({len(outliers)/len(data)*100:.2f}%) | Zakres normalny: [{lo:.3f}, {hi:.3f}]")
 
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
 
        # Boxplot
        axes[0].boxplot(data, vert=True, patch_artist=True)
        axes[0].set_title("Boxplot")
        axes[0].set_ylabel(col_out)
 
        # Scatter z zaznaczonymi outliersami
        inliers = data[~data.index.isin(outliers.index)]
        axes[1].scatter(range(len(inliers)), inliers.values, s=10, alpha=0.4, label="Normalne")
        if not outliers.empty:
            out_positions = [list(data.index).index(i) for i in outliers.index]
            axes[1].scatter(out_positions, outliers.values, s=40, color="red",
                            zorder=5, label=f"Odstające ({len(outliers)})")
        axes[1].axhline(lo, color="gray", ls="--", lw=1, label=f"Granica dolna ({lo:.2f})")
        axes[1].axhline(hi, color="gray", ls=":",  lw=1, label=f"Granica górna ({hi:.2f})")
        axes[1].legend(fontsize=8)
        axes[1].set_title("Wartości w kolejności obserwacji")
        axes[1].set_xlabel("Indeks")
        axes[1].set_ylabel(col_out)
 
        plt.tight_layout()
        st.pyplot(fig)