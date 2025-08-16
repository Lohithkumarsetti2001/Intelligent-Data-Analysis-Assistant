
import io
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hist(df, col):
    fig, ax = plt.subplots()
    sns.histplot(df[col].dropna(), ax=ax, kde=True)
    ax.set_title(f"Distribution of {col}")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return fig, buf

def plot_bar_counts(df, col):
    fig, ax = plt.subplots()
    df[col].value_counts().head(20).plot(kind="bar", ax=ax)
    ax.set_title(f"Top {col} categories")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return fig, buf

def plot_scatter(df, x, y):
    fig, ax = plt.subplots()
    ax.scatter(df[x], df[y])
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{x} vs {y}")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return fig, buf
