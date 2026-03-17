import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_URLS = [
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/nycflights13/flights.csv",
    "https://raw.githubusercontent.com/tidyverse/nycflights13/main/data-raw/flights.csv",
    "https://raw.githubusercontent.com/ismayc/nycflights13/main/data-raw/flights.csv",
]

AXIS_LABEL_ATRASO_MEDIO = "Atraso medio (min)"


def ensure_folders() -> Dict[str, Path]:
    base = Path.cwd()
    output = base / "output"
    figures = output / "figuras"
    tables = output / "tabelas"
    output.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    return {"base": base, "output": output, "figures": figures, "tables": tables}


def load_dataset() -> pd.DataFrame:
    local_file = Path("flights.csv")
    if local_file.exists():
        print(f"[INFO] Carregando dataset local: {local_file}")
        return pd.read_csv(local_file)

    for url in DATA_URLS:
        try:
            print(f"[INFO] Tentando baixar dataset: {url}")
            return pd.read_csv(url)
        except Exception as exc:
            print(f"[WARN] Falha ao carregar URL: {url} -> {exc}")

    raise RuntimeError(
        "Nao foi possivel carregar o dataset. "
        "Coloque um arquivo flights.csv na mesma pasta do script."
    )


def basic_eda(df: pd.DataFrame, folders: Dict[str, Path]) -> None:
    print("\n===== EDA: estrutura =====")
    print(df.head())
    print("\n===== EDA: info =====")
    df.info()
    print("\n===== EDA: describe numerico =====")
    print(df.describe(include=[np.number]).T.head(20))

    missing = pd.DataFrame(
        {
            "faltantes": df.isna().sum(),
            "percentual": (df.isna().mean() * 100).round(2),
        }
    ).sort_values("faltantes", ascending=False)
    missing.to_csv(folders["tables"] / "valores_ausentes.csv")
    print("\n===== EDA: valores ausentes (top 10) =====")
    print(missing.head(10))

    sns.set_theme(style="whitegrid")

    # Distribuicao de atraso
    plt.figure(figsize=(10, 5))
    arr_delay = df["arr_delay"].dropna().clip(lower=-60, upper=240)
    sns.histplot(arr_delay, bins=80, kde=True)
    plt.title("Distribuicao de atraso de chegada (clip -60 a 240 min)")
    plt.xlabel("Atraso chegada (min)")
    plt.ylabel("Frequencia")
    plt.tight_layout()
    plt.savefig(folders["figures"] / "distribuicao_arr_delay.png", dpi=140)
    plt.close()

    # Atraso por origem
    by_origin = (
        df.groupby("origin", as_index=False)["arr_delay"]
        .mean()
        .sort_values("arr_delay", ascending=False)
    )
    by_origin.to_csv(folders["tables"] / "atraso_por_origem.csv", index=False)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=by_origin, x="origin", y="arr_delay")
    plt.title("Atraso medio por aeroporto de origem")
    plt.xlabel("Aeroporto")
    plt.ylabel(AXIS_LABEL_ATRASO_MEDIO)
    plt.tight_layout()
    plt.savefig(folders["figures"] / "atraso_por_origem.png", dpi=140)
    plt.close()

    # Atraso por companhia
    by_carrier = (
        df.groupby("carrier", as_index=False)["arr_delay"]
        .mean()
        .sort_values("arr_delay", ascending=False)
    )
    by_carrier.to_csv(folders["tables"] / "atraso_por_companhia.csv", index=False)
    plt.figure(figsize=(12, 5))
    sns.barplot(data=by_carrier, x="carrier", y="arr_delay")
    plt.title("Atraso medio por companhia aerea")
    plt.xlabel("Companhia")
    plt.ylabel(AXIS_LABEL_ATRASO_MEDIO)
    plt.tight_layout()
    plt.savefig(folders["figures"] / "atraso_por_companhia.png", dpi=140)
    plt.close()

    # Atraso por hora
    by_hour = (
        df.groupby("hour", as_index=False)["arr_delay"]
        .mean()
        .sort_values("hour")
    )
    by_hour.to_csv(folders["tables"] / "atraso_por_hora.csv", index=False)
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=by_hour, x="hour", y="arr_delay", marker="o")
    plt.title("Atraso medio por hora prevista de partida")
    plt.xlabel("Hora")
    plt.ylabel(AXIS_LABEL_ATRASO_MEDIO)
    plt.tight_layout()
    plt.savefig(folders["figures"] / "atraso_por_hora.png", dpi=140)
    plt.close()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [c.upper() for c in data.columns]

    data["DATA_VOO"] = pd.to_datetime(
        data[["YEAR", "MONTH", "DAY"]].rename(
            columns={"YEAR": "year", "MONTH": "month", "DAY": "day"}
        ),
        errors="coerce",
    )
    data["DIA_SEMANA"] = data["DATA_VOO"].dt.dayofweek

    if "HOUR" in data.columns:
        data["HOUR"] = data["HOUR"].fillna((data["SCHED_DEP_TIME"] // 100).astype(float))
    else:
        data["HOUR"] = (data["SCHED_DEP_TIME"] // 100).astype(float)

    data.loc[~data["HOUR"].between(0, 23), "HOUR"] = np.nan

    def map_periodo(h: float) -> str:
        if pd.isna(h):
            return np.nan
        h = int(h)
        if 5 <= h <= 11:
            return "manha"
        if 12 <= h <= 17:
            return "tarde"
        return "noite"

    data["PERIODO_DIA"] = data["HOUR"].apply(map_periodo)
    data["ATRASO"] = np.where(data["ARR_DELAY"] > 0, 1, 0)

    return data


REGRESSION_FEATURE_COLS = [
    "MONTH",
    "DAY",
    "DIA_SEMANA",
    "HOUR",
    "DISTANCE",
    "AIR_TIME",
    "DEP_DELAY",
    "ORIGIN",
    "DEST",
    "CARRIER",
    "PERIODO_DIA",
]


def build_regression_dataset(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Dataset para regressão: prever quanto tempo o atraso de chegada vai durar (min)."""
    model_data = data[REGRESSION_FEATURE_COLS + ["ARR_DELAY"]].copy()
    model_data = model_data.loc[data["ARR_DELAY"].notna()].copy()
    # Clip target para reduzir impacto de outliers extremos
    model_data["ARR_DELAY"] = model_data["ARR_DELAY"].clip(lower=-60, upper=240)
    return model_data[REGRESSION_FEATURE_COLS], model_data["ARR_DELAY"]


def make_preprocessor() -> ColumnTransformer:
    categorical_features = ["ORIGIN", "DEST", "CARRIER", "PERIODO_DIA"]
    numeric_features = [
        "MONTH",
        "DAY",
        "DIA_SEMANA",
        "HOUR",
        "DISTANCE",
        "AIR_TIME",
        "DEP_DELAY",
    ]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))],
        memory=None,
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ],
        memory=None,
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def evaluate_regressor(
    name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    folders: Dict[str, Path],
) -> Dict[str, float]:
    """Avalia modelo de regressão com MAE, RMSE, R² e gráfico real vs predito."""
    pipeline.fit(x_train, y_train)
    pred = pipeline.predict(x_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    metrics = {
        "Modelo": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

    print(f"\n===== Regressão: {name} =====")
    print(f"  MAE  (min): {mae:.2f}")
    print(f"  RMSE (min): {rmse:.2f}")
    print(f"  R²:        {r2:.4f}")

    # Gráfico: valor real vs predito
    sample_size = min(5000, len(y_test))
    idx = np.random.RandomState(42).choice(len(y_test), sample_size, replace=False)
    y_s = y_test.iloc[idx].values if hasattr(y_test, "iloc") else y_test[idx]
    p_s = pred[idx]
    plt.figure(figsize=(6, 5))
    plt.scatter(y_s, p_s, alpha=0.3, s=15)
    max_val = max(y_s.max(), p_s.max())
    min_val = min(y_s.min(), p_s.min())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Predição ideal")
    plt.xlabel("Atraso real (min)")
    plt.ylabel("Atraso predito (min)")
    plt.title(f"Regressão: Real vs Predito - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        folders["figures"] / f"regressao_real_vs_pred_{name.lower().replace(' ', '_')}.png",
        dpi=140,
    )
    plt.close()

    return metrics


def run_regression(data: pd.DataFrame, folders: Dict[str, Path]) -> pd.DataFrame:
    """Regressão: prever quanto tempo o atraso vai durar. Compara 2+ algoritmos."""
    x, y = build_regression_dataset(data)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )
    print(f"\n[INFO] Regressão - Treino: {x_train.shape} | Teste: {x_test.shape}")
    print(f"[INFO] Target ARR_DELAY: min={y.min():.0f}, max={y.max():.0f} min (clip -60 a 240)")

    preprocessor = make_preprocessor()

    ridge = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", Ridge(alpha=1.0, random_state=42)),
        ],
        memory=None,
    )
    rf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ],
        memory=None,
    )

    m1 = evaluate_regressor("Ridge", ridge, x_train, x_test, y_train, y_test, folders)
    m2 = evaluate_regressor(
        "Random Forest", rf, x_train, x_test, y_train, y_test, folders
    )

    results = pd.DataFrame([m1, m2])
    results.to_csv(folders["tables"] / "comparacao_regressao.csv", index=False)
    print("\n===== Comparação dos modelos de regressão =====")
    print(results[["Modelo", "MAE", "RMSE", "R2"]])
    return results


def cluster_aeroportos(data: pd.DataFrame, folders: Dict[str, Path]) -> pd.DataFrame:
    """Clusterização de aeroportos: agrupar origens com perfil de atraso/volume semelhante."""
    agg = (
        data.groupby("ORIGIN", as_index=False)
        .agg(
            atraso_medio_chegada=("ARR_DELAY", "mean"),
            atraso_medio_partida=("DEP_DELAY", "mean"),
            n_voos=("ORIGIN", "count"),
            distancia_media=("DISTANCE", "mean"),
        )
        .dropna()
    )
    agg["atraso_medio_chegada"] = agg["atraso_medio_chegada"].clip(-30, 120)
    agg["atraso_medio_partida"] = agg["atraso_medio_partida"].clip(-30, 120)

    feat_cols = ["atraso_medio_chegada", "atraso_medio_partida", "n_voos", "distancia_media"]
    X = agg[feat_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    agg["CLUSTER"] = kmeans.fit_predict(X_scaled)

    # Gráfico: aeroportos por cluster (atraso chegada x volume)
    plt.figure(figsize=(9, 5))
    sns.scatterplot(
        data=agg,
        x="n_voos",
        y="atraso_medio_chegada",
        hue="CLUSTER",
        size="distancia_media",
        sizes=(40, 200),
        alpha=0.8,
        palette="Set2",
    )
    for _, row in agg.iterrows():
        plt.annotate(row["ORIGIN"], (row["n_voos"], row["atraso_medio_chegada"]), fontsize=9)
    plt.title("Clusterização de aeroportos (origem): perfil de volume e atraso")
    plt.xlabel("Número de voos")
    plt.ylabel("Atraso médio de chegada (min)")
    plt.legend(title="Cluster", loc="best")
    plt.tight_layout()
    plt.savefig(folders["figures"] / "clusters_aeroportos.png", dpi=140)
    plt.close()

    resumo = agg.groupby("CLUSTER", as_index=False)[feat_cols].mean().round(2)
    resumo.to_csv(folders["tables"] / "resumo_clusters_aeroportos.csv", index=False)
    agg[["ORIGIN", "CLUSTER"] + feat_cols].to_csv(
        folders["tables"] / "aeroportos_por_cluster.csv", index=False
    )

    print("\n===== Clusterização de aeroportos (interpretação) =====")
    print(resumo)
    print("Aeroportos por cluster:")
    for c in sorted(agg["CLUSTER"].unique()):
        oris = agg.loc[agg["CLUSTER"] == c, "ORIGIN"].tolist()
        print(f"  Cluster {c}: {', '.join(oris)}")
    return resumo


def run_kmeans(data: pd.DataFrame, folders: Dict[str, Path]) -> None:
    """Modelagem não supervisionada: clusters de voos + clusters de aeroportos."""
    # 1) Clusters de voos (perfil distância e atrasos)
    cluster_vars = ["DISTANCE", "DEP_DELAY", "ARR_DELAY"]
    cluster_data = data[cluster_vars].dropna().copy()
    cluster_data["DEP_DELAY"] = cluster_data["DEP_DELAY"].clip(-30, 300)
    cluster_data["ARR_DELAY"] = cluster_data["ARR_DELAY"].clip(-60, 300)

    scaler = StandardScaler()
    x_cluster = scaler.fit_transform(cluster_data)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_data["CLUSTER"] = kmeans.fit_predict(x_cluster)

    sample = cluster_data.sample(min(20000, len(cluster_data)), random_state=42)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=sample,
        x="DEP_DELAY",
        y="ARR_DELAY",
        hue="CLUSTER",
        palette="Set2",
        alpha=0.6,
        s=25,
    )
    plt.title("Clusters de voos: perfil de atraso na partida x chegada")
    plt.xlabel("Atraso partida (min)")
    plt.ylabel("Atraso chegada (min)")
    plt.tight_layout()
    plt.savefig(folders["figures"] / "clusters_scatter.png", dpi=140)
    plt.close()

    summary = (
        cluster_data.groupby("CLUSTER", as_index=False)[cluster_vars]
        .mean()
        .round(2)
        .sort_values("CLUSTER")
    )
    summary.to_csv(folders["tables"] / "resumo_clusters.csv", index=False)
    print("\n===== KMeans voos: resumo (interpretação) =====")
    print(summary)
    print("Cluster 0: voos com atrasos baixos; Cluster 1: distâncias longas; Cluster 2: atrasos altos.")

    # 2) Clusterização de aeroportos
    cluster_aeroportos(data, folders)


def save_business_answers(data: pd.DataFrame, folders: Dict[str, Path]) -> None:
    by_origin = (
        data.groupby("ORIGIN", as_index=False)["ARR_DELAY"]
        .mean()
        .sort_values("ARR_DELAY", ascending=False)
    )
    by_carrier = (
        data.groupby("CARRIER", as_index=False)["ARR_DELAY"]
        .mean()
        .sort_values("ARR_DELAY", ascending=False)
    )
    by_weekday = (
        data.groupby("DIA_SEMANA", as_index=False)["ARR_DELAY"]
        .mean()
        .sort_values("DIA_SEMANA")
    )

    by_origin.to_csv(folders["tables"] / "resposta_aeroportos.csv", index=False)
    by_carrier.to_csv(folders["tables"] / "resposta_companhias.csv", index=False)
    by_weekday.to_csv(folders["tables"] / "resposta_dia_semana.csv", index=False)

    print("\n===== Respostas de negocio =====")
    print("Top aeroportos com maior atraso medio:")
    print(by_origin.head(5))
    print("\nTop companhias com maior atraso medio:")
    print(by_carrier.head(10))


def main() -> None:
    print("Projeto: Analise e Predicao de Atrasos de Voos")
    folders = ensure_folders()
    df = load_dataset()

    print(f"[INFO] Shape dataset bruto: {df.shape}")
    basic_eda(df, folders)

    data = add_features(df)
    print(f"[INFO] Shape apos engenharia de features: {data.shape}")

    _ = run_regression(data, folders)
    run_kmeans(data, folders)
    save_business_answers(data, folders)

    print("\nConcluido com sucesso.")
    print(f"Arquivos gerados em: {folders['output']}")


if __name__ == "__main__":
    main()
