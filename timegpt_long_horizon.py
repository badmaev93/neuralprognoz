"""
TimeGPT Long-Horizon прогноз приттока в Колымское водохранилище
Горизонт: h=9 (январь–сентябрь)
Заблаговременность для июля/августа/сентября: 7/8/9 месяцев
Тестовый период: 2015–2024
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from nixtla import NixtlaClient

# ============================================================
# CONFIG
# ============================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
NC_PATH    = os.path.join(BASE_DIR, "data/generic_dataset/time_series/KR.nc")
OUT_DIR    = BASE_DIR

TIMEGPT_API_KEY  = os.environ.get("TIMEGPT_API_KEY", "your_api_key_here")
MODEL            = "timegpt-1-long-horizon"   # длинный горизонт; "timegpt-1" — короткий
HORIZON          = 9                          # январь…сентябрь
FINETUNE_STEPS   = 50                         # 0 — zero-shot, >0 — fine-tuning
TEST_YEARS       = list(range(2015, 2025))    # 2015–2024 включительно
TARGET_MONTHS    = {7: "Июль", 8: "Август", 9: "Сентябрь"}

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================
ds_nc = xr.open_dataset(NC_PATH)
df_raw = ds_nc[["streamflow_m3s"]].to_dataframe().reset_index()
df_raw = df_raw.rename(columns={"date": "ds", "streamflow_m3s": "y"})
df_raw["unique_id"] = "KR"
df_raw["ds"] = pd.to_datetime(df_raw["ds"])
df_raw = df_raw[["unique_id", "ds", "y"]].dropna().reset_index(drop=True)

print(f"Данные: {df_raw['ds'].min().date()} … {df_raw['ds'].max().date()}, "
      f"n={len(df_raw)}")

# ============================================================
# 2. КЛИЕНТ NIXTLA
# ============================================================
client = NixtlaClient(api_key=TIMEGPT_API_KEY)

# ============================================================
# 3. ROLLING FORECAST
#    Для каждого тестового года t:
#      - история: все данные по 31 декабря (t-1)
#      - прогноз: 9 шагов = январь(t) … сентябрь(t)
#      - извлекаем: шаги 6,7,8 → июль, август, сентябрь
# ============================================================
print(f"\nRolling forecast: model={MODEL}, h={HORIZON}, "
      f"finetune_steps={FINETUNE_STEPS}")
print("-" * 50)

records = []
for year in TEST_YEARS:
    cutoff = pd.Timestamp(f"{year - 1}-12-01")
    df_input = df_raw[df_raw["ds"] <= cutoff].copy()

    print(f"  {year}: история до {cutoff.date()}, n_train={len(df_input)} ...", end=" ")

    fcst = client.forecast(
        df=df_input,
        h=HORIZON,
        model=MODEL,
        finetune_steps=FINETUNE_STEPS,
        time_col="ds",
        target_col="y",
        id_col="unique_id",
        freq="MS",   # Month Start — начало месяца
    )

    fcst["ds"] = pd.to_datetime(fcst["ds"])
    # шаги горизонта относительно cutoff+1: январь(t) = шаг 1 … сентябрь(t) = шаг 9
    fcst_q3 = fcst[fcst["ds"].dt.month.isin([7, 8, 9])].copy()
    records.append(fcst_q3)

    for m, name in TARGET_MONTHS.items():
        row = fcst_q3[fcst_q3["ds"].dt.month == m]
        pred_val = row["TimeGPT"].values[0] if len(row) else float("nan")
        print(f"{name[:3]}={pred_val:.1f}", end=" ")
    print()

preds_df = pd.concat(records, ignore_index=True)

# ============================================================
# 4. ОБЪЕДИНЕНИЕ С НАБЛЮДЕНИЯМИ
# ============================================================
obs_df = df_raw.rename(columns={"y": "obs"})[["ds", "obs"]]
result_df = preds_df.merge(obs_df, on="ds", how="left")
result_df["month"] = result_df["ds"].dt.month
result_df["year"]  = result_df["ds"].dt.year

# ============================================================
# 5. МЕТРИКА S/σΔ
# ============================================================
def s_sigma(obs: np.ndarray, pred: np.ndarray) -> float:
    rmse  = np.sqrt(np.mean((obs - pred) ** 2))
    sigma = np.std(obs, ddof=1)
    return rmse / sigma

pred_col = "TimeGPT"   # nixtla возвращает колонку "TimeGPT"

print(f"\nS/σΔ на независимом тестовом периоде 2015–2024")
print(f"Модель: {MODEL}  |  horizon={HORIZON}  |  finetune_steps={FINETUNE_STEPS}")
print("-" * 45)
print(f"{'Месяц':<12} {'S/σΔ':>6}   {'Оценка'}")
print("-" * 45)

scores = {}
for m, name in TARGET_MONTHS.items():
    grp  = result_df[result_df["month"] == m]
    val  = s_sigma(grp["obs"].values, grp[pred_col].values)
    tag  = "Хорошо" if val <= 0.50 else ("Удовл." if val <= 0.80 else "Неудовл.")
    marker = "*" if val <= 0.80 else ""
    print(f"{name:<12} {val:>6.3f}{marker}  {tag}")
    scores[m] = val

print("-" * 45)

# ============================================================
# 6. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================
csv_path = os.path.join(OUT_DIR, "timegpt_lh_predictions.csv")
result_df.to_csv(csv_path, index=False)
print(f"\nПредсказания сохранены: {csv_path}")

# ============================================================
# 7. ВИЗУАЛИЗАЦИЯ
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"TimeGPT Long-Horizon (h=9) — прогноз июль/август/сентябрь\n"
    f"Тестовый период 2015–2024, finetune_steps={FINETUNE_STEPS}",
    fontsize=13,
)

for ax, (m, name) in zip(axes, TARGET_MONTHS.items()):
    grp = result_df[result_df["month"] == m].sort_values("year")
    ax.plot(grp["year"], grp["obs"],      "k-o",  label="Наблюдения", linewidth=1.5)
    ax.plot(grp["year"], grp[pred_col],   "r--s", label="TimeGPT",    linewidth=1.5)
    val = scores[m]
    color = "green" if val <= 0.80 else "red"
    ax.set_title(f"{name}\nS/σΔ = {val:.3f}", color=color, fontsize=11)
    ax.set_xlabel("Год")
    ax.set_ylabel("Q, м³/с")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "timegpt_lh_s_sigma_q3.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"График сохранён: {fig_path}")
