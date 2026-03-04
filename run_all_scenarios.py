import numpy as np

from simulation import run_simulation, save_scenario_plot


# ---- Сценарий 1. Штатный режим ----
print("Запуск сценария 1...")
results1 = run_simulation(
    ils_noise=np.radians(0.01),
    gbas_noise=0.5,
    failure_time_gbas=120,      # отказ GBAS вне интервала моделирования (сигнал не пропадает)
    multipath_time=None,
    ils_failure_time=None,
    jamming_time=None,
    spoofing_time=None
)
save_scenario_plot(
    results1,
    "scenario1_normal.png",
    "Сценарий 1: Штатный режим"
)


# ---- Сценарий 2. Отказ ILS на 80 с ----
print("Запуск сценария 2...")
results2 = run_simulation(
    ils_noise=np.radians(0.01),
    gbas_noise=0.5,
    failure_time_gbas=120,
    multipath_time=None,
    ils_failure_time=80,         # отказ ILS
    jamming_time=None,
    spoofing_time=None
)
save_scenario_plot(
    results2,
    "scenario2_ils_failure.png",
    "Сценарий 2: Отказ ILS на 80 с"
)


# ---- Сценарий 3. Потеря GBAS после 50 с (упрощённо, без восстановления) ----
print("Запуск сценария 3 (упрощённо)...")
results3 = run_simulation(
    ils_noise=np.radians(0.01),
    gbas_noise=0.5,
    failure_time_gbas=50,        # сигнал GBAS пропадает после 50 с и не восстанавливается
    multipath_time=None,
    ils_failure_time=None,
    jamming_time=None,
    spoofing_time=None
)
save_scenario_plot(
    results3,
    "scenario3_gbas_loss.png",
    "Сценарий 3: Потеря GBAS после 50 с"
)


# ---- Сценарий 4. Мультипуть ILS с 40 с ----
print("Запуск сценария 4...")
results4 = run_simulation(
    ils_noise=np.radians(0.01),
    gbas_noise=0.5,
    failure_time_gbas=120,
    multipath_time=40,           # мультипуть после 40 с
    ils_failure_time=None,
    jamming_time=None,
    spoofing_time=None
)
save_scenario_plot(
    results4,
    "scenario4_multipath.png",
    "Сценарий 4: Мультипуть ILS с 40 с"
)


print("Все графики сохранены!")

