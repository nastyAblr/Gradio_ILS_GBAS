import numpy as np
import matplotlib.pyplot as plt
from models import LocalFrame, ILSModel, GBASModel, IRSModel
from kalman_filter import EKF
from config import *

def generate_trajectory(duration=120, dt=0.1, initial_distance=10000):
    t = np.arange(0, duration, dt)
    n = len(t)
    x_start = -initial_distance
    x_end = 200
    x = np.linspace(x_start, x_end, n)
    z = -np.tan(GPA) * x + TCH
    z = np.maximum(z, 0)
    y = np.zeros(n)
    local = LocalFrame(LTP_ECEF, FPAP_ECEF, TCP_ECEF)

    def local_to_ecef(xyz_local):
        delta = np.linalg.inv(local.R) @ xyz_local
        return local.r_ltp + delta

    r_ac_list = [local_to_ecef([x[i], y[i], z[i]]) for i in range(n)]
    return t, np.array(r_ac_list), local

def run_simulation(ils_noise, gbas_noise, failure_time_gbas, multipath_time=None,
                   ils_failure_time=None, jamming_time=None, spoofing_time=None,
                   iono_gradient=IONO_GRADIENT):
    dt = DT
    duration = 120
    t, r_ac_true, local = generate_trajectory(duration, dt)

    # Инициализация моделей
    ils = ILSModel(LOC_ECEF, GS_ECEF, GPA, ils_noise, ils_noise, MULTIPATH_MAGNITUDE)
    if multipath_time is not None and multipath_time < duration:
        ils.set_multipath(multipath_time)
    if ils_failure_time is not None and ils_failure_time < duration:
        ils.set_failure(ils_failure_time)

    gbas = GBASModel(gbas_noise, iono_gradient)
    if failure_time_gbas < duration:
        gbas.set_failure(failure_time_gbas)
    if jamming_time is not None and jamming_time < duration:
        gbas.set_jamming(jamming_time)
    if spoofing_time is not None and spoofing_time < duration:
        gbas.set_spoofing(spoofing_time)

    irs = IRSModel(dt)

    Q = np.diag([
        PROCESS_NOISE_POS, PROCESS_NOISE_POS, PROCESS_NOISE_POS,
        PROCESS_NOISE_VEL, PROCESS_NOISE_VEL, PROCESS_NOISE_VEL,
        PROCESS_NOISE_ACC, PROCESS_NOISE_ACC, PROCESS_NOISE_ACC,
        PROCESS_NOISE_BIAS, PROCESS_NOISE_BIAS, PROCESS_NOISE_BIAS
    ])

    ekf = EKF(dt, Q, R_ILS_NOM, R_GBAS_NOM)

    error_hybrid, error_ils, error_gbas = [], [], []
    # mode_history будет содержать активный режим на каждом такте моделирования
    # и далее используется для визуализации в интерфейсе.
    mode_history = []
    mode = 'HYBRID'

    for i, ti in enumerate(t):
        r_ac = r_ac_true[i]
        true_local = local.ecef_to_local(r_ac)
        x_true, y_true, z_true = true_local

        # Измерения
        ils_meas = ils.get_measurement(r_ac, local, ti)
        ils_sqi = ils.get_sqi(ti)
        gbas_meas = gbas.get_measurement(r_ac, local, ti, LTP_ECEF)
        gbas_sqi = gbas.get_sqi(ti)

        # Инициализация EKF и IRS по первому доступному решению GBAS:
        # при наличии корректного дифференциального решения считаем его
        # наилучшей стартовой точкой инициализации состояния.
        if i == 0 and gbas_meas is not None:
            ekf.x[0] = gbas_meas['x']
            ekf.x[1] = gbas_meas['y']
            ekf.x[2] = gbas_meas['z']
            ekf.P = np.eye(12) * 10.0
            irs.set_initial(np.array([gbas_meas['x'], gbas_meas['y'], gbas_meas['z']]),
                            np.array([-70.0, 0.0, -3.0]))  # примерная скорость

        # --- Логика переключения режимов ---
        # Основная идея:
        #   - по умолчанию работаем в гибридном режиме (HYBRID), используя все доступные источники;
        #   - при пропадании одного из них переходим в режим «только ILS» или «только GBAS»;
        #   - при полном отсутствии внешних измерений остаётся лишь инерциальная система (IRS_ONLY).
        if mode == 'HYBRID':
            if gbas_meas is None and ils_meas is None:
                mode = 'IRS_ONLY'
            elif gbas_meas is None:
                mode = 'ILS_ONLY'
            elif ils_meas is None:
                mode = 'GBAS_ONLY'
        elif mode == 'ILS_ONLY':
            if ils_meas is None:
                if gbas_meas is not None:
                    mode = 'GBAS_ONLY'
                else:
                    mode = 'IRS_ONLY'
        elif mode == 'GBAS_ONLY':
            if gbas_meas is None:
                if ils_meas is not None:
                    mode = 'ILS_ONLY'
                else:
                    mode = 'IRS_ONLY'
        elif mode == 'IRS_ONLY':
            # Выход из IRS_ONLY: достаточно восстановиться хотя бы одному источнику.
            # Для простоты приоритет отдан GBAS как более точному источнику.
            if gbas_meas is not None:
                mode = 'GBAS_ONLY'
            elif ils_meas is not None:
                mode = 'ILS_ONLY'

        mode_history.append(mode)

        # --- Обновление фильтра / IRS ---
        if mode == 'HYBRID':
            ekf.predict()
            if gbas_meas is not None:
                ekf.update_gbas(gbas_meas['x'], gbas_meas['y'], gbas_meas['z'], gbas_sqi)
            if ils_meas is not None:
                ekf.update_ils(ils_meas['y'], ils_meas['z'], ils_sqi)
            est_pos = ekf.get_position()
        elif mode == 'ILS_ONLY':
            ekf.predict()
            if ils_meas is not None:
                ekf.update_ils(ils_meas['y'], ils_meas['z'], ils_sqi)
            est_pos = ekf.get_position()
        elif mode == 'GBAS_ONLY':
            ekf.predict()
            if gbas_meas is not None:
                ekf.update_gbas(gbas_meas['x'], gbas_meas['y'], gbas_meas['z'], gbas_sqi)
            est_pos = ekf.get_position()
        elif mode == 'IRS_ONLY':
            est_pos = irs.predict()
            # При восстановлении источника можно скорректировать фильтр (здесь опущено)
        else:
            est_pos = ekf.get_position()  # fallback

        # Ошибки для сравнения: норма ошибки гибридной оценки относительно
        # истинного положения в локальной системе координат.
        err_hyb = np.linalg.norm(est_pos - true_local)
        error_hybrid.append(err_hyb)

        # Ошибка ILS-only (если есть измерение)
        if ils_meas is not None:
            err_ils = np.sqrt((ils_meas['y'] - y_true)**2 + (ils_meas['z'] - z_true)**2)
        else:
            err_ils = np.nan
        error_ils.append(err_ils)

        # Ошибка GBAS-only (если есть измерение)
        if gbas_meas is not None:
            err_g = np.linalg.norm([gbas_meas['x'] - x_true, gbas_meas['y'] - y_true, gbas_meas['z'] - z_true])
            error_gbas.append(err_g)
        else:
            error_gbas.append(np.nan)

    return {
        'time': t,
        'error_hybrid': np.array(error_hybrid),
        'error_ils': np.array(error_ils),
        'error_gbas': np.array(error_gbas),
        'mode_history': mode_history  # при желании можно использовать
    }


def save_scenario_plot(results, filename, title):
    """
    Сохраняет график с результатами моделирования в файл с высоким разрешением,
    пригодный для вставки в текст диплома.

    Параметры:
        results  : словарь, возвращаемый run_simulation
        filename : имя файла (например, 'scenario1.png')
        title    : заголовок графика
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)  # dpi=150 для базового отображения

    ax.plot(results['time'], results['error_hybrid'],
            label='Гибридная система (EKF)', linewidth=2, color='blue')
    ax.plot(results['time'], results['error_ils'],
            label='ILS only', linestyle='--', alpha=0.7, color='red')
    ax.plot(results['time'], results['error_gbas'],
            label='GBAS only', linestyle='--', alpha=0.7, color='green')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Время (с)', fontsize=12)
    ax.set_ylabel('Ошибка по положению (м)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # Для печати и вставки в Word используем 300 dpi и аккуратное обрезание полей.
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"График сохранён: {filename}")