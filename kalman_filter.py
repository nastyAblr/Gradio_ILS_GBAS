# kalman_filter.py
import numpy as np
from config import HAL_CAT1, VAL_CAT1, HAL_CAT2, VAL_CAT2, HAL_CAT3, VAL_CAT3

class EKF:
    """
    Расширенный фильтр Калмана для слияния данных ILS и GBAS.
    Вектор состояния (12 элементов):
        x[0] : x       - координата вдоль ВПП (м)
        x[1] : y       - боковое отклонение (м)
        x[2] : z       - высота (м)
        x[3] : vx      - скорость по x (м/с)
        x[4] : vy      - скорость по y (м/с)
        x[5] : vz      - скорость по z (м/с)
        x[6] : ax      - ускорение по x (м/с²)
        x[7] : ay      - ускорение по y (м/с²)
        x[8] : az      - ускорение по z (м/с²)
        x[9] : b_ils_y - смещение ILS по боковому каналу (м)
        x[10]: b_ils_z - смещение ILS по вертикальному каналу (м)
        x[11]: b_gbas_z- смещение GBAS по высоте (м) (для учёта ионосферных эффектов)
    """

    def __init__(self, dt, Q, R_ils_nom, R_gbas_nom):
        """
        Инициализация фильтра.

        Параметры:
            dt         : шаг дискретизации (с)
            Q          : матрица ковариации шума процесса (12x12)
            R_ils_nom  : номинальная ковариация измерений ILS (2x2)
            R_gbas_nom : номинальная ковариация измерений GBAS (3x3)
        """
        self.dt = dt
        self.Q = Q
        self.R_ils_nom = R_ils_nom
        self.R_gbas_nom = R_gbas_nom

        # Размерность вектора состояния x (позиция, скорость, ускорение и смещения датчиков)
        self.n = 12

        # Вектор состояния; на этапе инициализации заполняется нулями и далее
        # будет «подтягиваться» к реальному положению по мере поступления измерений.
        self.x = np.zeros(self.n)

        # Ковариационная матрица ошибки оценки.
        # Большие начальные значения отражают высокую неопределённость до появления измерений.
        self.P = np.eye(self.n) * 1000.0

        # Матрица перехода состояния F соответствует модели
        # «постоянная скорость + ускорение как случайный процесс»:
        #   позиция интегрирует скорость, скорость интегрирует ускорение.
        self.F = np.eye(self.n)
        for i in range(3):
            self.F[i, i + 3] = dt      # позиция от скорости
            self.F[i + 3, i + 6] = dt  # скорость от ускорения

        # ---------- Матрица наблюдения для ILS ----------
        # Измерения ILS: y_ils = y + b_ils_y,   z_ils = z + b_ils_z
        self.H_ils = np.zeros((2, self.n))
        self.H_ils[0, 1] = 1   # y
        self.H_ils[0, 9] = 1   # b_ils_y
        self.H_ils[1, 2] = 1   # z
        self.H_ils[1, 10] = 1  # b_ils_z

        # ---------- Матрица наблюдения для GBAS ----------
        # Измерения GBAS: x_gbas = x,   y_gbas = y,   z_gbas = z + b_gbas_z
        self.H_gbas = np.zeros((3, self.n))
        self.H_gbas[0, 0] = 1   # x
        self.H_gbas[1, 1] = 1   # y
        self.H_gbas[2, 2] = 1   # z
        self.H_gbas[2, 11] = 1  # b_gbas_z

    # ----------------------------------------------------------------------
    #                              Предсказание
    # ----------------------------------------------------------------------
    def predict(self):
        """Шаг предсказания по модели динамики."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()   # возвращаем предсказанную позицию (x,y,z)

    # ----------------------------------------------------------------------
    #                     Обновление по измерениям ILS
    # ----------------------------------------------------------------------
    def update_ils(self, y_meas, z_meas, sqi_ils):
        """
        Обновление состояния по измерениям ILS с адаптивным весом.

        Параметры:
            y_meas   : измеренное боковое отклонение (м)
            z_meas   : измеренная высота от ILS (м)
            sqi_ils  : словарь с индикатором качества сигнала ILS,
                       должен содержать ключ 'signal_level'
        """
        # 1. Адаптивная настройка ковариации измерений
        alpha = self._compute_alpha_ils(sqi_ils)
        R = self.R_ils_nom * alpha

        # 2. Инновация (невязка) и её ковариация
        z_pred = self.H_ils @ self.x               # предсказанное измерение
        innov = np.array([y_meas, z_meas]) - z_pred
        S = self.H_ils @ self.P @ self.H_ils.T + R

        # 3. Проверка на выбросы (нормированная инновация > 3σ)
        norm_innov = np.abs(innov) / np.sqrt(np.diag(S))
        if np.any(norm_innov > 3.0):
            # При выбросе дополнительно увеличиваем ковариацию (уменьшаем доверие)
            R = R * 10.0
            S = self.H_ils @ self.P @ self.H_ils.T + R

        # 4. Вычисление коэффициента Калмана
        K = self.P @ self.H_ils.T @ np.linalg.inv(S)

        # 5. Обновление состояния и ковариации
        self.x = self.x + K @ innov
        self.P = (np.eye(self.n) - K @ self.H_ils) @ self.P

    # ----------------------------------------------------------------------
    #                     Обновление по измерениям GBAS
    # ----------------------------------------------------------------------
    def update_gbas(self, x_meas, y_meas, z_meas, sqi_gbas):
        """
        Обновление состояния по измерениям GBAS с адаптивным весом.

        Параметры:
            x_meas, y_meas, z_meas : измеренные координаты (м)
            sqi_gbas : словарь с индикаторами качества GBAS,
                       должен содержать ключи 'vpl' и 'val'
        """
        # 1. Адаптивная настройка ковариации
        alpha = self._compute_alpha_gbas(sqi_gbas)
        R = self.R_gbas_nom * alpha

        # 2. Инновация и её ковариация
        z_pred = self.H_gbas @ self.x
        innov = np.array([x_meas, y_meas, z_meas]) - z_pred
        S = self.H_gbas @ self.P @ self.H_gbas.T + R

        # 3. Проверка на выбросы
        norm_innov = np.abs(innov) / np.sqrt(np.diag(S))
        if np.any(norm_innov > 3.0):
            R = R * 10.0
            S = self.H_gbas @ self.P @ self.H_gbas.T + R

        # 4. Коэффициент Калмана
        K = self.P @ self.H_gbas.T @ np.linalg.inv(S)

        # 5. Обновление
        self.x = self.x + K @ innov
        self.P = (np.eye(self.n) - K @ self.H_gbas) @ self.P

    # ----------------------------------------------------------------------
    #              Вспомогательные методы для адаптации весов
    # ----------------------------------------------------------------------
    def _compute_alpha_ils(self, sqi):
        """
        Коэффициент адаптации для ILS на основе уровня сигнала.
        alpha = (ref_level / signal_level)^2.
        """
        if 'signal_level' in sqi and sqi['signal_level'] > 0:
            ref = sqi.get('ref_level', 100.0)
            return (ref / sqi['signal_level']) ** 2
        return 1.0

    def _compute_alpha_gbas(self, sqi):
        """
        Коэффициент адаптации для GBAS на основе отношения VPL/VAL.
        alpha = max(1, (VPL/VAL)^2).
        """
        if 'vpl' in sqi and 'val' in sqi and sqi['val'] > 0:
            ratio = sqi['vpl'] / sqi['val']
            return max(1.0, ratio ** 2)
        return 1.0

    # ----------------------------------------------------------------------
    #                       Получение текущей позиции
    # ----------------------------------------------------------------------
    def get_position(self):
        """Вернуть текущую оценку позиции (x, y, z)."""
        return self.x[:3].copy()

    # ----------------------------------------------------------------------
    #                    Методы контроля целостности
    # ----------------------------------------------------------------------
    def compute_protection_levels(self):
        """
        Вычисление вертикального (VPL) и горизонтального (HPL) уровней защиты.
        Используется коэффициент 5.33 для вероятности пропуска отказа 10^-7.
        """
        P_pos = self.P[:3, :3]                # подматрица для позиции
        HPL = 5.33 * np.sqrt(P_pos[0, 0] + P_pos[1, 1])   # горизонтальный
        VPL = 5.33 * np.sqrt(P_pos[2, 2])                  # вертикальный
        return HPL, VPL

    def check_availability(self, cat='CAT3'):
        """
        Проверка доступности для заданной категории посадки.
        Возвращает True, если уровни защиты не превышают пороги ICAO.
        """
        HPL, VPL = self.compute_protection_levels()
        if cat == 'CAT1':
            return HPL <= HAL_CAT1 and VPL <= VAL_CAT1
        elif cat == 'CAT2':
            return HPL <= HAL_CAT2 and VPL <= VAL_CAT2
        else:   # CAT3
            return HPL <= HAL_CAT3 and VPL <= VAL_CAT3

    def detect_faults(self, innovations, S_diag, threshold=3.0):
        """
        Обнаружение аномальных измерений по нормированной инновации.
        Возвращает булев массив, где True – аномалия.

        Параметры:
            innovations : массив инноваций (размерность m)
            S_diag      : диагональ ковариационной матрицы инноваций (размерность m)
            threshold   : порог (по умолчанию 3.0)
        """
        norm_innov = np.abs(innovations) / np.sqrt(S_diag)
        return norm_innov > threshold

    # ----------------------------------------------------------------------
    #   Метод итеративного исключения аномальных измерений (greedy exclusion)
    #   ВНИМАНИЕ: в текущей симуляции не используется, но оставлен как пример.
    # ----------------------------------------------------------------------
    def greedy_exclusion(self, z_available, H_available, R_available, x_pred, P_pred, max_iter=5):
        """
        Исключение аномальных измерений методом последовательного отбрасывания
        наихудшего (greedy). Полезно при работе с множеством спутниковых измерений,
        но в данной модели GBAS выдаёт готовое решение, поэтому метод не вызывается.

        Параметры:
            z_available : список векторов измерений (каждый np.array)
            H_available : список соответствующих матриц наблюдения
            R_available : список ковариационных матриц измерений
            x_pred      : предсказанное состояние (вектор)
            P_pred      : предсказанная ковариация
            max_iter    : максимальное число итераций

        Возвращает:
            remaining_indices : индексы оставшихся (чистых) измерений
            excluded_indices  : индексы исключённых измерений
        """
        remaining = list(range(len(z_available)))
        excluded = []
        x_current = x_pred.copy()
        P_current = P_pred.copy()

        for _ in range(max_iter):
            if len(remaining) < 4:   # минимальное количество для работы фильтра
                break

            innovations = []
            S_diags = []

            for idx in remaining:
                H = H_available[idx]
                R = R_available[idx]
                z_pred = H @ x_current
                innov = z_available[idx] - z_pred
                S = H @ P_current @ H.T + R
                innovations.append(innov)
                S_diags.append(np.diag(S))

            # Вычисляем нормированную инновацию как максимум по компонентам
            norm_innov = []
            for innov, S_diag in zip(innovations, S_diags):
                # Защита от деления на ноль (добавляем эпсилон)
                with np.errstate(divide='ignore', invalid='ignore'):
                    norm = np.abs(innov) / np.sqrt(np.maximum(S_diag, 1e-12))
                norm_innov.append(np.max(norm))   # максимум по компонентам

            faults = [n > 3.0 for n in norm_innov]   # порог 3σ

            if not any(faults):
                break

            # Исключаем измерение с максимальной нормированной инновацией
            worst_idx = np.argmax(norm_innov)
            excluded.append(remaining.pop(worst_idx))

            # При желании здесь можно обновить x_current и P_current,
            # выполнив коррекцию по оставшимся измерениям, но для простоты опустим.

        return remaining, excluded