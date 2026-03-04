# models.py
import numpy as np
from config import *


class LocalFrame:
    """Локальная система координат, связанная с порогом ВПП"""

    def __init__(self, r_ltp, r_fpap, r_tcp):
        """
        Параметры:
            r_ltp: координаты LTP (ECEF)
            r_fpap: координаты FPAP (ECEF)
            r_tcp: координаты TCP (ECEF)
        """
        self.r_ltp = np.array(r_ltp)

        # Ось X (вдоль ВПП)
        vec_rw = np.array(r_fpap) - self.r_ltp
        self.u_rw = vec_rw / np.linalg.norm(vec_rw)

        # Ось Z (вертикаль)
        vec_vert = np.array(r_tcp) - self.r_ltp
        self.u_vert = vec_vert / np.linalg.norm(vec_vert)

        # Ось Y (боковое отклонение) - перпендикуляр к X и Z
        self.u_lat = np.cross(self.u_vert, self.u_rw)
        self.u_lat = self.u_lat / np.linalg.norm(self.u_lat)

        # Матрица поворота из ECEF в локальную систему
        self.R = np.vstack([self.u_rw, self.u_lat, self.u_vert])

    def ecef_to_local(self, r_ecef):
        """Преобразование из ECEF в локальные координаты (x, y, z)"""
        delta = np.array(r_ecef) - self.r_ltp
        return self.R @ delta


class ILSModel:
    """Модель измерений ILS"""

    def __init__(self, r_loc, r_gs, gpa, sigma_lat, sigma_v, multipath_magnitude):
        self.r_loc = np.array(r_loc)
        self.r_gs = np.array(r_gs)
        self.gpa = gpa
        self.sigma_lat = sigma_lat
        self.sigma_v = sigma_v
        self.multipath_magnitude = multipath_magnitude
        self.multipath_time = None  # время начала мультипути
        self.failure_time = None  # время отказа ILS

    def set_multipath(self, time_start):
        self.multipath_time = time_start

    def set_failure(self, time_start):
        self.failure_time = time_start

    def compute_true_deviations(self, r_ac, local_frame):
        """Вычисление истинных угловых отклонений"""
        # Боковое отклонение
        delta_loc = r_ac - self.r_loc
        d_lat = np.dot(local_frame.u_lat, delta_loc)
        dist_along = np.dot(local_frame.u_rw, delta_loc)
        # избегаем деления на ноль
        if abs(dist_along) < 1e-6:
            alpha_lat = 0.0
        else:
            alpha_lat = np.arctan2(d_lat, abs(dist_along))

        # Вертикальное отклонение
        delta_gs = r_ac - self.r_gs
        proj_horiz = np.sqrt(
            np.dot(local_frame.u_lat, delta_gs) ** 2 +
            np.dot(local_frame.u_rw, delta_gs) ** 2
        )
        proj_vert = np.dot(local_frame.u_vert, delta_gs)
        if proj_horiz < 1e-6:
            alpha_v = 0.0
        else:
            alpha_v = np.arctan2(proj_vert, proj_horiz) - self.gpa

        return alpha_lat, alpha_v

    def get_measurement(self, r_ac, local_frame, t):
        """
        Получение измерения ILS с учётом шума и мультипути.

        Возвращает словарь с угловыми и линейными отклонениями или None
        в случае полного отказа ILS.
        """
        # При наступлении отказа считаем, что измерения от ILS полностью отсутствуют
        if self.failure_time is not None and t >= self.failure_time:
            return None

        alpha_lat_true, alpha_v_true = self.compute_true_deviations(r_ac, local_frame)

        # Добавление шума в измеряемые углы
        alpha_lat_meas = alpha_lat_true + np.random.normal(0, self.sigma_lat)
        alpha_v_meas = alpha_v_true + np.random.normal(0, self.sigma_v)

        # Добавление мультипути (ступенчатая добавка к боковому отклонению)
        if self.multipath_time is not None and t >= self.multipath_time:
            alpha_lat_meas += self.multipath_magnitude

        # Преобразование в линейные отклонения в локальной системе (для анализа в метрах)
        dist_to_loc = np.linalg.norm(r_ac - self.r_loc)
        y_ils = np.sin(alpha_lat_meas) * dist_to_loc

        dist_to_gs = np.linalg.norm(r_ac - self.r_gs)
        z_ils = np.sin(alpha_v_meas) * dist_to_gs

        return {
            'alpha_lat': alpha_lat_meas,
            'alpha_v': alpha_v_meas,
            'y': y_ils,
            'z': z_ils
        }
    def get_sqi(self, t):
        """Возвращает индикаторы качества для ILS"""
        sqi = {
            'signal_level': 100.0,  # номинал
            'ref_level': 100.0
        }
        # Если мультипуть активен, сигнал считается искажённым — уменьшаем уровень
        if self.multipath_time and t >= self.multipath_time:
            sqi['signal_level'] = 30.0  # резкое падение
        if self.failure_time and t >= self.failure_time:
            sqi['signal_level'] = 0.0  # сигнал отсутствует
        return sqi


class GBASModel:
    """Модель измерений GBAS"""

    def __init__(self, sigma, iono_gradient):
        self.sigma = sigma
        self.iono_gradient = iono_gradient  # м/км
        self.failure_time = None  # время отказа (потеря сигнала)
        self.jamming_time = None  # время начала глушения
        self.spoofing_time = None  # время начала спуфинга
        self.spoofing_offset = np.array([100.0, 100.0, 50.0])  # смещение при спуфинге

    def set_failure(self, time_start):
        self.failure_time = time_start

    def set_jamming(self, time_start):
        self.jamming_time = time_start

    def set_spoofing(self, time_start):
        self.spoofing_time = time_start

    def get_measurement(self, r_ac_true, local_frame, t, ref_station_pos):

        """
        Возвращает измеренные координаты в локальной системе с ошибками.
        ref_station_pos - позиция наземной станции GBAS (ECEF) для моделирования ионосферного градиента.
        """
        # Истинные координаты в локальной системе
        true_local = local_frame.ecef_to_local(r_ac_true)
        x_true, y_true, z_true = true_local

        # Если отказ - возвращаем None (сигнал пропал)
        if self.failure_time and t >= self.failure_time:
            return None

        # Моделирование ошибок
        # 1. Шум измерения (гауссовский)
        noise = np.random.normal(0, self.sigma, 3)

        # 2. Ионосферная ошибка, пропорциональная расстоянию до станции
        dist_to_station = np.linalg.norm(r_ac_true - np.array(ref_station_pos)) / 1000.0  # в км
        iono_error = self.iono_gradient * dist_to_station  # линейная модель

        # Ионосферная ошибка влияет в основном на высоту и горизонтальное положение.
        # В упрощённой модели учитываем её только в вертикальной компоненте.
        error = noise.copy()
        error[2] += iono_error  # добавим к высоте

        # Базовое измерение GBAS в локальной системе координат
        measured_local = true_local + error

        # При спуфинге навигационный приёмник «видит» смещённое положение.
        # Используем заранее заданный вектор смещения в локальной системе.
        if self.spoofing_time is not None and t >= self.spoofing_time:
            measured_local = measured_local + self.spoofing_offset

        return {
            'x': measured_local[0],
            'y': measured_local[1],
            'z': measured_local[2],
            'true': true_local
        }

    def get_sqi(self, t):
        """Возвращает индикаторы качества для GBAS"""
        sqi = {
            'vpl': 5.0,    # номинальный вертикальный уровень защиты (м)
            'val': 10.0    # предельный уровень (м)
        }
        # Если отказ, то VPL становится очень большим
        if self.failure_time and t >= self.failure_time:
            sqi['vpl'] = 100.0  # сигнал потерян
        if self.jamming_time and t >= self.jamming_time:
            sqi['vpl'] = 200.0  # глушение – резкое ухудшение
        if self.spoofing_time and t >= self.spoofing_time:
            sqi['vpl'] = 50.0  # при спуфинге VPL может быть умеренным
        # Можно также моделировать ионосферные возмущения увеличением VPL
        return sqi

class IRSModel:
    """Упрощённая модель инерциальной системы (для прогнозирования)"""
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros(3)   # позиция
        self.v = np.zeros(3)   # скорость
        self.bias_acc = np.zeros(3)   # смещения акселерометров (для простоты 0)
        self.drift_rate = 0.01        # скорость дрейфа (м/с²)

    def set_initial(self, pos, vel):
        self.x = pos.copy()
        self.v = vel.copy()

    def predict(self, dt=None):
        if dt is None:
            dt = self.dt
        # Моделируем движение с постоянной скоростью + дрейф
        self.v = self.v - self.bias_acc * dt   # учёт смещения
        self.x += self.v * dt
        return self.x.copy()

    def get_position(self):
        return self.x.copy()