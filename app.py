import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from simulation import run_simulation

def simulate(ils_noise_deg, gbas_noise_m, failure_time_gbas, multipath_time,
             ils_failure_time, jamming_time, spoofing_time):
    ils_noise_rad = np.radians(ils_noise_deg)

    # Запускаем численную модель гибридной системы посадки.
    # На выходе помимо ошибок возвращается история режимов работы (mode_history),
    # которую ниже визуализируем на отдельном подграфике.
    results = run_simulation(
        ils_noise=ils_noise_rad,
        gbas_noise=gbas_noise_m,
        failure_time_gbas=failure_time_gbas,
        multipath_time=multipath_time if multipath_time < 120 else None,
        ils_failure_time=ils_failure_time if ils_failure_time < 120 else None,
        jamming_time=jamming_time if jamming_time < 120 else None,
        spoofing_time=spoofing_time if spoofing_time < 120 else None
    )

    time = results['time']

    # Создаём два подграфика:
    #  1) ошибки по положению;
    #  2) дискретный график, показывающий какой режим был активен во времени.
    fig, (ax_err, ax_mode) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]}
    )

    # --- График ошибок ---
    ax_err.plot(time, results['error_hybrid'], label='Гибридная система (EKF)', linewidth=2, color='blue')
    ax_err.plot(time, results['error_ils'], label='ILS only', linestyle='--', alpha=0.7, color='red')
    ax_err.plot(time, results['error_gbas'], label='GBAS only', linestyle='--', alpha=0.7, color='green')
    ax_err.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_err.set_ylabel('Ошибка по положению (м)')
    ax_err.set_title('Ошибки гибридной системы и отдельных подсистем')
    ax_err.legend()
    ax_err.grid(True)

    # --- График режимов работы гибридной системы ---
    mode_history = results.get('mode_history', [])

    if mode_history:
        # Кодируем строковые режимы в целые числа, чтобы можно было построить ступенчатый график.
        mode_order = ['IRS_ONLY', 'ILS_ONLY', 'GBAS_ONLY', 'HYBRID']
        mode_to_int = {name: idx for idx, name in enumerate(mode_order)}
        mode_numeric = np.array([mode_to_int[m] for m in mode_history])

        # Ступенчатый график (режим меняется по времени дискретно).
        ax_mode.step(time, mode_numeric, where='post', color='black')
        ax_mode.set_yticks(range(len(mode_order)))
        ax_mode.set_yticklabels([
            'Только IRS',
            'Только ILS',
            'Только GBAS',
            'Гибридный режим'
        ])
        ax_mode.set_xlabel('Время (с)')
        ax_mode.set_ylabel('Режим')
        ax_mode.grid(True, axis='x', linestyle='--', alpha=0.5)
    else:
        # На случай, если симуляция будет вызываться без mode_history.
        ax_mode.text(
            0.5, 0.5, 'История режимов недоступна',
            ha='center', va='center', transform=ax_mode.transAxes
        )
        ax_mode.set_axis_off()

    return fig

demo = gr.Interface(
    fn=simulate,
    inputs=[
        gr.Slider(0.0, 0.1, value=0.01, step=0.001, label="Уровень шума ILS (градусы)"),
        gr.Slider(0.0, 2.0, value=0.5, step=0.1, label="Уровень шума GBAS (м)"),
        gr.Slider(0.0, 120.0, value=100.0, step=1.0, label="Время отказа GBAS (с)"),
        gr.Slider(0.0, 120.0, value=120.0, step=1.0, label="Время начала мультипути ILS (с)"),
        gr.Slider(0.0, 120.0, value=120.0, step=1.0, label="Время отказа ILS (с)"),
        gr.Slider(0.0, 120.0, value=120.0, step=1.0, label="Время начала глушения (с)"),
        gr.Slider(0.0, 120.0, value=120.0, step=1.0, label="Время начала спуфинга (с)")
    ],
    outputs=gr.Plot(label="График ошибки"),
    title="Моделирование гибридной системы посадки",
    description="Дипломный проект: настройте параметры — график обновляется автоматически",
    live=True
)

if __name__ == "__main__":
    demo.launch(share=True)