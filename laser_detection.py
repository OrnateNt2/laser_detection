import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def select_algorithm_menu():
    """
    Меню выбора метода ОБНАРУЖЕНИЯ координат лазера:
      1) По зелёному цвету (HSV)
      2) По максимальной яркости
      3) Комбинированный
      4) Выход
    """
    print("Выберите метод обнаружения лазера:")
    print("  1) По цвету (зелёный)")
    print("  2) По яркости (maxLoc)")
    print("  3) Комбинированный (цвет + яркость)")
    print("  4) Выход")

    while True:
        choice = input("Введите номер (1/2/3/4): ")
        if choice in ["1", "2", "3", "4"]:
            return choice
        else:
            print("Неверный ввод. Повторите.")


def select_measure_menu():
    """
    Меню выбора метода ФОРМИРОВАНИЯ сигнала (временного ряда), подаваемого в FFT:
      1) Средняя яркость (ROI)
      2) Признак наличия зелёного света (0/1)
      3) Разница яркости (с предыдущим кадром)
    """
    print("Выберите метод формирования значения (сигнала) для FFT:")
    print("  1) Средняя яркость в ROI (по координате лазера)")
    print("  2) Признак зелёного света (0 или 1)")
    print("  3) Разница яркости с предыдущим кадром")

    while True:
        choice = input("Введите номер (1/2/3): ")
        if choice in ["1", "2", "3"]:
            return choice
        else:
            print("Неверный ввод. Повторите.")


def find_laser_by_color(frame, lower_hsv, upper_hsv):
    """
    Ищем лазерную точку по HSV-диапазону (зелёный).
    Возвращаем (x, y) или None, если не нашли.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return (cx, cy)


def find_laser_by_brightness(frame_gray):
    """
    Ищем лазерную точку по максимальной яркости.
    Возвращаем (x, y) или None, если точка слишком тёмная.
    """
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(frame_gray)
    if maxVal < 50:  # Порог
        return None
    return (maxLoc[0], maxLoc[1])  # x,y


def combine_coordinates(coord_color, coord_bright, frame_size, alpha=0.5):
    """
    Комбинируем координаты (x,y), если найдены и по цвету, и по яркости.
    Если одна из них None, берём другую.
    alpha - вес для coord_color.
    """
    if coord_color is None and coord_bright is None:
        return None
    elif coord_color is None:
        return coord_bright
    elif coord_bright is None:
        return coord_color
    else:
        x = int(alpha * coord_color[0] + (1 - alpha) * coord_bright[0])
        y = int(alpha * coord_color[1] + (1 - alpha) * coord_bright[1])
        w, h = frame_size
        x = np.clip(x, 0, w-1)
        y = np.clip(y, 0, h-1)
        return (x, y)


def compute_average_brightness_around(frame_gray, center, roi_size=20):
    """
    Считаем среднюю яркость в окрестности (roi_size x roi_size) вокруг center=(x, y).
    """
    h, w = frame_gray.shape
    half = roi_size // 2

    x1 = max(center[0] - half, 0)
    x2 = min(center[0] + half, w-1)
    y1 = max(center[1] - half, 0)
    y2 = min(center[1] + half, h-1)

    roi = frame_gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0
    return np.mean(roi)


def realtime_laser_analysis(video_source=0, detect_method="1", measure_method="1", buffer_size=120):
    """
    Запуск анализа в реальном времени (или с видео-файла).
    detect_method: "1"/"2"/"3" (определение координат: по цвету/по яркости/комбинированно)
    measure_method: "1"/"2"/"3" (средняя яркость / признак зелёного света / разница яркости)
    buffer_size: сколько кадров храним для FFT (например, 120 кадров ~ 2 секунды при 60 FPS).
    """
    # HSV-диапазон для зелёного (можно подстроить по ситуации):
    lower_green = np.array([45, 80, 50], dtype=np.uint8)
    upper_green = np.array([75, 255, 255], dtype=np.uint8)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Не удалось открыть источник: {video_source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1.0:
        fps = 30.0  # если не удалось прочитать реальный FPS

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if total_frames < 1:
        total_frames = 0  # для веб-камеры

    print(f"Частота кадров (предположительно): {fps:.2f} FPS")

    # Настройки "живого" графика
    plt.ion()
    fig, (ax_time, ax_freq) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Реальный анализ сигнала лазера")

    brightness_buffer = []
    frame_count = 0
    prev_value = 0  # для вычисления разницы (measure_method=3)

    line_time, = ax_time.plot([], [], 'g-')
    line_freq, = ax_freq.plot([], [], 'b-')

    ax_time.set_title("Сигнал во времени")
    ax_time.set_xlabel("Номер кадра (в буфере)")
    ax_time.set_ylabel("Значение сигнала")

    ax_freq.set_title("Спектр (FFT)")
    ax_freq.set_xlabel("Частота, Гц")
    ax_freq.set_ylabel("Амплитуда")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Конец видео или ошибка чтения.")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame_gray.shape

        # 1) Определяем координаты лазера (coord_final)
        coord_color = None
        coord_bright = None

        if detect_method == "1":
            coord_color = find_laser_by_color(frame, lower_green, upper_green)
            coord_final = coord_color
        elif detect_method == "2":
            coord_bright = find_laser_by_brightness(frame_gray)
            coord_final = coord_bright
        else:
            # "3" - комбинированный
            coord_color = find_laser_by_color(frame, lower_green, upper_green)
            coord_bright = find_laser_by_brightness(frame_gray)
            coord_final = combine_coordinates(coord_color, coord_bright, (w, h), alpha=0.5)

        # 2) Считаем сигнал (value), который добавим в буфер
        if measure_method == "1":
            # Средняя яркость
            if coord_final is not None:
                value = compute_average_brightness_around(frame_gray, coord_final, roi_size=20)
            else:
                value = 0

        elif measure_method == "2":
            # Признак зелёного света (0 или 1)
            # Логично ориентироваться на find_laser_by_color, 
            # но если user выбрал detect_method="2" (по яркости), 
            #   тогда мы можем либо искать цвет "дополнительно", 
            #   либо дать 1, если любым способом найдено пятно.
            # Для упрощения — отдельно опять ищем по цвету:
            coord_green = find_laser_by_color(frame, lower_green, upper_green)
            if coord_green is not None:
                value = 1
            else:
                value = 0

        else:
            # "3" — Разница яркости с предыдущим кадром
            # Для этого нужно сначала получить текущую яркость (как в методе 1)
            if coord_final is not None:
                current_brightness = compute_average_brightness_around(frame_gray, coord_final, roi_size=20)
            else:
                current_brightness = 0
            value = current_brightness - prev_value
            prev_value = current_brightness  # запоминаем для следующего шага

        # Нарисуем кружок в кадре (только если координата найдена)
        if coord_final is not None:
            cv2.circle(frame, (coord_final[0], coord_final[1]), 10, (0, 0, 255), 2)

        # Добавляем значение в буфер
        brightness_buffer.append(value)
        if len(brightness_buffer) > buffer_size:
            brightness_buffer.pop(0)

        # FFT-анализ (если данных достаточно, хотя бы 2)
        data = np.array(brightness_buffer, dtype=float)
        N = len(data)
        if N > 1:
            # Удаляем DC (среднее) 
            data_centered = data - np.mean(data)

            fft_vals = np.fft.fft(data_centered)
            fft_freq = np.fft.fftfreq(N, d=1.0/fps)
            pos_mask = fft_freq >= 0
            fft_vals_pos = np.abs(fft_vals[pos_mask])
            fft_freq_pos = fft_freq[pos_mask]

            max_idx = np.argmax(fft_vals_pos)
            dominant_freq = fft_freq_pos[max_idx]

            # Обновляем графики
            line_time.set_xdata(range(N))
            line_time.set_ydata(data_centered)  # показываем уже "без DC"
            ax_time.set_xlim(0, N)
            minY, maxY = np.min(data_centered), np.max(data_centered)
            if minY == maxY:
                # если сигнал совсем плоский, сделаем небольшой диапазон
                ax_time.set_ylim(minY-1, maxY+1)
            else:
                ax_time.set_ylim(minY*1.2, maxY*1.2)

            line_freq.set_xdata(fft_freq_pos)
            line_freq.set_ydata(fft_vals_pos)
            ax_freq.set_xlim(0, 100)  # смотрим до 100 Гц
            ax_freq.set_ylim(0, max(1, np.max(fft_vals_pos) * 1.2))

            ax_freq.set_title(f"Спектр (пик ~ {dominant_freq:.1f} Гц)")

            plt.pause(0.001)

        # Вывод номера кадра
        frame_count += 1
        if total_frames > 0:
            text_frame = f"Frame: {frame_count}/{int(total_frames)}"
        else:
            text_frame = f"Frame: {frame_count}"
        cv2.putText(frame, text_frame, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        # Уменьшаем размер для вывода, чтобы окно было поменьше
        scale = 0.5
        frame_resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imshow("Laser detection", frame_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()
    print("Завершено.")


def main():
    print("=== Демонстрационный скрипт для анализа мерцания лазера ===")

    # Меню 1: метод обнаружения
    detect_method = select_algorithm_menu()
    if detect_method == "4":
        print("Выход по запросу пользователя.")
        sys.exit(0)

    # Меню 2: метод формирования сигнала
    measure_method = select_measure_menu()

    # Источник видео:
    print("Выберите источник видео:")
    print("  1) Веб-камера (индекс 0)")
    print("  2) Путь к файлу (например, 'laser.mp4')")
    choice_source = input("Введите 1 или 2: ").strip()

    if choice_source == "1":
        source = 0  # веб-камера
    else:
        source = input("Укажите путь к видео-файлу: ").strip()

    # Размер буфера (кол-во кадров для FFT)
    buffer_input = input("Введите размер буфера для FFT (число кадров), по умолчанию 120: ").strip()
    if buffer_input == "":
        buffer_size = 120
    else:
        try:
            buffer_size = int(buffer_input)
        except ValueError:
            buffer_size = 120
            print("Некорректный ввод, установлен по умолчанию 120.")

    realtime_laser_analysis(
        video_source=source,
        detect_method=detect_method,
        measure_method=measure_method,
        buffer_size=buffer_size
    )


if __name__ == "__main__":
    main()
