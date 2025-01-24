import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

def select_algorithm_menu():
    """
    Простое меню выбора алгоритма в терминале.
    """
    print("Выберите метод обнаружения лазерной точки:")
    print("  1) Поиск по цвету (зелёный) [threshold по HSV]")
    print("  2) Поиск максимальной яркости (minMaxLoc)")
    print("  3) Комбинированный (цвет + яркость)")
    print("  4) Выход")

    while True:
        choice = input("Введите номер (1/2/3/4): ")
        if choice in ["1", "2", "3", "4"]:
            return choice
        else:
            print("Неверный ввод. Повторите.")


def find_laser_by_color(frame, lower_hsv, upper_hsv):
    """
    Ищем лазерную точку по HSV-диапазону (зелёный цвет).
    Возвращаем: (x, y) в пикселях или None, если не найдено.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Морфологические операции для уменьшения шума
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    # Находим самый большой контур
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
    Возвращаем (x, y) или None, если не найдено.
    """
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(frame_gray)
    # По желанию можно ввести порог:
    if maxVal < 50:
        return None
    return (maxLoc[0], maxLoc[1])


def combine_coordinates(coord_color, coord_bright, frame_size, alpha=0.5):
    """
    Комбинированный способ: если обе координаты найдены — усредняем.
    Если одна None, используем другую.
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
        width, height = frame_size
        x = np.clip(x, 0, width-1)
        y = np.clip(y, 0, height-1)
        return (x, y)


def compute_average_brightness_around(frame_gray, center, roi_size=20):
    """
    Считаем среднюю яркость в окрестности (roi_size x roi_size) вокруг center (x, y).
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


def realtime_laser_analysis(video_source=0, method="1", buffer_size=120):
    """
    Запускаем реальное воспроизведение видео (с веб-камеры или файла).
    method - строка "1", "2", или "3" (выбор алгоритма).
    buffer_size - кол-во кадров для FFT (примерно на 1-2 секунды при 60 FPS).
    """
    # Порог для зелёного цвета (HSV)
    lower_green = np.array([45, 80, 50], dtype=np.uint8)
    upper_green = np.array([75, 255, 255], dtype=np.uint8)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Не удалось открыть источник: {video_source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1.0:
        fps = 30.0
    print(f"Частота кадров (предположительно): {fps:.2f} FPS")

    # Если это файл, попытаемся определить общее количество кадров
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if total_frames < 1:
        total_frames = 0  # для веб-камеры будет 0 или -1

    # Готовим "живой" график
    plt.ion()
    fig, (ax_time, ax_freq) = plt.subplots(1, 2, figsize=(10, 4))
    #fig.suptitle("Анализ яркости лазера в реальном времени")

    brightness_buffer = []
    time_buffer = []
    frame_count = 0

    line_time, = ax_time.plot([], [], 'g-')
    line_freq, = ax_freq.plot([], [], 'b-')

    ax_time.set_title("Яркость во времени")
    ax_time.set_xlabel("Номер кадра (в буфере)")
    ax_time.set_ylabel("Яркость (усл. ед.)")

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

        # Методы обнаружения
        coord_color = None
        coord_bright = None

        if method == "1":
            coord_color = find_laser_by_color(frame, lower_green, upper_green)
            coord_final = coord_color
        elif method == "2":
            coord_bright = find_laser_by_brightness(frame_gray)
            coord_final = coord_bright
        else:
            # Комбинированный
            coord_color = find_laser_by_color(frame, lower_green, upper_green)
            coord_bright = find_laser_by_brightness(frame_gray)
            coord_final = combine_coordinates(coord_color, coord_bright, (w, h), alpha=0.5)

        # Рисуем найденную точку (если есть)
        if coord_final is not None:
            cv2.circle(frame, (coord_final[0], coord_final[1]), 10, (0, 0, 255), 2)
            avg_b = compute_average_brightness_around(frame_gray, coord_final, roi_size=20)
        else:
            avg_b = 0

        brightness_buffer.append(avg_b)
        time_buffer.append(frame_count)
        if len(brightness_buffer) > buffer_size:
            brightness_buffer.pop(0)
            time_buffer.pop(0)

        frame_count += 1

        # Обновляем FFT-анализ
        data = np.array(brightness_buffer)
        N = len(data)
        if N > 1:
            fft_vals = np.fft.fft(data)
            fft_freq = np.fft.fftfreq(N, d=1.0/fps)
            pos_mask = fft_freq >= 0
            fft_vals_pos = np.abs(fft_vals[pos_mask])
            fft_freq_pos = fft_freq[pos_mask]
            max_idx = np.argmax(fft_vals_pos)
            dominant_freq = fft_freq_pos[max_idx]

            # Обновление графиков
            # Временная область:
            line_time.set_xdata(range(N))
            line_time.set_ydata(data)
            ax_time.set_xlim(0, N)
            ax_time.set_ylim(0, max(1, np.max(data) * 1.2))

            # Спектральная область:
            line_freq.set_xdata(fft_freq_pos)
            line_freq.set_ydata(fft_vals_pos)
            ax_freq.set_xlim(0, 100)  # смотрим до 100 Гц
            ax_freq.set_ylim(0, max(1, np.max(fft_vals_pos) * 1.2))

            #ax_freq.set_title(f"Спектр (пик ~ {dominant_freq:.1f} Гц)")
            ax_freq.set_title(f"Спектр (пик ~ {dominant_freq:.1f} Гц)")

            plt.pause(0.001)

        # Подпишем кадр
        if total_frames > 0:
            text_frame = f"Frame: {frame_count}/{int(total_frames)}"
        else:
            text_frame = f"Frame: {frame_count}"

        cv2.putText(frame, 
                    text_frame, 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, 
                    (255, 255, 255), 
                    2)

        # Уменьшаем изображение для вывода
        scale = 0.5  # в 2 раза меньше
        frame_resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Показываем кадр
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
    print("Демонстрационный скрипт для анализа мерцания лазера на видео / веб-камере.")
    method = select_algorithm_menu()
    if method == "4":
        print("Выход по запросу пользователя.")
        sys.exit(0)

    print("Выберите источник видео:")
    print("  1) Веб-камера (устройство 0)")
    print("  2) Путь к файлу (например, 'laser.mp4')")
    choice_source = input("Введите 1 или 2: ").strip()

    if choice_source == "1":
        source = 0  # веб-камера
    else:
        source = input("Укажите путь к видеофайлу: ").strip()

    buffer_input = input("Введите размер буфера для FFT (число кадров), по умолчанию 120: ").strip()
    if buffer_input == "":
        buffer_size = 120
    else:
        try:
            buffer_size = int(buffer_input)
        except ValueError:
            buffer_size = 120
            print("Некорректный ввод, установлен по умолчанию 120.")

    realtime_laser_analysis(video_source=source, method=method, buffer_size=buffer_size)


if __name__ == "__main__":
    main()
