import cv2
import numpy as np

backSub_mog = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

#cap = cv2.VideoCapture("web_2.mp4") # Пример с видео
cap = cv2.VideoCapture(0)   # Пример с вэбкамеры (0)

min_count = 100
max_count = 0
counts = []

  # Инициализация переменнызх для вывода результата в видеофайл  
    # Подколючает видеокодек
fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Имя и параметры файла для вывода результата программы
out = cv2.VideoWriter('web_3_result.avi',fourcc, 20.0, (640, 480))

# Запускаем бесконечный цикл
while True:
    success, frame = cap.read()

    # Выход из цикла когда нет кадров (закончилось видео)
    if not success:
        break

    # Устанавливаем размер окна
    frame = cv2.resize(frame, (640, 480))

    # Вычитание фона
    imgNoBg = backSub_mog.apply(frame)

    # Устанавливаем глобальный порог для удаления теней
    # Морфология   
    kernel = np.ones((2, 2), np.uint8)
    mask_thresh = cv2.threshold(imgNoBg, 100, 255, cv2.THRESH_BINARY)   # 2й аргумент от 0-255 изменяет определение теней на кадре
    imgNoBg = cv2.morphologyEx(imgNoBg, cv2.MORPH_OPEN, kernel, iterations=2)
    imgNoBg = cv2.dilate(imgNoBg, kernel, iterations=2)
    
     # Поиск контуров
    contours, hierarchy = cv2.findContours(imgNoBg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Минимальная площать контура
    min_contour_area = 20000  # Можно изменять, 0 - 307200(ширина окна * высота окна) чем больше значение, тем больше маленьких объектов отсечется, для видео из примера  надо 800
    # Если человек близко к камере то надо увеличивать, если далеко - уменьшать

    # отсекаем объекты площадь которых меньше заданного значения
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    frame_out = frame.copy()

    # Проверка максимального количества людей в кадре
    if len(large_contours) > max_count:
        max_count = len(large_contours)

    # Проверка минимального количества людей в кадре
    if len(large_contours) < min_count and len(large_contours) > 0:
        min_count = len(large_contours)

    # Добавление количества людей в список для определения среднего количества
    if len(large_contours) > 0:
        counts.append(len(large_contours))

    # Выбеляем контур прямоугольником
    for cnt in large_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        frame_out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)
    
    # Выводим информацию на кадр о количестве контуров (человек)
    cv2.putText(frame, f"Peoples in frame: {len(large_contours)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Выводим информацию на кадр о минимальном количестве контуров (человек) в кадре
    cv2.putText(frame, f"Mimimal count in frame: {min_count}", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Выводим информацию на кадр о максимальном количестве контуров (человек) в кадре
    cv2.putText(frame, f"Maximal count in frame: {max_count}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Исключение деления на 0
    if len(counts) > 0:
        # Выводим информацию на кадр о среднем количестве контуров (человек) в кажре
        cv2.putText(frame, f"Average count in frame: {round(sum(counts) / len(counts), 2)}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Отображаем результат
    cv2.imshow('Frame_final', frame_out)

    #Сохранение результата в видео
    out.write(frame_out)
    
    cv2.imshow('Mask', imgNoBg)

    # Задержка вывода кадров для видео и ожидание кнопки q для выхода
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Закрываем видео
cap.release()

out.release()
# Закрываем все окна
cv2.destroyAllWindows()