import cv2
import numpy as np

def detect_objects_yolo(image_path, config_path, weights_path, class_names_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Загрузка классов объектов
    with open(class_names_path, 'r') as f:
        class_names = f.read().strip().split('\n')

    # Загрузка модели YOLO
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # Получение имен всех слоев в сети YOLO
    layer_names = net.getLayerNames()
    print(layer_names)

    # output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Преобразование изображения в blob
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Передача blob в сеть YOLO
    net.setInput(blob)

    # в зависимости от версии CUDA надо писать либо i[0] - 1, либо i - 1
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers )

    class_ids = []
    confidences = []
    boxes = []

    # Перебор результатов обнаружения объектов
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Условие фильтрации по уверенности
            if confidence > 0.5:
                # Координаты центра и размеры прямоугольника
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Координаты верхнего левого угла
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Удаление повторяющихся рамок
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Нарисовать рамки и подписать объекты
    font = cv2.QT_FONT_NORMAL
    colors = np.random.uniform(0, 255, size=(len(class_names), 3))

    objects_count = {}
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(class_names[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y + 30), font, 0.5, color, 2)
            objects_count[label] = objects_count.get(label, 0) + 1

    # Вывод обработанного изображения
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return objects_count

config_path = 'yolov3.cfg'
weights_path = 'yolov3.weights'
class_names_path = 'coco.names'

detect_objects_yolo('cats_dogs.jpg', config_path, weights_path, class_names_path)
detect_objects_yolo('klass.jpg', config_path, weights_path, class_names_path)
detect_objects_yolo('Kafe.jpg', config_path, weights_path, class_names_path)
detect_objects_yolo('Kafe2.jpg', config_path, weights_path, class_names_path)
detect_objects_yolo('Ovcy.jpg', config_path, weights_path, class_names_path)
detect_objects_yolo('Ulitsa.jpg', config_path, weights_path, class_names_path)


def detect_objects_yolo_video(video_path, config_path, weights_path, class_names_path):
    # Загрузка классов объектов
    with open(class_names_path, 'r') as f:
        class_names = f.read().strip().split('\n')

    # Загрузка модели YOLO
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # Получение имен всех слоев в сети YOLO
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Открытие видеопотока
    cap = cv2.VideoCapture(video_path)

    # Создание словаря для подсчета количества объектов каждого класса
    frame_objects_count = {class_name: (0,0) for class_name in class_names}

    # Цикл обработки каждого кадра видео
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]

        # Преобразование изображения в blob
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        # Передача blob в сеть YOLO
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        # Перебор результатов обнаружения объектов
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Условие фильтрации по уверенности
                if confidence > 0.5:
                    # Координаты центра и размеры прямоугольника
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Координаты верхнего левого угла
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Удаление повторяющихся рамок
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Подсчет количества объектов каждого класса на текущем кадре
        objects_count = {}
        for i in range(len(boxes)):
            if i in indexes:
                label = class_names[class_ids[i]]
                objects_count[label] = objects_count.get(label, 0) + 1

        print(objects_count)
        for entry in objects_count:
            print(entry)
        # Сохранение количества объектов каждого класса на текущем кадре
            if (objects_count[entry] > frame_objects_count[entry][0]):
                frame_objects_count[entry] = (objects_count[entry], cap.get(cv2.CAP_PROP_POS_FRAMES))


    transformed_dict = {v[1]: (k, v[0]) for k, v in frame_objects_count.items()}


    # Первый проход по видео для добавления аннотаций к кадрам с наибольшим количеством объектов
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)

        if frame_index in transformed_dict:
            max_class_name, max_class_count = transformed_dict[frame_index]
            cv2.putText(frame, f"Max amount of {max_class_name}: {max_class_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Video with annotation", frame)
            cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов и закрытие окон
    cap.release()
    cv2.destroyAllWindows()

video_path = 'Ferma.mp4'

detect_objects_yolo_video(video_path, config_path, weights_path, class_names_path)
