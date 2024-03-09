import cv2

# Загрузка каскада Хаара для обнаружения лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Функция для обработки изображения и обнаружения лиц
def detect_faces(image):
    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Обнаружение лиц на изображении
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Отрисовка прямоугольников вокруг обнаруженных лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

def detect_faces_and_show(path):

    # Загрузка изображения
    image_path = path
    image = cv2.imread(image_path)

    # Обнаружение лиц и отображение результата
    faces_detected = detect_faces(image)
    cv2.imshow('Faces Detected', faces_detected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Функция для обработки видео и обнаружения лиц
def detect_faces_in_video(video_path):
    # Загрузка видео
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Преобразование кадра в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Обнаружение лиц на кадре
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # Отрисовка прямоугольников вокруг обнаруженных лиц
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Отображение кадра с лицами
        cv2.imshow('Faces Detected', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
# часть 1: только openCV

# detect_faces_and_show('IvanMladek.jpg')
#
# detect_faces_and_show('IvoPesak.jpg')
#
# detect_faces_in_video('JozinzBazin.mp4')

import face_recognition

# Функция для обнаружения лиц в видео с помощью Face Recognition
def detect_faces_with_facerecognition(video_path):
    # Загрузка видео
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Конвертация кадра в формат RGB (Face Recognition использует этот формат)
        rgb_frame = frame[:, :, ::-1]

        # Обнаружение лиц на кадре
        face_locations = face_recognition.face_locations(rgb_frame)

        # Отрисовка прямоугольников вокруг обнаруженных лиц
        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Отображение кадра с лицами
        cv2.imshow('Faces Detected', frame)

        # Выход из цикла при нажатии клавиши 'q'
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # Очистка ресурсов
    cap.release()
    cv2.destroyAllWindows()

# Функция для обнаружения лиц на изображении с помощью Face Recognition
def detect_faces_in_image_facerecognition(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Конвертация изображения в формат RGB (Face Recognition использует этот формат)
    rgb_image = image[:, :, ::-1]

    # Обнаружение лиц на изображении
    face_locations = face_recognition.face_locations(rgb_image)

    # Отрисовка прямоугольников вокруг обнаруженных лиц
    for top, right, bottom, left in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    # Отображение изображения с лицами
    cv2.imshow('Faces Detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# часть 2
detect_faces_in_image_facerecognition('IvoPesak.jpg')

detect_faces_in_image_facerecognition('IvanMladek.jpg')
# Обнаружение лиц в видео с помощью Face Recognition
detect_faces_with_facerecognition('JozinzBazin.mp4')

from mtcnn import MTCNN

# Функция для обнаружения лиц на изображении с помощью MTCNN
def detect_faces_image_with_mtcnn(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Инициализация MTCNN
    detector = MTCNN()

    # Обнаружение лиц на изображении
    result = detector.detect_faces(image)

    # Отрисовка прямоугольников вокруг обнаруженных лиц
    for face in result:
        x, y, w, h = face['box']
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Отображение изображения с лицами
    cv2.imshow('Faces Detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Функция для обнаружения лиц на видео с помощью MTCNN
def detect_faces_video_with_mtcnn(video_path):
    # Инициализация MTCNN
    detector = MTCNN()

    # Загрузка видео
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Обнаружение лиц на кадре
        result = detector.detect_faces(frame)

        # Отрисовка прямоугольников вокруг обнаруженных лиц
        for face in result:
            x, y, w, h = face['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Отображение кадра с лицами
        cv2.imshow('Faces Detected', frame)

        # Выход из цикла при нажатии клавиши 'q'
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # Очистка ресурсов
    cap.release()
    cv2.destroyAllWindows()


# часть 3 Обнаружение лиц на изображении с помощью MTCNN
detect_faces_image_with_mtcnn('IvoPesak.jpg')

detect_faces_image_with_mtcnn('IvanMladek.jpg')
# Обнаружение лиц в видео с помощью Face Recognition
detect_faces_video_with_mtcnn('JozinzBazin.mp4')

# выводы: библиотеки справились лучше, чем ручная реализация через OpenCV
# библиотека MTCNN работает значительно медленнее, чем face_recognition


# ищем заданные лица в видео


# Загрузка изображений с лицами
image1 = face_recognition.load_image_file("IvanMladek.jpg")
image2 = face_recognition.load_image_file("IvoPesak.jpg")

# Получение эмбеддингов (векторов признаков) для изображений с лицами
face_encoding1 = face_recognition.face_encodings(image1)[0]
face_encoding2 = face_recognition.face_encodings(image2)[0]

# Загрузка видео
video_path = "JozinzBazin.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Конвертирование кадра в формат RGB (требуется для работы с Face Recognition)
    rgb_frame = frame

    # Обнаружение лиц на кадре
    face_locations = face_recognition.face_locations(rgb_frame)

    if len(face_locations) > 0:
        # Получение эмбеддингов (векторов признаков) для лиц на кадре (если они обнаружены)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Сравнение эмбеддингов лиц на кадре с эмбеддингами изображений с лицами
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Сравнение с изображением 1
            match1 = face_recognition.compare_faces([face_encoding1], face_encoding, tolerance=0.6)
            # Сравнение с изображением 2
            match2 = face_recognition.compare_faces([face_encoding2], face_encoding, tolerance=0.6)

            # Отображение прямоугольников вокруг лиц с разными цветами, соответствующими каждому изображению
            if match1[0]:
                color = (0, 255, 0)  # Зеленый цвет для изображения 1
            elif match2[0]:
                color = (0, 0, 255)  # Красный цвет для изображения 2
            else:
                color = (255, 0, 0)  # Синий цвет для других лиц

            # Отрисовка прямоугольника вокруг лица
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Отображение кадра с выделенными лицами
        cv2.imshow('Faces Detected', frame)

        # Выход из цикла при нажатии клавиши 'q'
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

# Очистка ресурсов
cap.release()
cv2.destroyAllWindows()
