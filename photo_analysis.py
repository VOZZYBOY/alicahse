"""
Анализ набора фотографий с использованием нейросетевых инструментов.
Программа обнаруживает объекты на фотографиях, определяет лица, анализирует эмоции
и создает визуализации на основе полученных данных.
"""

import os
import glob
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from deepface import DeepFace


def get_image_paths(dir_path):
    """Получение списка путей ко всем изображениям в директории"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(dir_path, '**', ext), recursive=True))
    return image_paths


def detect_objects(image_path, yolo_model):
    """Обнаружение объектов на изображении с помощью YOLO"""
    try:
        # Выполняем инференс
        results = yolo_model(image_path)
        
        # Извлекаем названия классов и конфиденс
        objects = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls)
                class_name = r.names[class_id]
                confidence = float(box.conf)
                objects.append({'class': class_name, 'confidence': confidence})
        
        return objects
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        return []


def analyze_faces(image_path):
    """Обнаружение лиц и анализ эмоций с помощью DeepFace"""
    try:
        # Анализируем изображение
        result = DeepFace.analyze(image_path, 
                                 actions=['emotion', 'age', 'gender'],
                                 enforce_detection=False,
                                 silent=True)
        
        # DeepFace возвращает список или словарь в зависимости от количества обнаруженных лиц
        if not isinstance(result, list):
            result = [result]
            
        return result
    except Exception as e:
        print(f"Ошибка при анализе лиц в {image_path}: {e}")
        return []


def is_indoor(objects):
    """Определение типа местоположения (помещение/улица) на основе обнаруженных объектов"""
    indoor_objects = ['chair', 'couch', 'bed', 'tv', 'laptop', 'dining table', 'refrigerator', 'microwave', 'oven', 'toaster', 'sink', 'toilet']
    outdoor_objects = ['car', 'bicycle', 'motorcycle', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'tree', 'bench']
    
    indoor_count = sum(1 for obj in objects if obj['class'] in indoor_objects)
    outdoor_count = sum(1 for obj in objects if obj['class'] in outdoor_objects)
    
    if indoor_count > outdoor_count:
        return 'помещение'
    elif outdoor_count > indoor_count:
        return 'улица'
    else:
        return 'неопределено'


def has_animal(objects, animal_type):
    """Проверка наличия определенного животного на фото"""
    return any(obj['class'] == animal_type for obj in objects)


def extract_gender_from_path(path):
    """Извлечение информации о поле из пути к файлу (для датасета Kaggle)"""
    path_lower = path.lower()
    if 'male' in path_lower and not 'female' in path_lower:
        return 'male'
    elif 'female' in path_lower:
        return 'female'
    else:
        return 'unknown'


def visualize_object_frequency(objects_df, output_dir='results'):
    """Визуализация частоты встречаемости объектов на фотографиях"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Подсчитываем частоту встречаемости объектов
    object_counts = objects_df['object_class'].value_counts().reset_index()
    object_counts.columns = ['object_class', 'count']

    # Выбираем топ-20 объектов для визуализации
    top_objects = object_counts.head(20)

    # Визуализируем
    plt.figure(figsize=(14, 8))
    sns.barplot(data=top_objects, x='count', y='object_class', palette='viridis')
    plt.title('Топ-20 объектов, обнаруженных на фотографиях')
    plt.xlabel('Количество объектов')
    plt.ylabel('Класс объекта')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'object_frequency.png'), dpi=300)
    plt.close()


def visualize_location_distribution(image_df, output_dir='results'):
    """Визуализация распределения фотографий по типу местоположения"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Визуализируем распределение по типу местоположения
    plt.figure(figsize=(10, 6))
    locations = image_df['location_type'].value_counts()
    sns.barplot(x=locations.index, y=locations.values, palette='Set2')
    plt.title('Распределение фотографий по типу местоположения')
    plt.xlabel('Тип местоположения')
    plt.ylabel('Количество фотографий')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'location_bar.png'), dpi=300)
    plt.close()

    # Также сделаем круговую диаграмму
    plt.figure(figsize=(8, 8))
    plt.pie(locations.values, labels=locations.index, autopct='%1.1f%%', colors=sns.color_palette('Set2'))
    plt.title('Распределение фотографий по типу местоположения')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'location_pie.png'), dpi=300)
    plt.close()


def visualize_emotion_by_gender(faces_with_gender, output_dir='results'):
    """Визуализация распределения эмоций в зависимости от пола"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Проверяем, что данные не пусты
    if faces_with_gender.empty:
        print("Предупреждение: нет данных для визуализации эмоций по полу")
        return
    
    # Удаляем строки с NaN значениями в ключевых столбцах
    clean_df = faces_with_gender.dropna(subset=['gender_from_path', 'dominant_emotion'])
    
    if clean_df.empty:
        print("Предупреждение: после очистки от NaN значений не осталось данных для визуализации")
        return
    
    print(f"Визуализация эмоций по полу: обрабатываем {len(clean_df)} записей")
    
    try:
        # Анализируем распределение эмоций в зависимости от пола
        emotion_gender_counts = clean_df.groupby(['gender_from_path', 'dominant_emotion']).size().reset_index(name='count')

        # Визуализируем результаты
        plt.figure(figsize=(14, 8))
        sns.barplot(data=emotion_gender_counts, x='dominant_emotion', y='count', hue='gender_from_path', palette='Set1')
        plt.title('Распределение эмоций в зависимости от пола')
        plt.xlabel('Доминирующая эмоция')
        plt.ylabel('Количество лиц')
        plt.legend(title='Пол')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'emotion_by_gender.png'), dpi=300)
        plt.close()
        
        # Проверяем наличие столбца detected_gender
        if 'detected_gender' in clean_df.columns:
            try:
                # Удаляем строки с NaN в detected_gender
                clean_df = clean_df.dropna(subset=['detected_gender'])
                
                if not clean_df.empty:
                    # Проверим соответствие между полом из пути файла и полом, определенным DeepFace
                    gender_comparison = clean_df.groupby(['gender_from_path', 'detected_gender']).size().reset_index(name='count')

                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=gender_comparison, x='gender_from_path', y='count', hue='detected_gender', palette='pastel')
                    plt.title('Соответствие между полом из названия файла и определенным DeepFace')
                    plt.xlabel('Пол из названия файла')
                    plt.ylabel('Количество')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'gender_comparison.png'), dpi=300)
                    plt.close()
            except Exception as e:
                print(f"Предупреждение: не удалось создать график сравнения полов: {e}")
    except Exception as e:
        print(f"Ошибка при визуализации эмоций по полу: {e}")
        # Выведем информацию о данных для отладки
        print(f"Уникальные значения gender_from_path: {clean_df['gender_from_path'].unique()}")
        print(f"Уникальные значения dominant_emotion: {clean_df['dominant_emotion'].unique()}")


def visualize_age_by_gender(faces_with_gender, output_dir='results'):
    """Визуализация возрастного распределения в зависимости от пола"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Проверяем, что данные не пусты
    if faces_with_gender.empty:
        print("Предупреждение: нет данных для визуализации возраста по полу")
        return
    
    # Удаляем строки с NaN значениями в необходимых столбцах
    clean_df = faces_with_gender.dropna(subset=['gender_from_path', 'age'])
    
    if clean_df.empty:
        print("Предупреждение: после очистки от NaN значений не осталось данных для визуализации возраста")
        return
        
    # Преобразуем возраст в числовой тип, пропуская ошибки
    clean_df['age'] = pd.to_numeric(clean_df['age'], errors='coerce')
    clean_df = clean_df.dropna(subset=['age'])
    
    if clean_df.empty:
        print("Предупреждение: после преобразования возраста не осталось данных для визуализации")
        return
    
    print(f"Визуализация возраста по полу: обрабатываем {len(clean_df)} записей")
    
    try:
        # Анализ возрастного распределения в зависимости от пола
        plt.figure(figsize=(12, 6))
        sns.histplot(data=clean_df, x='age', hue='gender_from_path', bins=20, kde=True, palette='Set1')
        plt.title('Распределение возраста по полу')
        plt.xlabel('Возраст')
        plt.ylabel('Количество')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'age_histogram.png'), dpi=300)
        plt.close()

        # Ящик с усами для возраста по полу
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=clean_df, x='gender_from_path', y='age', palette='Set2')
        plt.title('Распределение возраста по полу (ящик с усами)')
        plt.xlabel('Пол')
        plt.ylabel('Возраст')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'age_boxplot.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Ошибка при визуализации возраста по полу: {e}")


def visualize_emotion_by_location(faces_with_location, output_dir='results'):
    """Визуализация распределения эмоций в зависимости от типа местоположения"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Проверяем, что данные не пусты
    if faces_with_location.empty:
        print("Предупреждение: нет данных для визуализации эмоций по местоположению")
        return
    
    # Удаляем строки с NaN значениями в ключевых столбцах
    clean_df = faces_with_location.dropna(subset=['location_type', 'dominant_emotion'])
    
    if clean_df.empty:
        print("Предупреждение: после очистки от NaN значений не осталось данных для визуализации местоположений")
        return
    
    print(f"Визуализация эмоций по местоположению: обрабатываем {len(clean_df)} записей")
    
    try:
        # Анализируем распределение эмоций в зависимости от типа местоположения
        emotion_location_counts = clean_df.groupby(['location_type', 'dominant_emotion']).size().reset_index(name='count')

        # Визуализируем результаты
        plt.figure(figsize=(14, 8))
        sns.barplot(data=emotion_location_counts, x='dominant_emotion', y='count', hue='location_type', palette='Set2')
        plt.title('Распределение эмоций в зависимости от типа местоположения')
        plt.xlabel('Доминирующая эмоция')
        plt.ylabel('Количество лиц')
        plt.legend(title='Тип местоположения')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'emotion_by_location.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Ошибка при визуализации эмоций по местоположению: {e}")
        # Выведем информацию о данных для отладки
        print(f"Уникальные значения location_type: {clean_df['location_type'].unique()}")
        print(f"Уникальные значения dominant_emotion: {clean_df['dominant_emotion'].unique()}")


def visualize_emotion_by_animals(faces_df, animal_df, image_df, output_dir='results'):
    """Визуализация влияния наличия животных (кошек и собак) на эмоции людей"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Объединяем данные о лицах и наличии животных
    faces_with_animals = pd.DataFrame()
    
    try:
        # Сначала создаем DataFrame с признаками животных для всех изображений
        image_with_animals = image_df[['image_path', 'has_cat', 'has_dog']]
        
        # Теперь объединяем с данными о лицах
        for idx, row in faces_df.iterrows():
            img_path = row['image_path']
            animals_data = image_with_animals[image_with_animals['image_path'] == img_path]
            if len(animals_data) > 0:
                has_cat = animals_data['has_cat'].values[0]
                has_dog = animals_data['has_dog'].values[0]
                
                faces_with_animals = pd.concat([faces_with_animals, pd.DataFrame([{
                    'image_path': img_path,
                    'dominant_emotion': row['dominant_emotion'],
                    'has_cat': has_cat,
                    'has_dog': has_dog,
                    'has_animal': has_cat or has_dog
                }])], ignore_index=True)
        
        # Удаляем строки с NaN значениями
        clean_df = faces_with_animals.dropna(subset=['dominant_emotion'])
        
        if clean_df.empty:
            print("Предупреждение: нет данных для визуализации влияния животных на эмоции")
            return
        
        print(f"Визуализация влияния животных на эмоции: обрабатываем {len(clean_df)} записей")
        
        # Влияние кошек на эмоции
        emotion_cat_counts = clean_df.groupby(['has_cat', 'dominant_emotion']).size().reset_index(name='count')
        
        plt.figure(figsize=(14, 8))
        sns.barplot(data=emotion_cat_counts, x='dominant_emotion', y='count', hue='has_cat', palette='Set1')
        plt.title('Распределение эмоций в зависимости от наличия кошек на фото')
        plt.xlabel('Доминирующая эмоция')
        plt.ylabel('Количество лиц')
        plt.legend(title='Наличие кошки', labels=['Нет кошки', 'Есть кошка'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'emotion_by_cats.png'), dpi=300)
        plt.close()
        
        # Влияние собак на эмоции
        emotion_dog_counts = clean_df.groupby(['has_dog', 'dominant_emotion']).size().reset_index(name='count')
        
        plt.figure(figsize=(14, 8))
        sns.barplot(data=emotion_dog_counts, x='dominant_emotion', y='count', hue='has_dog', palette='Set1')
        plt.title('Распределение эмоций в зависимости от наличия собак на фото')
        plt.xlabel('Доминирующая эмоция')
        plt.ylabel('Количество лиц')
        plt.legend(title='Наличие собаки', labels=['Нет собаки', 'Есть собака'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'emotion_by_dogs.png'), dpi=300)
        plt.close()
        
        # Влияние животных (кошек или собак) на эмоции
        emotion_animal_counts = clean_df.groupby(['has_animal', 'dominant_emotion']).size().reset_index(name='count')
        
        plt.figure(figsize=(14, 8))
        sns.barplot(data=emotion_animal_counts, x='dominant_emotion', y='count', hue='has_animal', palette='Set1')
        plt.title('Распределение эмоций в зависимости от наличия животных на фото')
        plt.xlabel('Доминирующая эмоция')
        plt.ylabel('Количество лиц')
        plt.legend(title='Наличие животного', labels=['Нет животного', 'Есть животное'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'emotion_by_animals.png'), dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Ошибка при визуализации влияния животных на эмоции: {e}")


def visualize_face_count_distribution(image_df, output_dir='results'):
    """Визуализация распределения количества лиц на фотографиях"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Проверяем, что данные не пусты и содержат нужный столбец
    if image_df.empty or 'face_count' not in image_df.columns:
        print("Предупреждение: нет данных для визуализации распределения количества лиц")
        return
    
    # Ограничиваем до разумных значений для наглядности (отсекаем выбросы)
    max_faces_to_show = 10
    clean_df = image_df.copy()
    clean_df.loc[clean_df['face_count'] > max_faces_to_show, 'face_count'] = max_faces_to_show
    
    try:
        # Распределение количества лиц
        plt.figure(figsize=(12, 6))
        face_count_values = clean_df['face_count'].value_counts().sort_index()
        
        # Визуализируем гистограмму
        plt.bar(face_count_values.index, face_count_values.values, color='skyblue')
        plt.title('Распределение количества лиц на фотографиях')
        plt.xlabel('Количество лиц')
        plt.ylabel('Количество фотографий')
        plt.xticks(range(max_faces_to_show + 1))
        if max_faces_to_show in face_count_values.index:
            plt.text(max_faces_to_show, face_count_values[max_faces_to_show], 
                    f"≥ {max_faces_to_show}", ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'face_count_distribution.png'), dpi=300)
        plt.close()
        
        # Распределение по полу
        if 'gender_from_path' in clean_df.columns:
            # Создаем сводную таблицу: количество лиц на фото и пол
            gender_face_counts = pd.crosstab(
                clean_df['gender_from_path'], 
                clean_df['face_count'], 
                normalize='index'
            ).reset_index()
            
            # Переводим в "длинный" формат для seaborn
            gender_face_counts_melted = pd.melt(
                gender_face_counts, 
                id_vars=['gender_from_path'], 
                var_name='face_count', 
                value_name='proportion'
            )
            
            # Визуализируем
            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=gender_face_counts_melted, 
                x='face_count', 
                y='proportion', 
                hue='gender_from_path',
                palette='Set1'
            )
            plt.title('Распределение количества лиц на фотографиях по полу')
            plt.xlabel('Количество лиц')
            plt.ylabel('Доля фотографий')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'face_count_by_gender.png'), dpi=300)
            plt.close()
    
    except Exception as e:
        print(f"Ошибка при визуализации распределения количества лиц: {e}")


def analyze_photo_archive(image_dir, sample_size=500, output_dir='results'):
    """Основная функция для анализа фотоархива"""
    
    print(f"Начинаем анализ фотоархива в директории: {image_dir}")
    
    # Получаем список путей к фотографиям
    image_paths = get_image_paths(image_dir)
    print(f"Найдено {len(image_paths)} изображений")
    
    if not image_paths:
        print("Не найдено изображений. Проверьте путь к директории.")
        return
    
    # Ограничиваем количество изображений для анализа
    sample_size = min(sample_size, len(image_paths))
    sampled_images = np.random.choice(image_paths, sample_size, replace=False)
    print(f"Будет проанализировано {sample_size} изображений")
    
    # Загрузка предобученной модели YOLO
    print("Загрузка модели YOLO...")
    yolo_model = YOLO('yolov8n.pt')
    
    # Обнаружение объектов на выбранных изображениях
    print("Обнаружение объектов...")
    object_results = {}
    
    for img_path in tqdm(sampled_images):
        objects = detect_objects(img_path, yolo_model)
        object_results[img_path] = objects
    
    # Создаем DataFrame с информацией об объектах
    objects_data = []
    for img_path, objects in object_results.items():
        for obj in objects:
            objects_data.append({
                'image_path': img_path,
                'object_class': obj['class'],
                'confidence': obj['confidence']
            })
    
    objects_df = pd.DataFrame(objects_data)
    print(f"Обнаружено {len(objects_data)} объектов на {len(object_results)} изображениях")
    
    # Обнаружение лиц и анализ эмоций на выбранных изображениях
    print("Обнаружение лиц и анализ эмоций...")
    face_results = {}
    
    for img_path in tqdm(sampled_images):
        faces = analyze_faces(img_path)
        face_results[img_path] = faces
    
    # Создаем DataFrame с информацией о лицах
    faces_data = []
    for img_path, faces in face_results.items():
        for face in faces:
            try:
                faces_data.append({
                    'image_path': img_path,
                    'emotion': face['emotion'],
                    'dominant_emotion': face['dominant_emotion'],
                    'age': face['age'],
                    'gender': face['gender']
                })
            except (KeyError, TypeError) as e:
                print(f"Ошибка при обработке данных лица: {e}")
    
    faces_df = pd.DataFrame(faces_data)
    print(f"Обнаружено {len(faces_data)} лиц на {len(face_results)} изображениях")
    
    # Применяем функцию определения местоположения ко всем изображениям
    location_data = []
    for img_path, objects in object_results.items():
        location = is_indoor(objects)
        location_data.append({
            'image_path': img_path,
            'location_type': location
        })
    
    location_df = pd.DataFrame(location_data)
    print("Распределение по типу местоположения:")
    print(location_df['location_type'].value_counts())
    
    # Создаем DataFrame с признаком наличия кота/собаки на фото
    animal_data = []
    for img_path, objects in object_results.items():
        cat_present = has_animal(objects, 'cat')
        dog_present = has_animal(objects, 'dog')
        animal_data.append({
            'image_path': img_path,
            'has_cat': cat_present,
            'has_dog': dog_present
        })
    
    animal_df = pd.DataFrame(animal_data)
    
    # Объединяем данные
    image_df = pd.DataFrame({'image_path': list(object_results.keys())})
    image_df = image_df.merge(location_df, on='image_path', how='left')
    image_df = image_df.merge(animal_df, on='image_path', how='left')
    
    # Добавляем информацию о количестве лиц на фото
    face_counts = {}
    for img_path, faces in face_results.items():
        face_counts[img_path] = len(faces)
    
    image_df['face_count'] = image_df['image_path'].map(face_counts).fillna(0).astype(int)
    
    # Добавляем информацию о поле (на основе пути файла)
    image_df['gender_from_path'] = image_df['image_path'].apply(extract_gender_from_path)
    
    # Объединяем данные о лицах и пути к файлам
    faces_with_gender = pd.DataFrame()
    for idx, row in faces_df.iterrows():
        img_path = row['image_path']
        gender_from_path = extract_gender_from_path(img_path)
        faces_with_gender = pd.concat([faces_with_gender, pd.DataFrame([{
            'image_path': img_path,
            'dominant_emotion': row['dominant_emotion'],
            'gender_from_path': gender_from_path,
            'detected_gender': row['gender'],
            'age': row['age']
        }])], ignore_index=True)
    
    # Объединяем данные о лицах и типе местоположения
    faces_with_location = pd.DataFrame()
    for idx, row in faces_df.iterrows():
        img_path = row['image_path']
        location_type = location_df[location_df['image_path'] == img_path]['location_type'].values
        if len(location_type) > 0:
            faces_with_location = pd.concat([faces_with_location, pd.DataFrame([{
                'image_path': img_path,
                'dominant_emotion': row['dominant_emotion'],
                'location_type': location_type[0]
            }])], ignore_index=True)
    
    print("Создание визуализаций...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Визуализируем частоту встречаемости объектов
    visualize_object_frequency(objects_df, output_dir)
    
    # Визуализируем распределение по типу местоположения
    visualize_location_distribution(image_df, output_dir)
    
    # Визуализируем распределение эмоций в зависимости от пола
    visualize_emotion_by_gender(faces_with_gender, output_dir)
    
    # Визуализируем возрастное распределение
    visualize_age_by_gender(faces_with_gender, output_dir)
    
    # Визуализируем распределение эмоций в зависимости от местоположения
    visualize_emotion_by_location(faces_with_location, output_dir)
    
    # Визуализируем влияние животных на эмоции
    visualize_emotion_by_animals(faces_df, animal_df, image_df, output_dir)
    
    # Визуализируем распределение количества лиц на фотографиях
    visualize_face_count_distribution(image_df, output_dir)
    
    print(f"Анализ завершен. Результаты сохранены в директории: {output_dir}")

    objects_df.to_csv(os.path.join(output_dir, 'objects_data.csv'), index=False)
    faces_df.to_csv(os.path.join(output_dir, 'faces_data.csv'), index=False)
    image_df.to_csv(os.path.join(output_dir, 'image_data.csv'), index=False)
    
    print("\nОсновные результаты анализа:")
    print(f"Топ-5 объектов на фото: {objects_df['object_class'].value_counts().head(5).to_dict()}")
    print(f"Распределение местоположений: {image_df['location_type'].value_counts().to_dict()}")
    print(f"Среднее количество лиц на фото: {image_df['face_count'].mean():.2f}")
    
    return {
        'objects_df': objects_df,
        'faces_df': faces_df,
        'image_df': image_df,
        'faces_with_gender': faces_with_gender,
        'faces_with_location': faces_with_location
    }


def main():
    """Основная функция программы"""
    parser = argparse.ArgumentParser(description='Анализ фотоархива с использованием нейросетевых методов')
    parser.add_argument('--image_dir', type=str, default='datasets/human_images',
                        help='Путь к директории с фотографиями')
    parser.add_argument('--sample_size', type=int, default=500,
                        help='Количество фотографий для анализа (максимум)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Директория для сохранения результатов')
    
    args = parser.parse_args()
    
    analyze_photo_archive(args.image_dir, args.sample_size, args.output_dir)


if __name__ == "__main__":
    main() 