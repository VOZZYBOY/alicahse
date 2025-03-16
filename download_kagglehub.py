#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для загрузки датасета "Human Images Dataset Men and Women" через kagglehub.
Этот метод проще, так как не требует настройки учетных данных в файле kaggle.json.
"""

import os
import shutil
import argparse
import kagglehub
from tqdm import tqdm


def download_dataset(dataset_name, output_dir='datasets/human_images'):
    """Загрузка датасета с Kaggle через kagglehub"""
    
    # Создаем директорию для хранения датасета, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Загрузка датасета {dataset_name}...")
    
    # Загружаем датасет
    try:
        path = kagglehub.dataset_download(dataset_name)
        print("Датасет успешно загружен в:", path)
        
        # Копируем файлы из временной директории в указанную
        print(f"Копирование файлов в {output_dir}...")
        
        # Подсчет количества файлов для прогресс-бара
        file_count = 0
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_count += 1
        
        # Копирование файлов с отображением прогресса
        copied = 0
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(root, file)
                    # Создаем относительный путь
                    rel_path = os.path.relpath(root, path)
                    # Создаем директорию назначения
                    dst_dir = os.path.join(output_dir, rel_path)
                    os.makedirs(dst_dir, exist_ok=True)
                    # Копируем файл
                    dst = os.path.join(dst_dir, file)
                    shutil.copy2(src, dst)
                    copied += 1
                    # Отображаем прогресс
                    if copied % 10 == 0:
                        print(f"Скопировано {copied}/{file_count} файлов...")
        
        print(f"Датасет успешно скопирован в {output_dir}")
        print(f"Всего скопировано {copied} изображений")
        
        return True
    except Exception as e:
        print(f"Ошибка при загрузке датасета: {e}")
        return False


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="Загрузка датасета с Kaggle через kagglehub")
    parser.add_argument("--dataset", type=str, default="snmahsa/human-images-dataset-men-and-women",
                       help="Имя датасета на Kaggle")
    parser.add_argument("--output", type=str, default="datasets/human_images",
                       help="Директория для сохранения датасета")
    
    args = parser.parse_args()
    
    # Загружаем датасет
    download_dataset(args.dataset, args.output)


if __name__ == "__main__":
    main() 