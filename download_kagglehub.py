#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import shutil
import argparse
import kagglehub
from tqdm import tqdm


def download_dataset(dataset_name, output_dir='datasets/human_images'):

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Загрузка датасета {dataset_name}...")
    
   
    try:
        path = kagglehub.dataset_download(dataset_name)
        print("Датасет успешно загружен в:", path)
        
        print(f"Копирование файлов в {output_dir}...")
        
        
        file_count = 0
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_count += 1
        
        copied = 0
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(root, file)
                    
                    rel_path = os.path.relpath(root, path)
                    
                    dst_dir = os.path.join(output_dir, rel_path)
                    os.makedirs(dst_dir, exist_ok=True)
                  
                    dst = os.path.join(dst_dir, file)
                    shutil.copy2(src, dst)
                    copied += 1
                   
                    if copied % 10 == 0:
                        print(f"Скопировано {copied}/{file_count} файлов...")
        
        print(f"Датасет успешно скопирован в {output_dir}")
        print(f"Всего скопировано {copied} изображений")
        
        return True
    except Exception as e:
        print(f"Ошибка при загрузке датасета: {e}")
        return False


def main():
    
    parser = argparse.ArgumentParser(description="Загрузка датасета с Kaggle через kagglehub")
    parser.add_argument("--dataset", type=str, default="snmahsa/human-images-dataset-men-and-women",
                       help="Имя датасета на Kaggle")
    parser.add_argument("--output", type=str, default="datasets/human_images",
                       help="Директория для сохранения датасета")
    
    args = parser.parse_args()
    
   
    download_dataset(args.dataset, args.output)


if __name__ == "__main__":
    main() 
