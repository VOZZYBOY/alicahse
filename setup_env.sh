#!/bin/bash

# Создание и активация виртуального окружения
python -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt

# Создание директорий для данных и результатов
mkdir -p datasets/human_images
mkdir -p results

echo "Виртуальное окружение настроено. Используйте 'source venv/bin/activate' для активации." 