Лабораторная работа 2 (вариант 7): полутоновое + бинаризация Сауволы 

1) Поместите полноцветные PNG/BMP изображения в input/.
2) Установите зависимости: pip install -r requirements.txt
3) Запуск: python main.py --input input --output output --windows 3,25 --k 0.5

Выходные файлы:
- *_original.png
- *_gray.bmp (взвешенное полутоновое)
- *_sauvola_w{window}_k{...}.bmp (бинарное)
