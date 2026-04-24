# Как запустить SHAP анализ на Kaggle

## 1. Заходишь на kaggle.com → New Notebook

## 2. Загружаешь датасет
- Справа нажимаешь **Add Data** → **Upload** → загружаешь `dataset.csv`
- Либо ищешь в поиске: `predict students dropout academic success`

## 3. Загружаешь скрипт
- Слева **+ Add** → **Upload** → загружаешь `shap_analysis_kaggle.py`
- Или просто копируешь весь код в ячейку ноутбука

## 4. Меняешь путь к датасету
Найди эту строку в коде:
```python
df = pd.read_csv("/kaggle/input/your-dataset-folder/dataset.csv")
```
Замени на реальный путь. Чтобы узнать путь — запусти в отдельной ячейке:
```python
import os
for f in os.walk('/kaggle/input'):
    print(f)
```

## 5. Запускаешь
- **Run All** (▶▶) — ждёшь ~3-5 минут
- SHAP на 4424 студентах считается быстро

## 6. Скачиваешь фигуры
После выполнения справа появится папка **Output** →
скачиваешь три файла:
- `fig16_shap_beeswarm.png`
- `fig17_shap_global_importance.png`  
- `fig18_shap_waterfall.png`

## 7. Готово
Три графика добавляешь в статью как новый подраздел **5.7 Explainability Analysis (SHAP)**
