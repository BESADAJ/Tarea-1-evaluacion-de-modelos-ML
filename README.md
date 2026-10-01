# Tarea 1 — Evaluación de modelos ML

**Proyecto integrador de aprendizaje automático: predicción de *default* en préstamos de Lending Club, con scikit-learn y PySpark.**

📖 **Libro publicado:** https://besadaj.github.io/Tarea-1-evaluacion-de-modelos-ML/

**Integrantes:** Jaime Andres Besada, Ivan Prada

---

## Descripción

Se construyen y comparan seis clasificadores para predecir si un préstamo de **Lending Club** terminará en *default* (`Charged Off`) o será pagado completamente (`Fully Paid`). Se usa el dataset completo de préstamos aprobados entre 2007 y 2018: **1 345 310 préstamos finalizados, sin muestreo**.

Los mismos modelos se implementan en dos entornos, **scikit-learn** y **PySpark**, con la misma partición de entrenamiento y prueba (80/20 estratificada), para comparar su precisión y su velocidad.

## Contenido

1. **Análisis exploratorio (EDA):** variables univariadas y bivariadas, valores faltantes y resumen ejecutivo.
2. **Preprocesamiento:** imputación, escalado y *one-hot encoding*, ajustados solo con entrenamiento.
3. **Modelado:** regresión logística, árbol de decisión, *random forest*, *gradient boosting*, SVM lineal y Naive Bayes, con búsqueda de hiperparámetros por validación cruzada (`GridSearchCV` / `CrossValidator`).
4. **Comparación estadística:** prueba de **DeLong** con corrección de Holm, **McNemar** y **bootstrap pareado**.
5. **Interpretabilidad:** explicaciones locales con **LIME** en ambos entornos.
6. **Comparación de resultados y reflexión crítica.**

## Resultados principales

| Modelo | AUC scikit-learn | AUC PySpark |
|---|---|---|
| Gradient Boosting | **0.7187** | **0.7181** |
| Regresión logística | 0.7113 | 0.7113 |
| Random Forest | 0.7150 | 0.7051* |
| Árbol de decisión | 0.7028 | 0.6997 |
| Naive Bayes | 0.6587 | 0.6589 |
| SVM lineal | 0.4700 | 0.6675 |

\* En PySpark, el *grid* de Random Forest se redujo por capacidad de cómputo insuficiente.

- **Precisión:** con el mismo algoritmo, ambos entornos producen modelos prácticamente equivalentes.
- **Velocidad:** en un solo equipo (7.7 GB de RAM), scikit-learn fue más rápido. El entrenamiento completo tomó 2.0 h en scikit-learn frente a 3.9 h en PySpark.

## Estructura del repositorio

```
docs/
├── notebooks.ipynb        # Notebook completo, con todas las salidas
├── intro.md               # Portada del libro
├── _config.yml, _toc.yml  # Configuración del Jupyter Book
├── requirements.txt       # Dependencias
└── resultados/            # Partición común, métricas y puntuaciones de test de ambos entornos
```

## Reproducir

1. Descargar `accepted_2007_to_2018Q4.csv` del dataset [Lending Club Loan Data (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club) y guardarlo en `docs/`.
2. Instalar las dependencias con `pip install -r docs/requirements.txt`. PySpark requiere Java 17 o superior.
3. Ejecutar `docs/notebooks.ipynb`. Una ejecución completa tarda unas 6 horas en un equipo de 8 núcleos y 8 GB de RAM.

Para construir y publicar el libro:

```bash
jupyter-book build docs
ghp-import -n -p -f docs/_build/html
```
