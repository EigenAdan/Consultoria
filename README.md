# Reporte Consultoría: Explicabilidad en Modelos de Texto para Perfilado de Autor

## Descripción
Este proyecto investiga la explicabilidad en modelos de clasificación de texto aplicados al perfilado de autor en Twitter. Combinamos tres componentes principales:
- **EvoMSA**: ensamblado de representaciones BoW y DenseBoW con SVM lineal.  
- **Hierarchical Attention Network (HAN)**: red neuronal con atención a nivel de palabra y oración.  
- **LIME modificado**: adaptado para explicar localmente colecciones de 100 tuits como una unidad.

El objetivo es comparar la precisión y generar explicaciones globales y locales coherentes que permitan entender los patrones textuales que sustentan las decisiones de género y nacionalidad.

## Requisitos
- **Python** ≥ 3.8  
- **PyTorch** 2.5.1+cpu  
- **scikit-learn** 1.5.2  
- **NLTK** 3.9.1  
- **WordCloud** 1.9.4  
- **matplotlib** 3.9.2  
- **pandas** 2.2.3  
- **NumPy** 1.26.4  

