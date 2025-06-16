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


## Descarga de Datos

Debido al tamaño de los datos, los archivos preprocesados (splits, XMLs ) se alojan en Google Drive. Puedes acceder a la carpeta completa desde el siguiente enlace:

[Reporte Consultoria 2025] (https://drive.google.com/drive/folders/1jD4hBMhydmnj4G4p2XauA_PYQSd1BEb-?hl=es)

### Instrucciones para la descarga

1. Accede al enlace y descarga el archivo `data_consultoria_2025.zip`.


2. Descomprime los datos en la carpeta `data/processed`:
   ```bash
   unzip data.zip -d data/processed
