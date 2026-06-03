# Sistema MLOps para despliegue automatico de un modelo ONNX

## 1. Resumen del proyecto

En este repositorio presentamos una solucion MLOps sencilla para automatizar el despliegue de nuevas versiones de un modelo de machine learning en formato ONNX. Nuestra solucion utiliza un modelo clasificador de flores Iris y expone una API REST para que usuarios finales puedan enviar datos de entrada y recibir una prediccion.

Disenamos el sistema para cumplir los requisitos del taller:

- Repositorio en GitHub.
- Pipeline CI/CD con GitHub Actions.
- Dos ramas principales: `dev` y `prod`.
- Un endpoint asociado a cada rama.
- Etapa de pruebas automatizadas.
- Etapa de construccion y promocion del contenedor.
- Modelo ONNX almacenado fuera del repositorio.
- Datos de prueba almacenados fuera del repositorio.
- Registro de predicciones en archivos TXT externos para monitoreo futuro.

## 2. Aplicacion seleccionada

La aplicacion que implementamos es una API de prediccion para el conjunto de datos Iris. A partir de cuatro variables numericas de una flor:

- Longitud del sepalo.
- Ancho del sepalo.
- Longitud del petalo.
- Ancho del petalo.

El modelo predice una de las siguientes clases:

- `setosa`
- `versicolor`
- `virginica`

Desarrollamos la API con FastAPI y utilizamos ONNX Runtime para ejecutar el modelo.

## 3. Proveedor de nube seleccionado

Para el despliegue seleccionamos **Hugging Face** porque permite crear Spaces con Docker sin costo para una demostracion academica. Esta alternativa nos permite evitar gastos y es suficiente para sustentar el funcionamiento del sistema ante el docente.

Se utilizan dos componentes de Hugging Face:

- **Hugging Face Spaces**: para desplegar los contenedores de la API.
- **Hugging Face Dataset**: como almacenamiento externo tipo bucket para modelo, datos de prueba y logs.

## 4. Arquitectura propuesta

La arquitectura general es la siguiente:

```text
GitHub branch dev
        |
        v
GitHub Actions: test + build/promote
        |
        v
Hugging Face Space DEV
        |
        v
Endpoint dev


GitHub branch prod
        |
        v
GitHub Actions: test + build/promote
        |
        v
Hugging Face Space PROD
        |
        v
Endpoint prod
```

El almacenamiento externo se maneja en un Dataset de Hugging Face:

```text
ccar9636/mlops-iris-assets
```

Este Dataset contiene:

```text
model/iris_model.onnx
test/test_data.csv
logs/predicciones_dev.txt
logs/predicciones_prod.txt
```

## 5. Ramas del repositorio

El repositorio trabaja con dos ramas:

```text
dev
prod
```

La rama `dev` representa el ambiente de desarrollo. Cada cambio enviado a esta rama ejecuta el pipeline y actualiza el endpoint de desarrollo.

La rama `prod` representa el ambiente de produccion. Cada cambio enviado a esta rama ejecuta el pipeline y actualiza el endpoint productivo.

## 6. Endpoints de la solucion

Endpoint de desarrollo:

```text
https://ccar9636-mlops-iris-dev.hf.space
```

Documentacion interactiva del endpoint dev:

```text
https://ccar9636-mlops-iris-dev.hf.space/docs
```

Endpoint de produccion:

```text
https://ccar9636-mlops-iris-prod.hf.space
```

Documentacion interactiva del endpoint prod:

```text
https://ccar9636-mlops-iris-prod.hf.space/docs
```

## 7. Modelo ONNX

El modelo se encuentra en formato ONNX y no esta almacenado directamente dentro del repositorio de GitHub. Esto cumple con el requisito de mantener el modelo fuera del repositorio y descargarlo desde un almacenamiento externo durante el proceso de CI/CD.

Ubicacion externa del modelo:

```text
https://huggingface.co/datasets/ccar9636/mlops-iris-assets/resolve/main/model/iris_model.onnx
```

Durante el pipeline, el modelo se descarga mediante el script:

```text
scripts/download_model.py
```

## 8. Datos de prueba

Los datos de prueba tampoco estan almacenados directamente en el repositorio. Se encuentran en el Dataset de Hugging Face:

```text
https://huggingface.co/datasets/ccar9636/mlops-iris-assets/resolve/main/test/test_data.csv
```

Durante la etapa de pruebas, estos datos se descargan para evaluar el comportamiento del modelo.

## 9. Pipeline CI/CD

El pipeline se encuentra definido en:

```text
.github/workflows/mlops.yml
```

Se ejecuta automaticamente cuando hay un `push` a cualquiera de estas ramas:

```text
dev
prod
```

El pipeline tiene dos jobs principales:

## 9.1. Job test

La etapa `test` realiza las siguientes acciones:

1. Descarga el codigo del repositorio.
2. Configura Python 3.11.
3. Instala las dependencias del proyecto.
4. Descarga el modelo ONNX desde Hugging Face Dataset.
5. Ejecuta pruebas unitarias con `pytest`.

Las pruebas implementadas son:

- Verificar que el modelo responde ante una entrada definida.
- Verificar que el modelo mantiene una exactitud minima de `0.90`.

Estas pruebas se encuentran en:

```text
tests/test_model.py
```

## 9.2. Job build/promote

La etapa `build/promote` se ejecuta solamente si `test` finaliza correctamente.

Esta etapa realiza:

1. Descarga el codigo del repositorio.
2. Construye una imagen Docker usando el `Dockerfile`.
3. Selecciona el Space de Hugging Face correspondiente segun la rama:
   - `dev` despliega en `mlops-iris-dev`.
   - `prod` despliega en `mlops-iris-prod`.
4. Valida que existan los secretos necesarios.
5. Publica el codigo en el Space correspondiente de Hugging Face.

## 10. Contenedor Docker

El repositorio incluye un archivo:

```text
Dockerfile
```

El contenedor instala las dependencias, copia la aplicacion y arranca la API con Uvicorn en el puerto `7860`, puerto esperado por Hugging Face Spaces.

La aplicacion tambien esta preparada para descargar el modelo al iniciar si el archivo ONNX no existe dentro del contenedor.

## 11. Aplicacion FastAPI

La API principal esta en:

```text
app/main.py
```

Endpoints disponibles:

```text
GET  /
GET  /health
POST /predict
GET  /docs
```

Ejemplo de entrada para `POST /predict`:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Ejemplo de respuesta:

```json
{
  "environment": "dev",
  "prediction": {
    "class_id": 0,
    "class_name": "setosa",
    "probabilities": {
      "setosa": 0.99,
      "versicolor": 0.01,
      "virginica": 0.0
    }
  }
}
```

## 12. Registro de predicciones

Cada llamada al endpoint `/predict` genera una linea nueva en un archivo TXT externo.

Para el ambiente dev:

```text
logs/predicciones_dev.txt
```

Para el ambiente prod:

```text
logs/predicciones_prod.txt
```

Ambos archivos se almacenan en el Dataset:

```text
https://huggingface.co/datasets/ccar9636/mlops-iris-assets/tree/main/logs
```

Cada linea contiene informacion en formato JSON con:

- Fecha y hora UTC.
- Ambiente (`dev` o `prod`).
- Datos enviados por el usuario.
- Prediccion generada por el modelo.

Estos archivos pueden usarse posteriormente para monitoreo, auditoria o analisis de comportamiento del modelo.

## 13. Secretos utilizados

El pipeline utiliza secretos de GitHub Actions para no exponer informacion sensible.

Secretos configurados:

```text
HF_TOKEN
HF_USERNAME
HF_SPACE_DEV
HF_SPACE_PROD
MODEL_URL
TEST_DATA_URL
```

Estos secretos permiten descargar activos, construir la imagen Docker y desplegar automaticamente en Hugging Face Spaces.

## 14. Evidencia de funcionamiento

La solucion fue validada con los siguientes resultados:

- Pipeline exitoso en rama `dev`.
- Pipeline exitoso en rama `prod`.
- Endpoint dev desplegado en Hugging Face Spaces.
- Endpoint prod desplegado en Hugging Face Spaces.
- Archivos `predicciones_dev.txt` y `predicciones_prod.txt` actualizados en Hugging Face Dataset.

## 15. Como sustentar la solucion

Para la sustentacion recomendamos mostrar:

1. Repositorio de GitHub con las ramas `dev` y `prod`.
2. Archivo `.github/workflows/mlops.yml`.
3. Ejecuciones exitosas del pipeline en GitHub Actions.
4. Dataset externo con el modelo ONNX y los datos de prueba.
5. Endpoint dev funcionando en `/docs`.
6. Endpoint prod funcionando en `/docs`.
7. Archivos TXT de predicciones actualizados en Hugging Face Dataset.

## 16. Conclusiones

La solucion que implementamos demuestra un flujo MLOps basico pero funcional. El sistema permite probar automaticamente un modelo ONNX, construir una aplicacion contenedorizada y desplegarla automaticamente en dos ambientes separados.

Ademas, la arquitectura evita almacenar el modelo y los datos de prueba dentro del repositorio, y registra las predicciones en archivos externos para posibles analisis futuros. Todo el despliegue se realiza con servicios gratuitos adecuados para una demostracion academica.
