---
title: MLOps Iris ONNX
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# Taller MLOps: despliegue automatico de un modelo ONNX

Este proyecto presenta una solucion sencilla para sustentar el taller. Usamos:

- **FastAPI** para exponer un endpoint de prediccion.
- **ONNX Runtime** para ejecutar un modelo ONNX.
- **GitHub Actions** para CI/CD en las ramas `dev` y `prod`.
- **Hugging Face Spaces** para desplegar gratis dos endpoints: uno de desarrollo y uno de produccion.
- **Hugging Face Dataset** como "bucket" para guardar el modelo, los datos de prueba y los logs de predicciones.

La aplicacion predice la especie de una flor Iris usando cuatro numeros:

- `sepal_length`
- `sepal_width`
- `petal_length`
- `petal_width`

## Por que Hugging Face

Recomendamos **Hugging Face** para este taller porque permite desplegar una app Docker en Spaces sin pagar para una demo pequena. Tambien permite crear un repositorio tipo Dataset para guardar archivos, funcionando como un bucket sencillo.

Importante: el disco gratis de Hugging Face Spaces no es persistente. Por eso los logs de predicciones se guardan en un Dataset de Hugging Face, no dentro del contenedor.

## Arquitectura

```text
GitHub branch dev  -> GitHub Actions -> Hugging Face Space DEV  -> endpoint dev
GitHub branch prod -> GitHub Actions -> Hugging Face Space PROD -> endpoint prod

Hugging Face Dataset:
- model/iris_model.onnx
- test/test_data.csv
- logs/predicciones_dev.txt
- logs/predicciones_prod.txt
```

## Archivos principales

```text
app/main.py                  API FastAPI
app/model.py                 Carga el modelo ONNX y predice
app/logging_hf.py            Guarda predicciones en Hugging Face Dataset
tests/test_model.py          Pruebas del modelo
scripts/prepare_hf_assets.py Crea y sube modelo ONNX + datos de prueba al bucket
scripts/download_model.py    Descarga el modelo ONNX
Dockerfile                   Contenedor Docker
.github/workflows/mlops.yml  Pipeline CI/CD
```

## Paso 1: crear cuentas

Necesitas:

1. Una cuenta de GitHub.
2. Una cuenta de Hugging Face: <https://huggingface.co/>

## Paso 2: crear el token de Hugging Face

En Hugging Face:

1. Entra a tu perfil.
2. Ve a **Settings**.
3. Ve a **Access Tokens**.
4. Crea un token con permisos de escritura.
5. Copia el token. Se usara como `HF_TOKEN`.

No publiques ese token y no lo escribas dentro del codigo.

## Antes de seguir: palabras importantes

Si no sabes nada de esto, piensa en estas piezas asi:

- **Terminal**: ventana donde se escriben comandos. En Mac se llama **Terminal**.
- **Dataset de Hugging Face**: lo usaremos como una carpeta en internet. Ahi van el modelo, los datos de prueba y los logs.
- **Space de Hugging Face**: pagina web donde quedara corriendo la API.
- **Token**: una clave secreta para que Python, GitHub y la API puedan subir archivos a Hugging Face.
- **Secret de GitHub**: una variable privada que GitHub Actions puede usar sin mostrarla en publico.

En todos los ejemplos, cambia:

```text
tu_usuario
```

por tu usuario real de Hugging Face. Por ejemplo, si tu perfil es:

```text
https://huggingface.co/juanperez
```

entonces tu usuario es:

```text
juanperez
```

## Paso 3: crear el Dataset en Hugging Face

Este Dataset sera el "bucket" del taller. Es decir, sera la carpeta en la nube donde guardaremos archivos.

1. Abre esta pagina en el navegador:

```text
https://huggingface.co/new-dataset
```

2. En **Dataset name**, escribe:

```text
mlops-iris-assets
```

3. En **Owner**, deja tu usuario.

4. En **License**, puedes elegir:

```text
MIT
```

5. En **Visibility**, puedes escoger **Public** para que sea mas facil la sustentacion.

6. Haz clic en **Create dataset**.

Cuando termines, Hugging Face te dejara en una pagina parecida a esta:

```text
https://huggingface.co/datasets/tu_usuario/mlops-iris-assets
```

Ese nombre completo es importante:

```text
tu_usuario/mlops-iris-assets
```

Ejemplo si tu usuario fuera `juanperez`:

```text
juanperez/mlops-iris-assets
```

## Paso 4: abrir la Terminal en la carpeta del proyecto

En Mac:

1. Abre la aplicacion **Terminal**.
2. Copia este comando completo.
3. Pegalo en Terminal.
4. Presiona **Enter**.

```bash
cd "/Users/kamiro/Downloads/Notebooks_ICESI/Segundo_semestre/MLOPS/MLOps_Proyecto_Final"
```

Para comprobar que estas en la carpeta correcta, escribe:

```bash
pwd
```

Debe aparecer algo parecido a:

```text
/Users/kamiro/Downloads/Notebooks_ICESI/Segundo_semestre/MLOPS/MLOps_Proyecto_Final
```

## Paso 5: instalar lo necesario en tu computador

En la misma Terminal, ejecuta estos comandos uno por uno. Es decir: pegas el primero, presionas Enter, esperas que termine; luego el segundo, y asi.

Comando 1: crear un entorno de Python para este proyecto.

```bash
python3 -m venv .venv
```

Comando 2: activar ese entorno.

```bash
source .venv/bin/activate
```

Despues de activar, normalmente veras `(.venv)` al inicio de la linea de la Terminal. Eso esta bien.

Comando 3: instalar librerias.

```bash
pip install -r requirements.txt
```

Este comando puede tardar varios minutos. Si ves muchas lineas moviendose, no pasa nada: esta instalando.

## Paso 6: preparar el token y el nombre del Dataset

Necesitas decirle a la Terminal dos cosas:

- Tu token secreto de Hugging Face.
- El nombre del Dataset que creaste.

En la Terminal, ejecuta estos comandos, cambiando los valores.

Primero pega tu token. Donde dice `pega_aqui_tu_token`, reemplazalo por tu token real:

```bash
export HF_TOKEN="pega_aqui_tu_token"
```

Ejemplo inventado:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
```

Ahora escribe el nombre de tu Dataset. Cambia `tu_usuario` por tu usuario real:

```bash
export HF_ASSETS_REPO_ID="tu_usuario/mlops-iris-assets"
```

Ejemplo:

```bash
export HF_ASSETS_REPO_ID="juanperez/mlops-iris-assets"
```

Para verificar que quedo guardado en la Terminal, ejecuta:

```bash
echo $HF_ASSETS_REPO_ID
```

Debe mostrar algo como:

```text
juanperez/mlops-iris-assets
```

## Paso 7: crear y subir el modelo ONNX al Dataset

Ahora ejecuta:

```bash
python scripts/prepare_hf_assets.py
```

Que hace este comando:

- Crea un modelo ONNX sencillo.
- Crea un archivo CSV con datos de prueba.
- Sube esos archivos al Dataset de Hugging Face.
- Crea dos archivos TXT vacios para logs:

```text
logs/predicciones_dev.txt
logs/predicciones_prod.txt
```

Cuando termine, entra en el navegador a:

```text
https://huggingface.co/datasets/tu_usuario/mlops-iris-assets/tree/main
```

Debes ver carpetas llamadas:

```text
model
test
logs
```

Si no aparecen, algo fallo en este paso.

## Paso 8: probar que el modelo descarga y responde

En la Terminal, configura la URL del modelo. Cambia `tu_usuario` por tu usuario real:

```bash
export MODEL_URL="https://huggingface.co/datasets/tu_usuario/mlops-iris-assets/resolve/main/model/iris_model.onnx"
```

Ejemplo:

```bash
export MODEL_URL="https://huggingface.co/datasets/juanperez/mlops-iris-assets/resolve/main/model/iris_model.onnx"
```

Ahora descarga el modelo:

```bash
python scripts/download_model.py
```

Si todo va bien, debe decir algo parecido a:

```text
Modelo descargado en artifacts/model.onnx
```

## Paso 9: correr las pruebas del taller

Configura la URL de los datos de prueba. Cambia `tu_usuario`:

```bash
export TEST_DATA_URL="https://huggingface.co/datasets/tu_usuario/mlops-iris-assets/resolve/main/test/test_data.csv"
```

Ejemplo:

```bash
export TEST_DATA_URL="https://huggingface.co/datasets/juanperez/mlops-iris-assets/resolve/main/test/test_data.csv"
```

Ahora corre las pruebas. Usa `python -m pytest` y no solo `pytest`, porque asi nos aseguramos de usar el Python del entorno `.venv` y no el de Anaconda:

```bash
python -m pytest
```

Si todo esta bien, veras algo como:

```text
2 passed
```

Si vuelve a salir una ruta como `/opt/anaconda3/...`, significa que no esta activo el entorno `.venv`. En ese caso ejecuta otra vez:

```bash
source .venv/bin/activate
python -m pytest
```

Estas pruebas son las que pide el taller:

- El modelo responde con una entrada definida.
- La exactitud del modelo es mayor o igual a `0.90`.

## Paso 10: levantar la API en tu computador

Primero confirma que sigues en la carpeta del proyecto y que el entorno `.venv` esta activo:

```bash
cd "/Users/kamiro/Downloads/Notebooks_ICESI/Segundo_semestre/MLOPS/MLOps_Proyecto_Final"
source .venv/bin/activate
```

Ahora ejecuta:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 7860
```

La Terminal quedara ocupada mostrando mensajes. Eso es normal: significa que la API esta corriendo.

Antes de abrir el navegador, verifica que la Terminal muestre una linea parecida a esta:

```text
Uvicorn running on http://127.0.0.1:7860
```

No cierres esa Terminal. Si la cierras, la API se apaga.

Ahora abre esta pagina en el navegador:

```text
http://127.0.0.1:7860/docs
```

Si no abre, revisa la Terminal. Si ves un error rojo, la API no esta corriendo. En ese caso copia el error y revisa estos casos comunes:

- Si dice `No existe el modelo ONNX`, vuelve al paso 8.
- Si dice `Address already in use`, cambia el puerto: `python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 7861` y abre `http://127.0.0.1:7861/docs`.
- Si vuelve a aparecer `/opt/anaconda3/...`, activa otra vez el entorno con `source .venv/bin/activate`.

Para probar una prediccion:

1. Busca el bloque **POST /predict**.
2. Haz clic en el bloque.
3. Haz clic en **Try it out**.
4. Borra o reemplaza el JSON por este:

```json
   {
   "sepal_length": 5.1,
   "sepal_width": 3.5,
   "petal_length": 1.4,
   "petal_width": 0.2
   }
```

5. Haz clic en **Execute**.
6. Debes ver una respuesta con algo como:

```json
{
  "environment": "dev",
  "prediction": {
    "class_id": 0,
    "class_name": "setosa"
  }
}
```

Para detener la API local, vuelve a la Terminal y presiona:

```text
Control + C
```

## Paso 11: crear el Space dev en Hugging Face

Este sera el endpoint de desarrollo.

1. Abre:

```text
https://huggingface.co/new-space
```

2. En **Space name**, escribe:

```text
mlops-iris-dev
```

3. En **License**, puedes escoger:

```text
MIT
```

4. En **Select the Space SDK**, escoge:

```text
Docker
```

5. En **Hardware**, deja la opcion gratis de CPU.

6. En **Visibility**, puedes escoger **Public**.

7. Haz clic en **Create Space**.

Cuando termine, tu endpoint dev sera parecido a:

```text
https://tu_usuario-mlops-iris-dev.hf.space
```

## Paso 12: crear el Space prod en Hugging Face

Repite el paso anterior, pero ahora el nombre debe ser:

```text
mlops-iris-prod
```

El endpoint prod sera parecido a:

```text
https://tu_usuario-mlops-iris-prod.hf.space
```

## Paso 13: configurar variables del Space dev

Entra al Space dev:

```text
https://huggingface.co/spaces/tu_usuario/mlops-iris-dev
```

Luego:

1. Haz clic en **Settings**.
2. Busca la seccion **Variables and secrets**.
3. Agrega estas variables.

Variables normales:

```text
APP_ENV=dev
MODEL_URL=https://huggingface.co/datasets/tu_usuario/mlops-iris-assets/resolve/main/model/iris_model.onnx
HF_LOG_REPO_ID=tu_usuario/mlops-iris-assets
```

Secret:

```text
HF_TOKEN=tu_token_de_hugging_face
```

Si Hugging Face te deja escoger entre **Variable** y **Secret**, usa **Secret** para `HF_TOKEN`.

## Paso 14: configurar variables del Space prod

Entra al Space prod:

```text
https://huggingface.co/spaces/tu_usuario/mlops-iris-prod
```

Agrega estas variables:

```text
APP_ENV=prod
MODEL_URL=https://huggingface.co/datasets/tu_usuario/mlops-iris-assets/resolve/main/model/iris_model.onnx
HF_LOG_REPO_ID=tu_usuario/mlops-iris-assets
```

Y este secret:

```text
HF_TOKEN=tu_token_de_hugging_face
```

## Paso 15: crear el repositorio en GitHub

1. Abre:

```text
https://github.com/new
```

2. En **Repository name**, escribe algo como:

```text
mlops-proyecto-final
```

3. Puedes dejarlo publico.

4. No marques opciones como README, `.gitignore` o license, porque este proyecto ya tiene archivos.

5. Haz clic en **Create repository**.

GitHub te mostrara una URL parecida a:

```text
https://github.com/tu_usuario_github/mlops-proyecto-final.git
```

Copiala, la necesitaremos en la Terminal.

## Paso 16: subir este proyecto a GitHub

En la Terminal, asegurate de estar en la carpeta:

```bash
cd "/Users/kamiro/Downloads/Notebooks_ICESI/Segundo_semestre/MLOPS/MLOps_Proyecto_Final"
```

Ejecuta estos comandos uno por uno.

Comando 1:

```bash
git init
```

Comando 2:

```bash
git add .
```

Comando 3:

```bash
git commit -m "Proyecto MLOps ONNX con FastAPI"
```

Comando 4:

```bash
git branch -M prod
```

Comando 5: cambia la URL por la de tu repositorio de GitHub.

```bash
git remote add origin https://github.com/tu_usuario_github/mlops-proyecto-final.git
```

Comando 6:

```bash
git push -u origin prod
```

Comando 7:

```bash
git checkout -b dev
```

Comando 8:

```bash
git push -u origin dev
```

Con esto ya tienes las dos ramas que pide el taller:

```text
dev
prod
```

## Paso 17: crear secretos en GitHub

Entra a tu repositorio en GitHub.

Luego:

1. Haz clic en **Settings**.
2. En el menu izquierdo, haz clic en **Secrets and variables**.
3. Haz clic en **Actions**.
4. Haz clic en **New repository secret**.
5. Crea un secreto por cada fila de esta tabla.

```text
HF_TOKEN=tu_token_de_hugging_face
HF_USERNAME=tu_usuario_de_hugging_face
HF_SPACE_DEV=mlops-iris-dev
HF_SPACE_PROD=mlops-iris-prod
MODEL_URL=https://huggingface.co/datasets/tu_usuario_huggingface/mlops-iris-assets/resolve/main/model/iris_model.onnx
TEST_DATA_URL=https://huggingface.co/datasets/tu_usuario_huggingface/mlops-iris-assets/resolve/main/test/test_data.csv
```

Ejemplo de `HF_USERNAME`:

```text
juanperez
```

Ejemplo de `MODEL_URL`:

```text
https://huggingface.co/datasets/juanperez/mlops-iris-assets/resolve/main/model/iris_model.onnx
```

## Paso 18: ejecutar el pipeline

Cada vez que subas cambios a `dev` o `prod`, GitHub Actions ejecutara el pipeline.

Para verlo:

1. Entra al repositorio de GitHub.
2. Haz clic en **Actions**.
3. Haz clic en el flujo llamado **mlops-ci-cd**.
4. Abre la ejecucion mas reciente.

Debe aparecer:

```text
test
build/promote
```

Si ambos quedan en verde, el despliegue funciono.

## Paso 19: probar los endpoints finales

Endpoint dev:

```text
https://tu_usuario-mlops-iris-dev.hf.space/docs
```

Endpoint prod:

```text
https://tu_usuario-mlops-iris-prod.hf.space/docs
```

En cada uno puedes probar `POST /predict` igual que en local.

## Paso 20: revisar los TXT de predicciones

Despues de llamar `/predict`, entra al Dataset:

```text
https://huggingface.co/datasets/tu_usuario/mlops-iris-assets/tree/main/logs
```

Debes ver:

```text
predicciones_dev.txt
predicciones_prod.txt
```

Si hiciste una prediccion en dev, revisa `predicciones_dev.txt`.

Si hiciste una prediccion en prod, revisa `predicciones_prod.txt`.

## Como funciona el pipeline

Cada push a `dev` o `prod` ejecuta:

1. **test**
   - Descarga el modelo ONNX desde Hugging Face Dataset.
   - Descarga los datos de prueba desde Hugging Face Dataset.
   - Verifica que el modelo responda con una entrada definida.
   - Verifica que la exactitud sea mayor o igual a `0.90`.

2. **build/promote**
   - Construye el contenedor Docker.
   - Si la rama es `dev`, despliega al Space dev.
   - Si la rama es `prod`, despliega al Space prod.

## Endpoints de la API

```text
GET  /health
POST /predict
GET  /docs
```

## Logs de predicciones

Cada vez que alguien llama `/predict`, la app agrega una linea al archivo correcto:

```text
logs/predicciones_dev.txt
logs/predicciones_prod.txt
```

Estos archivos viven en el Dataset de Hugging Face configurado en `HF_LOG_REPO_ID`.

## Guion corto para sustentacion

1. Mostrar el repositorio con ramas `dev` y `prod`.
2. Mostrar `.github/workflows/mlops.yml`.
3. Explicar que el modelo ONNX no esta en GitHub, sino en Hugging Face Dataset.
4. Mostrar el pipeline corriendo en GitHub Actions.
5. Abrir el endpoint dev y hacer una prediccion en `/docs`.
6. Abrir el Dataset y mostrar que se actualizo `logs/predicciones_dev.txt`.
7. Repetir rapidamente con prod si el docente lo pide.
