# 🐍 OFIDIA - API de Clasificación de Serpientes

API REST desarrollada con FastAPI que clasifica imágenes de serpientes en tres categorías:
- **No venenosa**
- **Coral**
- **Víbora**

Utiliza un modelo de deep learning basado en ResNet50 entrenado con PyTorch.

---

## � Inicio Rápido

```bash
# 1. Construir la imagen
docker build -t ofidia-api .

# 2. Ejecutar el contenedor (requiere sudo para puerto 80)
sudo docker run -d -p 80:80 --name ofidia-container ofidia-api

# 3. Probar que funciona
curl http://localhost/
```

---

## �📋 Requisitos Previos

### Para ejecutar con Docker (Recomendado)
- Docker instalado en tu sistema Ubuntu
- Al menos 2GB de espacio en disco
- Conexión a internet para descargar la imagen base

### Para ejecutar sin Docker
- Python 3.11 o superior
- pip (gestor de paquetes de Python)

---

## 🚀 Instalación y Ejecución en Ubuntu

### Opción 1: Usando Docker (Recomendado)

#### 1. Instalar Docker en Ubuntu

Si no tienes Docker instalado, ejecuta:

```bash
# Actualizar el sistema
sudo apt update

# Instalar dependencias
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# Agregar la clave GPG oficial de Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Agregar el repositorio de Docker
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Instalar Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Verificar instalación
sudo docker --version

# (Opcional) Permitir ejecutar Docker sin sudo
sudo usermod -aG docker $USER
newgrp docker
```

#### 2. Clonar o descargar el proyecto

```bash
# Si tienes el proyecto en un repositorio
git clone <URL_DEL_REPOSITORIO>
cd OFIDIA_backsito_ML

# O si tienes los archivos localmente, navega al directorio
cd /ruta/al/proyecto/OFIDIA_backsito_ML
```

#### 3. Construir la imagen Docker

```bash
docker build -t ofidia-api .
```

Este proceso puede tardar varios minutos la primera vez, ya que descarga todas las dependencias.

#### 4. Ejecutar el contenedor

```bash
docker run -d -p 80:80 --name ofidia-container ofidia-api
```

**Parámetros:**
- `-d`: Ejecuta el contenedor en segundo plano (detached mode)
- `-p 80:80`: Mapea el puerto 80 del contenedor al puerto 80 de tu máquina
- `--name ofidia-container`: Asigna un nombre al contenedor

**Nota:** En Ubuntu, para usar el puerto 80 necesitas permisos de administrador. Puedes ejecutar:
```bash
sudo docker run -d -p 80:80 --name ofidia-container ofidia-api
```

#### 5. Verificar que está funcionando

```bash
# Ver logs del contenedor
docker logs ofidia-container

# Probar la API
curl http://localhost:8000/
```

#### 6. Comandos útiles de Docker

```bash
# Detener el contenedor
docker stop ofidia-container

# Iniciar el contenedor
docker start ofidia-container

# Reiniciar el contenedor
docker restart ofidia-container

# Ver contenedores en ejecución
docker ps

# Ver todos los contenedores (incluidos los detenidos)
docker ps -a

# Eliminar el contenedor
docker rm ofidia-container

# Eliminar la imagen
docker rmi ofidia-api

# Ver logs en tiempo real
docker logs -f ofidia-container
```

---

### Opción 2: Sin Docker (Instalación Local)

#### 1. Instalar Python 3.11

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip
```

#### 2. Crear entorno virtual

```bash
cd /ruta/al/proyecto/OFIDIA_backsito_ML
python3.11 -m venv venv
source venv/bin/activate
```

#### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Ejecutar la aplicación

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 🧪 Probar la API

### 1. Desde el navegador

Abre tu navegador y visita:
- http://localhost/ - Información general de la API
- http://localhost/docs - Documentación interactiva (Swagger UI)

### 2. Desde la línea de comandos con curl

```bash
# Verificar que la API está funcionando
curl http://localhost/

# Enviar una imagen para clasificación
curl -X POST "http://localhost/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/ruta/a/tu/imagen.jpg"
```

### 3. Desde Python

```python
import requests

url = "http://localhost/predict"
files = {"file": open("imagen_serpiente.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

---

## 📁 Estructura del Proyecto

```
OFIDIA_backsito_ML/
├── app.py                              # API FastAPI principal
├── modelo_serpientes.pth               # Modelo entrenado (PyTorch)
├── modelo_serpientes_completo.pth      # Modelo completo (backup)
├── modelo_serpientes.onnx              # Modelo en formato ONNX
├── requirements.txt                    # Dependencias de Python
├── Dockerfile                          # Configuración de Docker
└── README.md                           # Este archivo
```

---

## 🐛 Solución de Problemas

### El modelo no se carga correctamente

Verifica que el archivo `modelo_serpientes.pth` esté en el directorio raíz del proyecto.

### Error de puerto en uso

Si el puerto 80 ya está en uso, puedes usar otro puerto:

```bash
# Con Docker - mapear a puerto 8080 en tu máquina
docker run -d -p 8080:80 --name ofidia-container ofidia-api

# Sin Docker
uvicorn app:app --host 0.0.0.0 --port 8080
```

### Error de memoria con Docker

Si tienes problemas de memoria, aumenta los recursos asignados a Docker en la configuración.

---

## 📝 Respuesta de la API

### Endpoint: POST /predict

**Respuesta exitosa:**
```json
{
  "filename": "serpiente.jpg",
  "predicted_class": "Coral",
  "confidence": 95.67
}
```

**Clases posibles:**
- `No venenosa`
- `Coral`
- `Víbora`

---

## 👨‍💻 Desarrollo

### Reconstruir después de cambios

Si modificas el código:

```bash
# Detener y eliminar el contenedor actual
docker stop ofidia-container
docker rm ofidia-container

# Reconstruir la imagen
docker build -t ofidia-api .

# Ejecutar nuevamente
sudo docker run -d -p 80:80 --name ofidia-container ofidia-api
```

---

## 📄 Licencia

[Especifica tu licencia aquí]

---

## 👥 Contacto

[Tu información de contacto]
