# Usa la imagen base oficial de Python
FROM python:3.9-slim

# Establece un directorio de trabajo
WORKDIR /app

# Copia los requisitos al contenedor
COPY requirements.txt /app/requirements.txt

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos del proyecto
COPY . /app

# Declarar un volumen para persistencia
VOLUME ["/app/data"]

# Expone el puerto 8007 para la aplicación
EXPOSE 8007

# Comando para ejecutar la aplicación
CMD ["python", "main.py"]



