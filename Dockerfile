# Dockerfile para sistemas desktop
FROM python:3.9-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \\
    tesseract-ocr \\
    tesseract-ocr-spa \\
    libopencv-dev \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    libgstreamer1.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements/ requirements/

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements/desktop.txt

# Copiar código fuente
COPY . .

# Crear directorio para eventos
RUN mkdir -p /app/plate_events

# Exponer puerto si es necesario
EXPOSE 8080

# Comando por defecto
CMD ["python", "main.py"]