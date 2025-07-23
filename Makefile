.PHONY: install install-embedded install-desktop test run clean docker-build docker-run

# Instalación automática
install:
	python3 setup.py

# Instalación específica para embebidos
install-embedded:
	pip install -r requirements/embedded.txt

# Instalación específica para desktop
install-desktop:
	pip install -r requirements/desktop.txt

# Ejecutar pruebas
test:
	python -m pytest tests/ -v

# Ejecutar el sistema
run:
	python main.py

# Limpiar archivos temporales
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info/

# Construir imagen Docker
docker-build:
	docker build -t plate-recognition-system .

# Ejecutar con Docker
docker-run:
	docker-compose up -d

# Detener Docker
docker-stop:
	docker-compose down

# Ver logs
logs:
	docker-compose logs -f plate-recognition

# Backup de eventos
backup:
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz plate_events/ config/