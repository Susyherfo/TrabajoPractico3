# MLOps DVC Base

Repositorio base para el Taller de DVC - BCD4205

## Estructura

```
mlops-dvc-base/
├── src/
│   └── train.py
├── pyproject.toml
└── README.md
```

## Instrucciones

Sigue las instrucciones del taller para:

1. Inicializar DVC
2. Descargar y versionar el dataset
3. Entrenar y versionar el modelo
4. Configurar Google Drive como remote
5. Sincronizar con tu repositorio

## Instalación

```bash
uv sync
```

## Ejecución

```bash
uv run python src/train.py
```

# Comentarios importantes:

```bash
# Configurar AWS CLI
aws configure --profile ulead-mlops

# Agregar remote de S3
uv run dvc remote add  -d data-model s3://mlops-tp4-herrera-2026 

# Si usas un perfil específico de AWS
uv run dvc remote modify data-model profile ulead-mlops
```