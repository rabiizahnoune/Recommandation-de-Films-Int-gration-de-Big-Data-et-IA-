import os

# Définition de la structure du projet
project_structure = {
    "backend/app": [
        "__init__.py",
        "main.py",
        "models.py",
        "routes.py",
        "elasticsearch_client.py",
        "recommender.py",
        "config.py"
    ],
    "backend": [
        "requirements.txt",
        "Dockerfile",
        ".env"
    ],
    "spark_jobs": [
        "data_preprocessing.py",
        "train_model.py",
        "save_to_elasticsearch.py",
        "Dockerfile"
    ],
    "elasticsearch": [
        "elasticsearch.yml"
    ],
    "kibana": [
        "kibana.yml"
    ],
    "data/raw": [],
    "data/processed": [],
    ".": [  # Racine du projet
        "docker-compose.yml",
        ".gitignore",
        "README.md"
    ]
}

# Fonction pour créer les fichiers et dossiers
def create_project_structure(base_path="."):
    for folder, files in project_structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)  # Création du dossier
        
        for file in files:
            file_path = os.path.join(folder_path, file)
            with open(file_path, "w") as f:
                f.write("")  # Créer un fichier vide

if __name__ == "__main__":
    create_project_structure()
    print("✅ Structure du projet créée avec succès !")
