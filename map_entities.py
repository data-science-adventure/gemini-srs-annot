import json
import os
import glob
from typing import Dict, Any, Tuple, List

def load_inferences_by_id(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Carga un archivo JSONL y crea un índice por ID para búsqueda rápida.
    """
    index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Usamos el id como llave para acceso O(1)
            if "id" in data:
                index[str(data["id"])] = data
    return index

def main():
    SOURCE_FILE = "source/srs-test.jsonl"
    ENTITIES_DIR = "entities_extracted/"
    TARGET_DIR = "target/"

    # 1. Asegurar que el directorio de destino existe
    os.makedirs(TARGET_DIR, exist_ok=True)

    # 2. Obtener todos los archivos procesados en la etapa anterior
    extracted_files = glob.glob(os.path.join(ENTITIES_DIR, "*.jsonl"))

    if not extracted_files:
        print(f"No se encontraron archivos en {ENTITIES_DIR}")
        return

    print(f"Iniciando cruce de datos para {len(extracted_files)} archivos...")

    for current_file_path in extracted_files:
        file_name = os.path.basename(current_file_path)
        output_path = os.path.join(TARGET_DIR, file_name)
        
        print(f"Procesando: {file_name}")

        # 3. Cargar las inferencias de este modelo en memoria para búsqueda rápida
        # Esto equivale a tu función find_entities_and_relations_in_current_file
        inference_index = load_inferences_by_id(current_file_path)

        # 4. Leer el archivo fuente y generar el nuevo archivo target
        with open(SOURCE_FILE, 'r', encoding='utf-8') as f_source, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_source:
                if not line.strip():
                    continue
                
                target_row = json.loads(line)
                target_id = str(target_row.get("id"))

                # Buscar en el índice de inferencias
                if target_id in inference_index:
                    inference_data = inference_index[target_id]
                    target_row["entities"] = inference_data.get("entities", [])
                    target_row["relations"] = inference_data.get("relations", [])
                else:
                    # Si no hay inferencia para este ID, dejamos listas vacías
                    target_row["entities"] = []
                    target_row["relations"] = []

                # Guardar la fila enriquecida
                f_out.write(json.dumps(target_row, ensure_ascii=False) + "\n")

    print(f"\n✅ Proceso finalizado. Archivos generados en '{TARGET_DIR}'")

if __name__ == "__main__":
    main()