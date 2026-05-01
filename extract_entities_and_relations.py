import json
import os
import logging
import glob
from typing import Any, List, Dict

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Processor:
    @staticmethod
    def _enrich(result: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Enriquece entidades con offsets de posición e IDs secuenciales,
        luego vincula las relaciones a los IDs de las entidades.
        """
        raw_entities: List[Dict[str, Any]] = result.get("entities", [])
        enriched_entities: List[Dict[str, Any]] = []
        search_pos: Dict[str, int] = {}
        text_to_id: Dict[str, int] = {}

        # 1. Procesar Entidades
        for entity_id, entity in enumerate(raw_entities, start=1):
            entity_text: str = entity.get("text", "")
            base = {**entity, "id": entity_id}

            if not entity_text:
                enriched_entities.append({**base, "start_offset": None, "end_offset": None})
                continue

            start_from = search_pos.get(entity_text, 0)
            start_idx = text.find(entity_text, start_from)
            
            if start_idx == -1:
                start_idx = text.lower().find(entity_text.lower(), start_from)

            if start_idx != -1:
                end_idx = start_idx + len(entity_text)
                search_pos[entity_text] = end_idx
                enriched_entities.append({**base, "start_offset": start_idx, "end_offset": end_idx})
                text_to_id.setdefault(entity_text, entity_id)
                text_to_id.setdefault(entity_text.lower(), entity_id)
            else:
                logger.debug(f"Entity '{entity_text}' not found in source text; offsets set to None.")
                enriched_entities.append({**base, "start_offset": None, "end_offset": None})

        # 2. Procesar Relaciones
        raw_relations: List[Dict[str, Any]] = result.get("relations", [])
        enriched_relations: List[Dict[str, Any]] = []

        for relation_id, relation in enumerate(raw_relations, start=1):
            subject = relation.get("subject", "")
            obj = relation.get("object", "")
            predicate = relation.get("predicate", "")
            
            from_id = text_to_id.get(subject) or text_to_id.get(subject.lower(), -1)
            to_id = text_to_id.get(obj) or text_to_id.get(obj.lower(), -1)
            
            enriched_relations.append({
                **relation,
                "id": relation_id,
                "from_id": from_id,
                "to_id": to_id,
                "type": predicate,
            })

        result["entities"] = enriched_entities
        result["relations"] = enriched_relations
        return result

def main():
    INPUT_DIR = "inferences"
    OUTPUT_DIR = "entities_extracted"

    # Crear directorio de salida si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Buscar todos los archivos .json en la carpeta de entrada
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))

    if not json_files:
        print(f"No se encontraron archivos JSON en la carpeta '{INPUT_DIR}'")
        return

    processor = Processor()
    print(f"Iniciando procesamiento de {len(json_files)} archivos...")

    for file_path in json_files:
        # Obtener el nombre base sin extensión
        base_name = os.path.basename(file_path)
        file_name_no_ext = os.path.splitext(base_name)[0]
        output_file_path = os.path.join(OUTPUT_DIR, f"{file_name_no_ext}.jsonl")

        print(f"Procesando: {base_name} -> {file_name_no_ext}.jsonl")

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                srs_list = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Error: {base_name} no es un JSON válido. Saltando...")
                continue

        with open(output_file_path, "w", encoding="utf-8") as fout:
            for doc in srs_list:
                inferences = doc.get("inferences", {})
                text = doc.get("text", "")

                predictions = processor._enrich(inferences, text)

                # Clonar doc y limpiar
                output_record = doc.copy()
                output_record.pop("inferences", None) 
                
                output_record.update({
                    "entities": predictions.get("entities", []),
                    "relations": predictions.get("relations", []),
                })

                fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    print(f"\n✅ Proceso completado. Archivos guardados en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()