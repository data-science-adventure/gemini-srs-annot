import json
import os
import logging
from typing import Any, List, Dict

# Configuración de logging para capturar los mensajes de depuración de _enrich
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
            
            # Búsqueda fallback en minúsculas si no se encuentra exacta
            if start_idx == -1:
                start_idx = text.lower().find(entity_text.lower(), start_from)

            if start_idx != -1:
                end_idx = start_idx + len(entity_text)
                search_pos[entity_text] = end_idx
                enriched_entities.append({**base, "start_offset": start_idx, "end_offset": end_idx})
                # Mapear texto a ID para las relaciones
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
    INPUT_JSON = "inferences/gemini3-cot.json"
    OUTPUT_JSONL = "gemini2-cot-with-annotations.jsonl"

    # Verificar si el archivo de entrada existe
    if not os.path.exists(INPUT_JSON):
        print(f"Error: No se encuentra el archivo {INPUT_JSON}")
        return

    # Leer el archivo JSON de entrada
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        try:
            srs_list = json.load(f)
        except json.JSONDecodeError:
            print("Error: El archivo de entrada no es un JSON válido.")
            return

    # Asegurar que el directorio de salida existe
    output_dir = os.path.dirname(OUTPUT_JSONL)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    processor = Processor()

    # Procesar y escribir en formato JSONL
    print(f"Procesando {len(srs_list)} documentos...")
    
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for doc in srs_list:
            # Según tu pseudocódigo: extraemos inferences y text
            # El método _enrich modifica el dict 'inferences' internamente
            inferences = doc.get("inferences", {})
            text = doc.get("text", "")

            # Enriquecemos la data
            predictions = processor._enrich(inferences, text)

            # 1. Creamos una copia para no modificar el objeto original doc
            output_record = doc.copy()
            # 2. Eliminamos 'inferences'
            output_record.pop("inferences", None) 
            # 3. Actualizamos con las nuevas entidades y relaciones
            output_record.update({
                "entities": predictions.get("entities", []),
                "relations": predictions.get("relations", []),
            })

            # Escribir línea en el JSONL
            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    print(f"✅ Proceso completado. Archivo generado: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()