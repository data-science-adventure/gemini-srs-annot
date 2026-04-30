import json
import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- Initialization ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Configuration
MODEL_ID = "gemini-3.1-flash-lite-preview"
INPUT_FILE = "srs-test.jsonl"
INSTRUCTION_FILE = "prompts/cot.txt"
OUTPUT_FILE = "gemini3-cot.jsonl"

def load_instruction(path):
    """Loads your Chain of Thought logic from cot.txt."""
    if not os.path.exists(path):
        print(f"⚠️ Error: {path} not found. Please create it.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def process_srs_requirements():
    if not API_KEY:
        print("❌ Error: Missing GEMINI_API_KEY in .env file.")
        return

    client = genai.Client(api_key=API_KEY)
    system_instruction = load_instruction(INSTRUCTION_FILE)
    
    if not system_instruction:
        return

    print(f"🚀 Starting extraction with {MODEL_ID}...")
    print(f"Reading: {INPUT_FILE} | Writing: {OUTPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        
        for i, line in enumerate(infile, 1):
            if not line.strip():
                continue
                
            try:
                # 1. Load the original JSONL row
                row = json.loads(line)
                req_text = row.get("text", "")
                
                if not req_text:
                    row["inferences"] = {"entities": [], "relations": []}
                    outfile.write(json.dumps(row) + "\n")
                    continue

                # 2. Use the Chat Method
                # We start a fresh session for each row to prevent history interference.
                chat = client.chats.create(
                    model=MODEL_ID,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json"
                    )
                )

                # 3. Send the message
                response = chat.send_message(req_text)

                # 4. Robust Parsing of the 'inferences'
                try:
                    # Using SDK auto-parsing for JSON mode
                    if hasattr(response, 'parsed') and response.parsed is not None:
                        row["inferences"] = response.parsed
                    else:
                        # Manual fallback cleaning
                        clean_content = response.text.strip()
                        if clean_content.startswith("```json"):
                            clean_content = clean_content[7:-3].strip()
                        row["inferences"] = json.loads(clean_content)
                except Exception as parse_err:
                    print(f"[{i}] JSON Parse Error: {parse_err}")
                    row["inferences"] = {"error": "parse_failure", "raw": response.text}

                # 5. Append to the original row and save
                outfile.write(json.dumps(row) + "\n")
                print(f"✅ [{i}] Processed ID: {row.get('id', 'N/A')}")

                # Rate limiting safety (0.5s pause)
                time.sleep(0.5)

            except Exception as e:
                print(f"❌ [{i}] Critical Error: {e}")

    print("\n✨ Processing complete! Check 'gemini3-cot.jsonl' for results.")

if __name__ == "__main__":
    process_srs_requirements()