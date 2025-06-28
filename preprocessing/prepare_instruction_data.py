import json

INPUT_PATH = "data/preprocessed/sample_businesses.jsonl"
OUTPUT_PATH = "data/preprocessed/mistral_instruction_data.jsonl"

# Format for Mistral instruction tuning
PROMPT_TEMPLATE = "<s>[INST] {instruction}\n{input} [/INST]"

with open(INPUT_PATH, "r", encoding="utf-8") as infile, open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
    for line in infile:
        example = json.loads(line)
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")
        prompt = PROMPT_TEMPLATE.format(instruction=instruction.strip(), input=input_text.strip())
        # Save as a JSONL with 'prompt' and 'response' fields
        json.dump({"prompt": prompt, "response": output_text.strip()}, outfile)
        outfile.write("\n")

print(f"Processed data saved to {OUTPUT_PATH}") 