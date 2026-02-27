import json
from datasets import load_dataset

def main():
    print("Downloading SWE-bench full (test split, 2294 cases)...")
    dataset = load_dataset("princeton-nlp/SWE-bench", split="test")
    
    # Write to JSONL
    output_file = "benchmarks/data/swebench-full.jsonl"
    print(f"Saving to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
    print(f"Done! Downloaded {len(dataset)} cases.")

if __name__ == "__main__":
    main()
