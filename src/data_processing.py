import json, argparse
import os
import yaml
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

CFG_PATH = "config/deberta_base_pointer.yaml"
if not os.path.exists(CFG_PATH):
    raise FileNotFoundError(f"Config not found at {CFG_PATH}. Create it first.")

with open(CFG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

ENCODER_NAME = CFG.get("encoder_name", "microsoft/deberta-v3-base")
MAX_LENGTH = int(CFG.get("max_length", 384))
DOC_STRIDE = int(CFG.get("doc_stride", 128))

# Initialize tokenizer from config
tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME, use_fast=True)

#Read SQuAD-style JSON data
def read_data(file_path):
    with open(file_path, 'r') as file:
        file = json.load(file)
    storage = []
    for data in file["data"]:
        for para in data["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                qid = qa["id"]
                question = qa["question"]
                answers = qa["answers"]  
                #remove duplicate answers
                unique_answers = []
                seen_texts = set()
                for a in answers:
                    key = (a["text"], a["answer_start"])
                    if key not in seen_texts:
                        seen_texts.add(key)
                        unique_answers.append(a)
                storage.append({
                    "id": qid,
                    "context": context,
                    "question": question,
                    "answers": unique_answers
                })
    return storage

# Tokenize question and context into model input with sliding window (from config)
def tokenize(question, context):
    encoding = tokenizer(
        question,
        context,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        truncation=True,  # Changed from "only_second" to handle edge cases
        padding="max_length",
    )
    return encoding

#Find start and end token positions for answer text in tokenized input
def find_answer_positions(tokenized_input, answer_start, answer_end, window_index):
    offsets = list(tokenized_input["offset_mapping"][window_index])
    sequence_ids = tokenized_input.sequence_ids(window_index)
    for i in range(len(offsets)):
        if sequence_ids[i] != 1:
            offsets[i] = (None, None)
    start_token_index = 0
    end_token_index = 0
    has_start = any(s is not None and s <= answer_start < e for (s, e) in offsets)
    has_end = any(s is not None and s < answer_end <= e for (s, e) in offsets)
    if not (has_start and has_end):
        return None, None
    for idx, (start, end) in enumerate(offsets):
        if start is None and end is None:
            continue
        if start <= answer_start < end:
            start_token_index = idx
        if start < answer_end <= end:
            end_token_index = idx
            break
    return start_token_index, end_token_index

#Convert data into format suitable for BERT training
def Finalize_data(input):
    final_data = []
    skipped = 0
    
    for idx, data in enumerate(input):
        question = data["question"] 
        context = data["context"]
        
        try:
            tokenized_test_input = tokenize(question, context)
        except Exception as e:
            print(f"WARNING: Skipping example {idx} (ID: {data['id']}): Tokenization failed - {e}")
            skipped += 1
            continue
        
        has_tti = "token_type_ids" in tokenized_test_input
        answers = data["answers"]
        #Store all answer positions(for multiple answers) can be chosen randomly during training    
        id = data["id"]
        feats = len(tokenized_test_input["input_ids"])
        #Sliding window used for inputs longer than max length
        for i in range(feats):
            answers_processed = []
            for a in answers:
                answer_start = a["answer_start"]
                answer_end = answer_start + len(a["text"])
                answer_txt = a["text"]  
                start_token_index, end_token_index = find_answer_positions(
                    tokenized_test_input, answer_start, answer_end, i)
                if start_token_index is not None and end_token_index is not None:
                    answers_processed.append({
                            "start_token_index": start_token_index,
                            "end_token_index": end_token_index,
                            "answer_text": answer_txt
                    })
    
            item = {
                "id": id,
                "input_ids": tokenized_test_input["input_ids"][i],
                "attention_mask": tokenized_test_input["attention_mask"][i],
                "answers": answers_processed,
            }
            # Include token_type_ids only if tokenizer produced it (e.g., BERT). DeBERTa/Roberta omit it.
            if has_tti:
                item["token_type_ids"] = tokenized_test_input["token_type_ids"][i]
            final_data.append(item)
    
    if skipped > 0:
        print(f"\nWARNING: Skipped {skipped} examples due to tokenization errors")
        print(f"Successfully processed {len(input) - skipped} examples")

    return final_data

#Save processed data to disk using Hugging Face Datasets
def save_to_dataset(final_data, output_path):
    dataset = Dataset.from_list(final_data)
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.save_to_disk(output_path)

#Load processed data from disk using Hugging Face Datasets
def load_from_dataset(input_path):
    dataset_dict = DatasetDict.load_from_disk(input_path)
    return dataset_dict["train"]



def check_lengths(data):
    for item in data:
        input_ids = item["input_ids"]
        if len(input_ids) > MAX_LENGTH:
            print(f"Warning: Input ID length exceeds {MAX_LENGTH} tokens for ID {item['id']}")



#sample test run
def main():
    parser = argparse.ArgumentParser(description='Process SQuAD dataset for BERT training')
    parser.add_argument('--input_file', type=str, 
                       default='data/dev-v1.1.json',
                       help='Path to the SQuAD JSON file (default: data/train-v1.1.json)')
    parser.add_argument('--output_dir', type=str, default='data/processed_dataset',
                       help='Directory to save processed dataset (default: data/processed_dataset)')
    
    args = parser.parse_args()
    
    try:
        print("Reading dataset...")
        examples = read_data(args.input_file)
        print(f"Loaded {len(examples)} QA pairs.")

        print("Tokenizing and aligning answers...")
        final_data = Finalize_data(examples)

        check_lengths(final_data)

        print("\nSample output (first window feature):")
        for i in range(5):
            sample = final_data[i]
            print(f"ID: {sample['id']}")
            print(f"Question: {examples[i]['question']}")
            # Show up to first two answers that fit this window
            if sample["answers"]:
                for j, ans in enumerate(sample["answers"][:2]):
                    print(f"Answer {j+1} text: {ans['answer_text']}")
                    print(f"  Start token index: {ans['start_token_index']}, End token index: {ans['end_token_index']}")
            else:
                print("No gold answers lie inside this window (answers list is empty).")
        
        print(f"First 20 tokens: {tokenizer.convert_ids_to_tokens(sample['input_ids'][:20])}")
        
        print(f"\nSaving processed dataset to {args.output_dir}...")
        save_to_dataset(final_data, args.output_dir)
        print("Loading dataset from disk...")
        loaded_dataset = load_from_dataset(args.output_dir)
        
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found. Please check the file path.")
        print("Make sure the SQuAD dataset file is in the ../data/ folder")
        print("Usage example: python data_processing.py --input_file ../data/train-v1.1.json")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
