import json, csv, argparse
from transformers import BertTokenizerFast
from datasets import Dataset, DatasetDict

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

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

#Tokenize question and context into BERT input with sliding window
def tokenize(question, context):
    encoding = tokenizer(question, context, return_offsets_mapping=True,
                        return_overflowing_tokens=True, max_length=384, 
                        stride=128, truncation="only_second", padding="max_length")
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
    for data in input:
        question = data["question"] 
        context = data["context"]
        tokenized_test_input = tokenize(question, context)
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
    
            final_data.append({
                "id": id,
                "input_ids": tokenized_test_input["input_ids"][i],
                "attention_mask": tokenized_test_input["attention_mask"][i],
                "token_type_ids": tokenized_test_input["token_type_ids"][i],
                "answers": answers_processed
            })

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
        if len(input_ids) > 384:
            print(f"Warning: Input ID length exceeds 384 tokens for ID {item['id']}")



#sample test run
def main():
    file_path = "C:\\Users\\zonli\\Downloads\\train-v1.1.json"

    print("Reading dataset...")
    examples = read_data(file_path)
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
    
    dataset_path = "processed_dataset"
    print(f"\nSaving processed dataset to {dataset_path}...")
    save_to_dataset(final_data, dataset_path)
    print("Loading dataset from disk...")
    loaded_dataset = load_from_dataset(dataset_path)


if __name__ == "__main__":
    main()
