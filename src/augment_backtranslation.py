#!/usr/bin/env python3
"""
Optimized back-translation data augmentation for SQuAD dataset.
Uses batching and faster generation for significant speedup.
"""

import json
import argparse
import random
import time
import sys
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import torch
from transformers import MarianMTModel, MarianTokenizer


class BackTranslator:
    """Handles back-translation using Helsinki-NLP MarianMT models."""
    
    def __init__(self, device="cpu", batch_size=32):
        self.device = device
        self.batch_size = batch_size
        self.model_cache = {}
        
    def translate_batch(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """
        Translate batch of texts (MUCH FASTER than one-by-one).
        
        Args:
            texts: List of texts to translate
            src_lang: Source language code
            tgt_lang: Target language code
        
        Returns:
            List of translated texts
        """
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        
        # Load model (or use cached)
        if model_name not in self.model_cache:
            print(f"Loading model: {model_name}", flush=True)
            t0 = time.time()
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name).to(self.device)
            model.eval()  # Set to eval mode
            self.model_cache[model_name] = (tokenizer, model)
            print(f"   Loaded in {time.time()-t0:.1f}s", flush=True)
        else:
            tokenizer, model = self.model_cache[model_name]
        
        # Process in batches
        all_translations = []
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        print(f"   Processing {len(texts):,} texts in {num_batches:,} batches (batch_size={self.batch_size})", flush=True)
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translations with optimized settings
            with torch.no_grad():
                translated = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=1,           # Greedy decoding (fastest)
                    do_sample=False,
                    early_stopping=True,
                    use_cache=True,        # Use KV cache
                    num_return_sequences=1
                )
            
            # Decode batch
            batch_translations = tokenizer.batch_decode(translated, skip_special_tokens=True)
            all_translations.extend(batch_translations)
            
            # Progress every 200 batches
            if (i // self.batch_size + 1) % 200 == 0:
                pct = (i + len(batch_texts)) / len(texts) * 100
                print(f"   Progress: {pct:.1f}% ({i+len(batch_texts):,}/{len(texts):,})", flush=True)
        
        return all_translations
    
    def back_translate_batch(
        self,
        texts: List[str],
        pivot_lang: str
    ) -> List[str]:
        """
        Back-translate batch of texts through intermediate language.
        
        Args:
            texts: List of English texts
            pivot_lang: Intermediate language code
        
        Returns:
            List of back-translated English texts
        """
        print(f"   EN → {pivot_lang.upper()}", flush=True)
        # English → Pivot language
        translated = self.translate_batch(texts, "en", pivot_lang)
        
        print(f"   {pivot_lang.upper()} → EN", flush=True)
        # Pivot language → English
        back_translated = self.translate_batch(translated, pivot_lang, "en")
        
        return back_translated


def find_answer_in_context(context: str, answer_text: str, original_start: int = None) -> Optional[Tuple[int, int]]:
    """Find answer text in context and return character positions."""
    # Try exact match first
    pos = context.find(answer_text)
    if pos != -1:
        return (pos, pos + len(answer_text))
    
    # Try case-insensitive match
    lower_context = context.lower()
    lower_answer = answer_text.lower()
    pos = lower_context.find(lower_answer)
    
    if pos != -1:
        return (pos, pos + len(answer_text))
    
    return None


def augment_squad_dataset(
    input_path: str,
    output_path: str,
    pivot_languages: List[str] = ["de", "fr", "es"],
    max_examples: int = 100,
    device: str = "cpu",
    batch_size: int = 32
):
    """
    Augment SQuAD dataset with back-translation (optimized with batching).
    
    Args:
        input_path: Path to input SQuAD JSON
        output_path: Path to save augmented JSON
        pivot_languages: List of intermediate languages
        max_examples: Maximum examples to augment (per language)
        device: Device for models ("cpu", "cuda", "mps")
        batch_size: Batch size for translation (larger = faster)
    """
    print("=" * 70, flush=True)
    print("BACK-TRANSLATION DATA AUGMENTATION", flush=True)
    print("=" * 70, flush=True)
    print(f"Input file: {input_path}", flush=True)
    print(f"Output file: {output_path}", flush=True)
    print(f"Pivot languages: {pivot_languages}", flush=True)
    print(f"Max examples per language: {max_examples:,}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    print("=" * 70, flush=True)
    sys.stdout.flush()
    
    # Load original data
    print(f"\nLoading data from {input_path}...", flush=True)
    t_load = time.time()
    
    try:
        with open(input_path, 'r') as f:
            squad_data = json.load(f)
        print(f"Loaded {len(squad_data['data']):,} articles in {time.time()-t_load:.1f}s", flush=True)
    except Exception as e:
        print(f"ERROR loading file: {e}", flush=True)
        raise
    
    # Initialize translator with batching
    print(f"Initializing translator...", flush=True)
    sys.stdout.flush()
    translator = BackTranslator(device=device, batch_size=batch_size)
    
    # Statistics
    stats = {
        "original_questions": 0,
        "augmented_questions": 0,
        "filtered_questions": 0,
        "by_language": {lang: {"success": 0, "filtered": 0} for lang in pivot_languages}
    }
    
    # Collect all QA pairs
    print(f"Collecting QA pairs...", flush=True)
    t_collect = time.time()
    all_qa_pairs = []
    for article in squad_data["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                all_qa_pairs.append({
                    "article_title": article["title"],
                    "context": para["context"],
                    "qa": qa
                })
                stats["original_questions"] += 1
    
    print(f"Collected {len(all_qa_pairs):,} QA pairs in {time.time()-t_collect:.1f}s", flush=True)
    
    # Sample subset for augmentation
    total_to_augment = min(max_examples, len(all_qa_pairs))
    print(f"Sampling {total_to_augment:,} examples...", flush=True)
    sampled_qa_pairs = random.sample(all_qa_pairs, total_to_augment)
    
    print(f"\nDataset Summary:", flush=True)
    print(f"   Total questions: {len(all_qa_pairs):,}", flush=True)
    print(f"   Augmenting: {total_to_augment:,} per language", flush=True)
    print(f"   Languages: {len(pivot_languages)}", flush=True)
    print(f"   Expected new examples: {total_to_augment * len(pivot_languages):,}", flush=True)
    sys.stdout.flush()
    
    # Create augmented data structure
    augmented_data = {
        "version": squad_data["version"],
        "data": []
    }
    
    # Process each language
    for lang_idx, lang in enumerate(pivot_languages):
        print(f"\n{'='*70}", flush=True)
        print(f"Language {lang_idx+1}/{len(pivot_languages)}: {lang.upper()}", flush=True)
        print(f"{'='*70}", flush=True)
        sys.stdout.flush()
        
        t_lang = time.time()
        
        # Extract questions and contexts
        print(f"Extracting texts...", flush=True)
        questions = [item["qa"]["question"] for item in sampled_qa_pairs]
        contexts = [item["context"] for item in sampled_qa_pairs]
        print(f"   Questions: {len(questions):,}", flush=True)
        print(f"   Contexts: {len(contexts):,}", flush=True)
        sys.stdout.flush()
        
        # Batch back-translate questions
        print(f"\nBack-translating questions through {lang.upper()}...", flush=True)
        sys.stdout.flush()
        bt_questions = translator.back_translate_batch(questions, lang)
        
        # Batch back-translate contexts
        print(f"\nBack-translating contexts through {lang.upper()}...", flush=True)
        sys.stdout.flush()
        bt_contexts = translator.back_translate_batch(contexts, lang)
        
        # Create augmented examples
        augmented_article = {
            "title": f"Augmented_{lang.upper()}",
            "paragraphs": []
        }
        
        print(f"\nAligning answers...", flush=True)
        sys.stdout.flush()
        t_align = time.time()
        
        for i, item in enumerate(tqdm(sampled_qa_pairs, desc=f"Aligning ({lang})")):
            try:
                qa = item["qa"]
                bt_question = bt_questions[i]
                bt_context = bt_contexts[i]
                
                # Get original answer
                original_answer = qa["answers"][0]
                answer_text = original_answer["text"]
                original_start = original_answer["answer_start"]
                
                # Find answer in back-translated context
                new_positions = find_answer_in_context(bt_context, answer_text, original_start)
                
                if new_positions is None:
                    stats["filtered_questions"] += 1
                    stats["by_language"][lang]["filtered"] += 1
                    continue
                
                new_start, new_end = new_positions
                extracted = bt_context[new_start:new_end]
                
                # Verify extraction
                if extracted.lower() != answer_text.lower():
                    stats["filtered_questions"] += 1
                    stats["by_language"][lang]["filtered"] += 1
                    continue
                
                # Create augmented QA
                augmented_qa = {
                    "id": f"{qa['id']}_aug_{lang}",
                    "question": bt_question,
                    "answers": [{
                        "text": extracted,
                        "answer_start": new_start
                    }],
                    "is_impossible": qa.get("is_impossible", False)
                }
                
                # Add to paragraphs
                augmented_article["paragraphs"].append({
                    "context": bt_context,
                    "qas": [augmented_qa]
                })
                
                stats["augmented_questions"] += 1
                stats["by_language"][lang]["success"] += 1
                
            except Exception as e:
                print(f"Warning: Error at index {i}: {e}", flush=True)
                stats["filtered_questions"] += 1
                stats["by_language"][lang]["filtered"] += 1
        
        augmented_data["data"].append(augmented_article)
        
        # Language summary
        elapsed = time.time() - t_lang
        align_time = time.time() - t_align
        success = stats['by_language'][lang]['success']
        filtered = stats['by_language'][lang]['filtered']
        
        print(f"\n{lang.upper()} Complete!", flush=True)
        print(f"   Total time: {elapsed:.1f}s", flush=True)
        print(f"   Alignment time: {align_time:.1f}s", flush=True)
        print(f"   Success: {success:,}", flush=True)
        print(f"   Filtered: {filtered:,}", flush=True)
        print(f"   Success rate: {success/(success+filtered)*100:.1f}%", flush=True)
        sys.stdout.flush()
    
    # Add original data
    print(f"\nAdding original {len(squad_data['data']):,} articles...", flush=True)
    augmented_data["data"].extend(squad_data["data"])
    
    # Save augmented dataset
    print(f"\nSaving to {output_path}...", flush=True)
    sys.stdout.flush()
    t_save = time.time()
    
    with open(output_path, 'w') as f:
        json.dump(augmented_data, f, indent=2)
    
    print(f"Saved in {time.time()-t_save:.1f}s", flush=True)
    
    # Print statistics
    total_qs = stats['original_questions'] + stats['augmented_questions']
    mult_factor = total_qs / stats['original_questions']
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║           AUGMENTATION COMPLETE                            ║
╠════════════════════════════════════════════════════════════╣
║ Original questions:      {stats['original_questions']:>6,}                         ║
║ Augmented questions:     {stats['augmented_questions']:>6,}                         ║
║ Filtered out:            {stats['filtered_questions']:>6,}                         ║
╠════════════════════════════════════════════════════════════╣
║ Total questions:         {total_qs:>6,}                         ║
║ Multiplication factor:   {mult_factor:>6.2f}x                        ║
╠════════════════════════════════════════════════════════════╣
║ By Language:                                               ║
""", flush=True)
    
    for lang in pivot_languages:
        success = stats['by_language'][lang]['success']
        filtered = stats['by_language'][lang]['filtered']
        total = success + filtered
        success_rate = (success / total * 100) if total > 0 else 0
        print(f"║   {lang.upper()}: {success:>5} success, {filtered:>5} filtered ({success_rate:>5.1f}% success) ║", flush=True)
    
    print(f"""╠════════════════════════════════════════════════════════════╣
║ Saved to: {output_path:<44} ║
╚════════════════════════════════════════════════════════════╝
""", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Augment SQuAD dataset with back-translation")
    parser.add_argument("--input", type=str, required=True, help="Input SQuAD JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output augmented JSON file")
    parser.add_argument("--languages", nargs="+", default=["de", "fr", "es"], help="Pivot languages")
    parser.add_argument("--max-examples", type=int, default=100, help="Max examples per language")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for translation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}", flush=True)
    print(f"Command-line arguments:", flush=True)
    print(f"  --input: {args.input}", flush=True)
    print(f"  --output: {args.output}", flush=True)
    print(f"  --languages: {args.languages}", flush=True)
    print(f"  --max-examples: {args.max_examples:,}", flush=True)
    print(f"  --device: {args.device}", flush=True)
    print(f"  --batch-size: {args.batch_size}", flush=True)
    print(f"  --seed: {args.seed}", flush=True)
    print(f"{'='*70}\n", flush=True)
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Run augmentation
    augment_squad_dataset(
        input_path=args.input,
        output_path=args.output,
        pivot_languages=args.languages,
        max_examples=args.max_examples,
        device=args.device,
        batch_size=args.batch_size
    )
    
    print(f"\nALL DONE!", flush=True)


if __name__ == "__main__":
    main()