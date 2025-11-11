#!/usr/bin/env python3
"""
Back-translation data augmentation for SQuAD dataset.
Augments questions and contexts using MarianMT models.
"""

import json
import argparse
import random
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import torch
from transformers import MarianMTModel, MarianTokenizer


class BackTranslator:
    """Handles back-translation using Helsinki-NLP MarianMT models."""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.model_cache = {}  # Cache loaded models
        
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            src_lang: Source language code (e.g., "en")
            tgt_lang: Target language code (e.g., "de")
        
        Returns:
            Translated text
        """
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        
        # Load model (or use cached)
        if model_name not in self.model_cache:
            print(f"Loading model: {model_name}")
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name).to(self.device)
            self.model_cache[model_name] = (tokenizer, model)
        else:
            tokenizer, model = self.model_cache[model_name]
        
        # Translate
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Use faster generation settings
            translated = model.generate(**inputs, max_length=512, num_beams=1, do_sample=False)
        
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        return result
    
    def back_translate(self, text: str, pivot_lang: str) -> str:
        """
        Back-translate text through an intermediate language.
        
        Args:
            text: Original English text
            pivot_lang: Intermediate language code (e.g., "de", "fr", "es")
        
        Returns:
            Back-translated English text
        """
        # English → Pivot language
        translated = self.translate(text, "en", pivot_lang)
        
        # Pivot language → English
        back_translated = self.translate(translated, pivot_lang, "en")
        
        return back_translated


def find_answer_in_context(context: str, answer_text: str, original_start: int = None) -> Optional[Tuple[int, int]]:
    """
    Find answer text in context and return character positions.
    
    Args:
        context: Context text to search in
        answer_text: Answer text to find
        original_start: Original start position (used to prefer closest match)
    
    Returns:
        (start_pos, end_pos) or None if not found
    """
    # Try exact match first
    pos = context.find(answer_text)
    if pos != -1:
        return (pos, pos + len(answer_text))
    
    # Try case-insensitive match
    lower_context = context.lower()
    lower_answer = answer_text.lower()
    pos = lower_context.find(lower_answer)
    
    if pos != -1:
        # Extract actual text from context (preserves casing)
        actual_text = context[pos:pos + len(answer_text)]
        return (pos, pos + len(answer_text))
    
    # Try finding all occurrences if multiple exist
    all_positions = []
    start = 0
    while True:
        pos = context.find(answer_text, start)
        if pos == -1:
            break
        all_positions.append((pos, pos + len(answer_text)))
        start = pos + 1
    
    if all_positions:
        # If original position provided, choose closest
        if original_start is not None:
            closest = min(all_positions, key=lambda p: abs(p[0] - original_start))
            return closest
        return all_positions[0]
    
    return None


def augment_qa_example(
    qa: Dict,
    context: str,
    translator: BackTranslator,
    pivot_lang: str
) -> Optional[Dict]:
    """
    Augment a single QA example using back-translation.
    
    Args:
        qa: QA dict with "id", "question", "answers"
        context: Paragraph context
        translator: BackTranslator instance
        pivot_lang: Intermediate language for back-translation
    
    Returns:
        Augmented QA dict or None if answer not found
    """
    try:
        # Back-translate question and context
        bt_question = translator.back_translate(qa["question"], pivot_lang)
        bt_context = translator.back_translate(context, pivot_lang)
        
        # Process each answer
        original_answer = qa["answers"][0]  # Use first answer
        answer_text = original_answer["text"]
        original_start = original_answer["answer_start"]
        
        # Find answer in back-translated context
        new_positions = find_answer_in_context(bt_context, answer_text, original_start)
        
        if new_positions is None:
            # Answer not found - skip this augmentation
            return None
        
        new_start, new_end = new_positions
        
        # Verify extraction
        extracted = bt_context[new_start:new_end]
        if extracted.lower() != answer_text.lower():
            # Mismatch - skip
            return None
        
        # Create augmented QA
        augmented_qa = {
            "id": f"{qa['id']}_aug_{pivot_lang}",
            "question": bt_question,
            "answers": [{
                "text": extracted,  # Use extracted text (preserves casing)
                "answer_start": new_start
            }],
            "is_impossible": qa.get("is_impossible", False)
        }
        
        return augmented_qa, bt_context
        
    except Exception as e:
        print(f"⚠️  Error augmenting {qa['id']}: {e}")
        return None


def augment_squad_dataset(
    input_path: str,
    output_path: str,
    pivot_languages: List[str] = ["de", "fr", "es"],
    max_examples: int = 100,
    device: str = "cpu"
):
    """
    Augment SQuAD dataset with back-translation.
    
    Args:
        input_path: Path to input SQuAD JSON
        output_path: Path to save augmented JSON
        pivot_languages: List of intermediate languages
        max_examples: Maximum examples to augment (per language)
        device: Device for models ("cpu", "cuda", "mps")
    """
    print(f"🚀 Starting back-translation augmentation")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Languages: {pivot_languages}")
    print(f"   Max examples per language: {max_examples}")
    print(f"   Device: {device}")
    
    # Load original data
    with open(input_path, 'r') as f:
        squad_data = json.load(f)
    
    # Initialize translator
    translator = BackTranslator(device=device)
    
    # Statistics
    stats = {
        "original_questions": 0,
        "augmented_questions": 0,
        "filtered_questions": 0,
        "by_language": {lang: {"success": 0, "filtered": 0} for lang in pivot_languages}
    }
    
    # Collect all QA pairs first
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
    
    # Sample subset for augmentation
    total_to_augment = min(max_examples, len(all_qa_pairs))
    sampled_qa_pairs = random.sample(all_qa_pairs, total_to_augment)
    
    print(f"\n📊 Total questions in dataset: {len(all_qa_pairs)}")
    print(f"   Augmenting {total_to_augment} questions per language")
    
    # Create augmented data structure
    augmented_data = {
        "version": squad_data["version"],
        "data": []
    }
    
    # Group by article and paragraph for output structure
    # For simplicity, create a new article for augmented data
    for lang in pivot_languages:
        print(f"\n🔄 Processing language: {lang.upper()}")
        
        augmented_article = {
            "title": f"Augmented_{lang.upper()}",
            "paragraphs": []
        }
        
        for item in tqdm(sampled_qa_pairs, desc=f"Augmenting ({lang})"):
            result = augment_qa_example(
                qa=item["qa"],
                context=item["context"],
                translator=translator,
                pivot_lang=lang
            )
            
            if result is not None:
                augmented_qa, bt_context = result
                
                # Add to paragraph (create new paragraph for each QA for simplicity)
                augmented_article["paragraphs"].append({
                    "context": bt_context,
                    "qas": [augmented_qa]
                })
                
                stats["augmented_questions"] += 1
                stats["by_language"][lang]["success"] += 1
            else:
                stats["filtered_questions"] += 1
                stats["by_language"][lang]["filtered"] += 1
        
        augmented_data["data"].append(augmented_article)
    
    # Add original data
    augmented_data["data"].extend(squad_data["data"])
    
    # Save augmented dataset
    with open(output_path, 'w') as f:
        json.dump(augmented_data, f, indent=2)
    
    # Print final statistics
    print(f"""
╔════════════════════════════════════════════════════════════╗
║           AUGMENTATION COMPLETE                            ║
╠════════════════════════════════════════════════════════════╣
║ Original questions:      {stats['original_questions']:>6,}                         ║
║ Augmented questions:     {stats['augmented_questions']:>6,}                         ║
║ Filtered out:            {stats['filtered_questions']:>6,}                         ║
╠════════════════════════════════════════════════════════════╣
║ Total questions:         {stats['original_questions'] + stats['augmented_questions']:>6,}                         ║
║ Multiplication factor:   {(stats['original_questions'] + stats['augmented_questions']) / stats['original_questions']:>6.2f}x                        ║
╠════════════════════════════════════════════════════════════╣
║ By Language:                                               ║
""")
    
    for lang in pivot_languages:
        success = stats['by_language'][lang]['success']
        filtered = stats['by_language'][lang]['filtered']
        total = success + filtered
        success_rate = (success / total * 100) if total > 0 else 0
        print(f"║   {lang.upper()}: {success:>4} success, {filtered:>4} filtered ({success_rate:>5.1f}% success) ║")
    
    print(f"""╠════════════════════════════════════════════════════════════╣
║ Saved to: {output_path:<44} ║
╚════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(description="Augment SQuAD dataset with back-translation")
    parser.add_argument(
        "--input",
        type=str,
        default="data/train-subset.json",
        help="Input SQuAD JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/train-subset-augmented.json",
        help="Output augmented JSON file"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["de", "fr", "es"],
        help="Pivot languages for back-translation (e.g., de fr es)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum examples to augment per language"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run models on"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Run augmentation
    augment_squad_dataset(
        input_path=args.input,
        output_path=args.output,
        pivot_languages=args.languages,
        max_examples=args.max_examples,
        device=args.device
    )
    
    print("\n✅ Done! You can now preprocess the augmented dataset:")
    print(f"   python src/data_processing.py --input_file {args.output} --output_dir data/processed_augmented")


if __name__ == "__main__":
    main()
