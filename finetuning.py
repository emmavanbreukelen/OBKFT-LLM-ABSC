import json
import torch
from unsloth import FastLanguageModel
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import huggingface_hub
from huggingface_hub import login
from transformers import TrainingArguments, DataCollatorForLanguageModeling, AutoModel, AutoTokenizer
from trl import SFTTrainer
from peft import LoraConfig
import wandb
import uuid
from datasets import Dataset
from typing import Dict, List, Any
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Log in to HuggingFace and Wandb with you pass keys
key_hf = os.getenv("HUGGINGFACE_KEY")
try:
    login(token=key_hf)
    print("Hugging Face login successful.")
except Exception as e:
    print(f"Hugging Face login failed: {e}")
    exit()

wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(project="NAME_YOUR_PROJECT", name=f"run-{uuid.uuid4().hex[:8]}")

# Loads the SemEval datasets
# This works only for JSON files (so for the files after all the pre-processing steps)
# This code is adjusted to iterate through data in the format of data after pre-processing
# This also skips instances with no opinion
def load_dataset(file_path: str) -> List[Dict[str, str]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{file_path}'")
        exit()
    processed_sentences = []
    reviews_list = data.get('Reviews', {}).get('Review', [])
    if not reviews_list:
        top_level_sentences = data.get('sentences', {}).get('sentence', [])
        if top_level_sentences:
            for sentence_obj in top_level_sentences:
                reviews_list.append({'sentences': {'sentence': [sentence_obj]}})
    if not isinstance(reviews_list, list):
        reviews_list = [reviews_list]
    for review in reviews_list:
        if not review:
            continue
        sentences_dict = review.get('sentences')
        if not sentences_dict:
            continue
        sentences_data = sentences_dict.get('sentence', [])
        if not isinstance(sentences_data, list):
            sentences_data = [sentences_data]
        for sentence in sentences_data:
            if not sentence:
                continue
            text, opinions = sentence.get('text'), sentence.get('Opinions', {}).get('Opinion', [])
            if not opinions:
                continue
            if isinstance(opinions, dict):
                opinions = [opinions]
            for opinion in opinions:
                target, polarity = opinion.get('@target'), opinion.get('@polarity')
                if text and target and polarity and polarity != 'conflict':
                    processed_sentences.append({
                        'text': text,
                        'target': target,
                        'polarity': polarity.lower()
                    })
    if not processed_sentences:
        print(f"Warning: No valid data points extracted from '{file_path}'.")
    return processed_sentences

# Loads the verbalized domain ontology
def load_ontology(file_path: str) -> Dataset:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            ontology = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ontology file not found at '{file_path}'")
        exit()
    normalized_ontology = [
        entry.rsplit(" is ", 1)[0] + " is " + entry.rsplit(" is ", 1)[1].lower()
        for entry in ontology
    ]
    data = {"text": normalized_ontology}
    return Dataset.from_dict(data)

# Create zero shot prompt for the LLMs to perform ABSC
def create_strong_zero_shot_prompt(text: str, target: str) -> str:
    return f"""Your task is to classify the sentiment of a target aspect within a sentence.
You must respond with only one of the following words: positive, negative, or neutral.

Sentence: {text}
Target Aspect: {target}
Sentiment:"""

# Create few shot prompt for the LLMs to perform ABSC
# This automatically includes three examples from the training data that best match each test instance
def create_few_shot_prompt(text: str, target: str, few_shot_examples: List[Dict[str, str]]) -> str:
    prompt = """Your task is to classify the sentiment of a target aspect within a sentence.
You must respond with only one of the following words: positive, negative, or neutral.

Here are some examples:
"""
    for example in few_shot_examples:
        prompt += f"""Sentence: {example['text']}
Target Aspect: {example['target']}
Sentiment: {example['polarity']}
---
"""
    prompt += f"""Sentence: {text}
Target Aspect: {target}
Sentiment:"""
    return prompt

def create_training_prompt(text: str) -> str:
    return f"{text}\n"

# Custom data dollator
# This makes sure that only the last word in each sentence in the ontology will be masked
# Because you want to teach the model to perform ABSC, which also classifies only sentiment
class CustomDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(features)
        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']

        batch_size, seq_length = labels.shape
        new_labels = torch.full_like(labels, -100)
        for i in range(batch_size):
            valid_indices = (input_ids[i] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            if len(valid_indices) > 0:
                last_valid_idx = valid_indices[-1]
                new_labels[i, last_valid_idx] = labels[i, last_valid_idx]

        batch['labels'] = new_labels
        return batch

# SimCSE-based example selection
# Computes sentence embeddings for the training data using SimCSE
def precompute_train_embeddings(train_corpus, model, tokenizer, device):
    model.eval()
    embeddings = []
    batch_size = 32
    for i in range(0, len(train_corpus), batch_size):
        batch = train_corpus[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings.append(batch_embeddings.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

# This part then selects the top k (k=3) from the training data that best match a test instance
# This is done with cosine similarity over the SimCSE embeddings from the previous function
def select_few_shot_examples(item, train_dataset, simcse_model, simcse_tokenizer, device, num_shots=3):
    query_sentence = item['text']
    train_corpus = [example['text'] for example in train_dataset]
    train_embeddings = precompute_train_embeddings(train_corpus, simcse_model, simcse_tokenizer, device)
    query_inputs = simcse_tokenizer([query_sentence], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        query_embedding = simcse_model(**query_inputs).last_hidden_state[:, 0, :].cpu().numpy()
    similarities = cosine_similarity(query_embedding, train_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:num_shots]
    selected_examples = [train_dataset[idx] for idx in top_indices]
    return selected_examples

# Flexible logit prediction function
# This computes the likelihood of each sentiment and chooses the one with the highest likelihood
def get_flexible_logit_prediction(model, tokenizer, prompt: str, sentiment_token_map: Dict[str, List[int]], device: torch.device) -> str:
    inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[:, -1, :]
    final_logits = []
    for sentiment in ["positive", "negative", "neutral"]:
        sentiment_specific_logits = next_token_logits[:, sentiment_token_map[sentiment]]
        max_logit, _ = torch.max(sentiment_specific_logits, dim=1)
        final_logits.append(max_logit)
    final_logits = torch.stack(final_logits, dim=1)
    probabilities = torch.softmax(final_logits, dim=1)
    prediction_index = torch.argmax(probabilities, dim=1).item()
    sentiment_labels = ["positive", "negative", "neutral"]
    return sentiment_labels[prediction_index]

# Training function to fine-tune the LLMs
def train_model(model, tokenizer, ontology_dataset: Dataset):
    training_args = TrainingArguments(
        output_dir="./llama3-finetuned",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb",
        push_to_hub=False,
        gradient_checkpointing=True,
    )

    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ontology_dataset,
        data_collator=data_collator,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
    )

    trainer.train()

    model.save_pretrained("SAVE_FINETUNED_MODEL")
    tokenizer.save_pretrained("SAVE_FINETUNED_MODEL")
    wandb.save("SAVE_FINETUNED_MODEL*")

    return model, tokenizer

# Main evaluation function
def main():

    # Fill in the verbalized and pre-processed data paths
    ontology_file = "VERBALIZED_ONTOLOGY_PATH"
    test_file = "SEMEVAL_TEST_PATH"
    train_file = "SEMEVAL_TRAINING_PATH"

    # Set device to determine whether a GPU is available (because unsloth only work with GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Loads datasets
    ontology_dataset = load_ontology(ontology_file)
    test_dataset = load_dataset(test_file)
    train_dataset = load_dataset(train_file)
    if not test_dataset or not train_dataset:
        print("No test or train data available. Exiting.")
        return

    # Loads SimCSE model and tokenizer
    print("Loading SimCSE model...")
    simcse_model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
    simcse_tokenizer = AutoTokenizer.from_pretrained(simcse_model_name)
    simcse_model = AutoModel.from_pretrained(simcse_model_name).to(device)
    print("SimCSE model loaded successfully.")

    # Load model (quantized 4-bit model)
    # Fill in huggingface path of the LLM
    model_name = "LLM_HUGGINGFACE_PATH"
    print(f"Loading base model: {model_name}...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42
    )
    print("Base model loaded successfully.")

    # Sentiment token map which ensures that different spaces or capital letters do not result in a wrong prediction when it actually is correct
    # This ensures that any structural irregularities in the verbalized ontology do not affect the predictions of the LLMs for sentiment value
    sentiment_token_map = {
        "positive": (
            tokenizer.encode("positive", add_special_tokens=False) +
            tokenizer.encode(" positive", add_special_tokens=False) +
            tokenizer.encode("Positive", add_special_tokens=False) +
            tokenizer.encode(" Positive", add_special_tokens=False)
        ),
        "negative": (
            tokenizer.encode("negative", add_special_tokens=False) +
            tokenizer.encode(" negative", add_special_tokens=False) +
            tokenizer.encode("Negative", add_special_tokens=False) +
            tokenizer.encode(" Negative", add_special_tokens=False)
        ),
        "neutral": (
            tokenizer.encode("neutral", add_special_tokens=False) +
            tokenizer.encode(" neutral", add_special_tokens=False) +
            tokenizer.encode("Neutral", add_special_tokens=False) +
            tokenizer.encode(" Neutral", add_special_tokens=False)
        ),
    }

    print("\nToken IDs for sentiment labels (with variations):")
    print(sentiment_token_map)
    print("-" * 30)

    # Evaluate zero-shot performance before fine-tuning
    print("Evaluating zero-shot performance...")
    true_labels_zero, pred_labels_zero = [], []
    for item in tqdm(test_dataset, desc="Zero-Shot Classifying"):
        text, target = item['text'], item['target']
        true_polarity = item['polarity']
        prompt = create_strong_zero_shot_prompt(text, target)
        predicted_polarity = get_flexible_logit_prediction(
            model, tokenizer, prompt, sentiment_token_map, device
        )
        true_labels_zero.append(true_polarity)
        pred_labels_zero.append(predicted_polarity)

    class_labels = ['positive', 'negative', 'neutral']
    accuracy_zero = accuracy_score(true_labels_zero, pred_labels_zero)
    macro_f1_zero = f1_score(true_labels_zero, pred_labels_zero, average='macro', labels=class_labels, zero_division=0)
    weighted_f1_zero = f1_score(true_labels_zero, pred_labels_zero, average='weighted', labels=class_labels, zero_division=0)

    print("\n--- Zero-Shot Evaluation Results ---")
    print(f"Total samples evaluated: {len(true_labels_zero)}")
    print(f"Accuracy: {accuracy_zero:.4f}")
    print(f"Macro F1 Score: {macro_f1_zero:.4f}")
    print(f"Weighted F1 Score: {weighted_f1_zero:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels_zero, pred_labels_zero, labels=class_labels, digits=4, zero_division=0))

    wandb.log({
        "zero_shot_accuracy": accuracy_zero,
        "zero_shot_macro_f1": macro_f1_zero,
        "zero_shot_weighted_f1": weighted_f1_zero,
        "zero_shot_classification_report": classification_report(true_labels_zero, pred_labels_zero, labels=class_labels, digits=4, zero_division=0, output_dict=True)
    })

    # Evaluate three-shot performance before fine-tuning
    print("Evaluating three-shot performance...")
    true_labels_three, pred_labels_three = [], []
    for item in tqdm(test_dataset, desc="Three-Shot Classifying"):
        text, target = item['text'], item['target']
        true_polarity = item['polarity']
        few_shot_examples = select_few_shot_examples(item, train_dataset, simcse_model, simcse_tokenizer, device, num_shots=3)
        prompt = create_few_shot_prompt(text, target, few_shot_examples)
        predicted_polarity = get_flexible_logit_prediction(
            model, tokenizer, prompt, sentiment_token_map, device
        )
        true_labels_three.append(true_polarity)
        pred_labels_three.append(predicted_polarity)

    accuracy_three = accuracy_score(true_labels_three, pred_labels_three)
    macro_f1_three = f1_score(true_labels_three, pred_labels_three, average='macro', labels=class_labels, zero_division=0)
    weighted_f1_three = f1_score(true_labels_three, pred_labels_three, average='weighted', labels=class_labels, zero_division=0)

    print("\n--- Three-Shot Evaluation Results ---")
    print(f"Total samples evaluated: {len(true_labels_three)}")
    print(f"Accuracy: {accuracy_three:.4f}")
    print(f"Macro F1 Score: {macro_f1_three:.4f}")
    print(f"Weighted F1 Score: {weighted_f1_three:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels_three, pred_labels_three, labels=class_labels, digits=4, zero_division=0))

    wandb.log({
        "three_shot_accuracy": accuracy_three,
        "three_shot_macro_f1": macro_f1_three,
        "three_shot_weighted_f1": weighted_f1_three,
        "three_shot_classification_report": classification_report(true_labels_three, pred_labels_three, labels=class_labels, digits=4, zero_division=0, output_dict=True)
    })

    # This initiates the fine-tuning of the model
    print("Starting fine-tuning...")
    model, tokenizer = train_model(model, tokenizer, ontology_dataset)
    print("Fine-tuning completed.")

    model.eval()

    # Evaluate zero-shot performance after fine-tuning
    print("Evaluating zero-shot performance after fine-tuning...")
    true_labels_zero_ft, pred_labels_zero_ft = [], []
    for item in tqdm(test_dataset, desc="Zero-Shot Classifying (Fine-Tuned)"):
        text, target = item['text'], item['target']
        true_polarity = item['polarity']
        prompt = create_strong_zero_shot_prompt(text, target)
        predicted_polarity = get_flexible_logit_prediction(
            model, tokenizer, prompt, sentiment_token_map, device
        )
        true_labels_zero_ft.append(true_polarity)
        pred_labels_zero_ft.append(predicted_polarity)

    accuracy_zero_ft = accuracy_score(true_labels_zero_ft, pred_labels_zero_ft)
    macro_f1_zero_ft = f1_score(true_labels_zero_ft, pred_labels_zero_ft, average='macro', labels=class_labels, zero_division=0)
    weighted_f1_zero_ft = f1_score(true_labels_zero_ft, pred_labels_zero_ft, average='weighted', labels=class_labels, zero_division=0)

    print("\n--- Zero-Shot Evaluation Results (Fine-Tuned) ---")
    print(f"Total samples evaluated: {len(true_labels_zero_ft)}")
    print(f"Accuracy: {accuracy_zero_ft:.4f}")
    print(f"Macro F1 Score: {macro_f1_zero_ft:.4f}")
    print(f"Weighted F1 Score: {weighted_f1_zero_ft:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels_zero_ft, pred_labels_zero_ft, labels=class_labels, digits=4, zero_division=0))

    wandb.log({
        "zero_shot_accuracy_finetuned": accuracy_zero_ft,
        "zero_shot_macro_f1_finetuned": macro_f1_zero_ft,
        "zero_shot_weighted_f1_finetuned": weighted_f1_zero_ft,
        "zero_shot_classification_report_finetuned": classification_report(true_labels_zero_ft, pred_labels_zero_ft, labels=class_labels, digits=4, zero_division=0, output_dict=True)
    })

    # Evaluate three-shot performance after fine-tuning
    print("Evaluating three-shot performance after fine-tuning...")
    true_labels_three_ft, pred_labels_three_ft = [], []
    for item in tqdm(test_dataset, desc="Three-Shot Classifying (Fine-Tuned)"):
        text, target = item['text'], item['target']
        true_polarity = item['polarity']
        few_shot_examples = select_few_shot_examples(item, train_dataset, simcse_model, simcse_tokenizer, device, num_shots=3)
        prompt = create_few_shot_prompt(text, target, few_shot_examples)
        predicted_polarity = get_flexible_logit_prediction(
            model, tokenizer, prompt, sentiment_token_map, device
        )
        true_labels_three_ft.append(true_polarity)
        pred_labels_three_ft.append(predicted_polarity)

    # Define class labels for evaluation
    class_labels = ['positive', 'negative', 'neutral']
    
    accuracy_three_ft = accuracy_score(true_labels_three_ft, pred_labels_three_ft)
    macro_f1_three_ft = f1_score(true_labels_three_ft, pred_labels_three_ft, average='macro', labels=class_labels, zero_division=0)
    weighted_f1_three_ft = f1_score(true_labels_three_ft, pred_labels_three_ft, average='weighted', labels=class_labels, zero_division=0)

    print("\n--- Three-Shot Evaluation Results (Fine-Tuned) ---")
    print(f"Total samples evaluated: {len(true_labels_three_ft)}")
    print(f"Accuracy: {accuracy_three_ft:.4f}")
    print(f"Macro F1 Score: {macro_f1_three_ft:.4f}")
    print(f"Weighted F1 Score: {weighted_f1_three_ft:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels_three_ft, pred_labels_three_ft, labels=class_labels, digits=4, zero_division=0))

    wandb.log({
        "three_shot_accuracy_finetuned": accuracy_three_ft,
        "three_shot_macro_f1_finetuned": macro_f1_three_ft,
        "three_shot_weighted_f1_finetuned": weighted_f1_three_ft,
        "three_shot_classification_report_finetuned": classification_report(true_labels_three_ft, pred_labels_three_ft, labels=class_labels, digits=4, zero_division=0, output_dict=True)
    })

    wandb.finish()

if __name__ == "__main__":
    main()
