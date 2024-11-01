import torch
import os
import json
import argparse
from utils import load_config
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import torch.nn.functional as F
from dataloaders.PRM_D import PRM_DDataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
accelerator = Accelerator()  # Initialize the Accelerator for handling distributed setup

def collate_fn(batch):
    return [item for item in batch]

def generate_batch_attention_mask(input_ids, batch_step_end_positions, device):
    attention_mask = torch.zeros_like(input_ids, device=device)
    for i, sample_end_positions in enumerate(batch_step_end_positions):
        for end_pos in sample_end_positions:
            attention_mask[i, :end_pos + 1] = 1  # Activate up to current step's end position
    return attention_mask

def run_inference(args, config, dataset):
    # Initialize model and tokenizer, and let Accelerator handle the device assignment
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Prepare the DataLoader with Accelerator
    dataloader = DataLoader(dataset, batch_size=config.get("batch_size", 1), collate_fn=collate_fn)
    model, dataloader = accelerator.prepare(model, dataloader)

    output_file = os.path.join(config["output_path"], f"scored_{config['desc']}.json")
    os.makedirs(config["output_path"], exist_ok=True)

    for problems, steps_list, answers, corrects, subjects, levels, unique_ids, ground_truth_answers in dataloader:
        batch_input_texts = []
        batch_step_end_positions = []

        for problem, steps in zip(problems, steps_list):
            input_text = f"Problem: {problem}\n\n" + "\n".join([f"[STEP {i+1}] {step}" for i, step in enumerate(steps)])
            batch_input_texts.append(input_text)
            
            step_end_positions = []
            current_pos = 0
            for step_text in steps:
                step_length = len(tokenizer.encode(step_text, add_special_tokens=False))
                end_pos = current_pos + step_length - 1
                step_end_positions.append(end_pos)
                current_pos = end_pos + 1
            batch_step_end_positions.append(step_end_positions)

        # Tokenize and move tensors to the device handled by Accelerator
        inputs = tokenizer(batch_input_texts, return_tensors="pt", padding=True, truncation=True)
        attention_mask = generate_batch_attention_mask(inputs['input_ids'], batch_step_end_positions, device=accelerator.device)

        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'].to(accelerator.device), attention_mask=attention_mask)

            positive_token_id = tokenizer.convert_tokens_to_ids("positive")
            neutral_token_id = tokenizer.convert_tokens_to_ids("neutral")
            negative_token_id = tokenizer.convert_tokens_to_ids("negative")
            
            batch_results = []
            for i, step_end_positions in enumerate(batch_step_end_positions):
                step_probs = []
                prm_score = 1
                for end_pos in step_end_positions:
                    logits = outputs.logits[i, end_pos, [positive_token_id, neutral_token_id, negative_token_id]]
                    probabilities = F.softmax(logits, dim=-1).tolist()
                    step_probs.append({
                        "1": probabilities[0],
                        "0": probabilities[1],
                        "-1": probabilities[2],
                    })
                    prm_score *= (probabilities[0] + probabilities[1])

                batch_results.append({
                    "problem": problems[i],
                    "answer": answers[i],
                    "is_correct": corrects[i],
                    "subject": subjects[i],
                    "level": levels[i],
                    "unique_id": unique_ids[i],
                    "steps": steps_list[i],
                    "rating_probs": step_probs,
                    "prm_score": prm_score,
                    "ground_truth_answer": ground_truth_answers[i]
                })

        # Save results only if this process is the main process
        if accelerator.is_main_process:
            with open(output_file, "a") as f:
                for result in batch_results:
                    f.write(json.dumps(result) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML config")
    args = parser.parse_args()

    config = load_config(args.config_path)
    dataset = PRM_DDataset(config["data_path"])

    run_inference(args, config, dataset)

if __name__ == "__main__":
    main()
