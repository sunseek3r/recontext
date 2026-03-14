
import json
import random
import string

from openai import OpenAI

import sacrebleu
from src.create_context import create_context_for_repo

from pathlib import Path
from tqdm import tqdm

import os
from dotenv import load_dotenv

load_dotenv()
CONTEXT_LENGTH_LIMIT = 16_000


def truncate_context_from_left(context_str: str, limit: int = CONTEXT_LENGTH_LIMIT) -> str:
    if len(context_str) <= limit:
        return context_str

    return context_str[-limit:]


def complete_middle(prefix, suffix, context_str):
    """
    Placeholder for a call to an LLM to complete the middle.
    """
    # For now, return a fixed string.
    MODEL_ID = os.getenv("LLM_MODEL")

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")


    full_prompt = (
        f"<|fim_prefix|>{context_str}\n"
        f"// Current file:\n"
        f"{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    )

    response = client.completions.create(
        model=MODEL_ID,
        prompt=full_prompt,
        max_tokens=512,    # completions are usually short
        temperature=0.0,   # stay precise for coding
        # Stop tokens prevent the model from continuing into the suffix code
        stop=["<|file_separator|>", "<|endoftext|>", "\n\n"]
    )

    return response.choices[0].text

def calculate_chrf_jsonl(predictions_file: str, references_file: str, output_file: str):
    sum_chf = 0
    total = 0
    with open(predictions_file, 'r') as f_preds, \
        open(references_file, 'r') as f_refs, \
        open(output_file, 'w') as f_out:

        for line_num, (pred, ref) in enumerate(zip(f_preds, f_refs)):
            try:
                data_p = json.loads(pred)
                data_r = json.loads(ref)

                comp_p = data_p.get('middle', '')
                comp_r = data_r.get('middle', '')


                chrf_metric = sacrebleu.sentence_chrf(comp_p, [comp_r])
                score = chrf_metric.score
                sum_chf += score
                total += 1

                print(f"Line: {line_num} | score: {score:.2f}")

                output_data = {
                    "line_number": line_num,
                    "score": score,
                    "prediction": comp_p,
                    "reference": comp_r
                }
                f_out.write(json.dumps(output_data) + "\n")
            except json.JSONDecodeError:
                print(f"Line {line_num:03d} | Error: Invalid JSON format. Skipping.")
            except Exception as e:
                print(f"Line {line_num:03d} | Error: {e}")

    print(f"Final Score: {sum_chf / total}")


def evaluate_filler(file_path, answers_path):
    input_path = Path(file_path).name

    out_path = f'predictions/{input_path}'
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    res_path = f'results/{input_path}'
    Path(res_path).parent.mkdir(parents=True, exist_ok=True)

    eval_res_path = f'eval_results/{input_path}'
    Path(eval_res_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'r') as input_file, \
        open(f'{out_path}', 'w') as pred_file, \
            open(f'{res_path}', 'w') as res_file:
            for line in tqdm(input_file):
                data = json.loads(line)

                repo_name = data['repo'].replace('/', '__') + '-' + data['revision']

                file_prefix = data['prefix']
                file_suffix = data['suffix']
                modified_files = data.get('modified')


                context_str = create_context_for_repo(
                    repo_name,
                    file_prefix,
                    file_suffix,
                    use_rag=False,
                    use_modified_files=True,
                    summarize_code_samples=True,
                    summarize_prefix_suffix=False,
                    modified_files=modified_files,
                )
                context_str = truncate_context_from_left(context_str)
                # data['context'] = context_str


                middle = complete_middle(file_prefix, file_suffix, context_str)

                pred_file.write(json.dumps({'middle': middle}) + '\n')
                res_file.write(json.dumps({"context": context_str}) + '\n')
    

    calculate_chrf_jsonl(
        predictions_file=out_path,
        references_file=answers_path,
        output_file=eval_res_path
    )

    
    

if __name__ == '__main__':
    evaluate_filler(f'data/python-{os.getenv("STAGE")}.jsonl', f'data/answers-python-{os.getenv("STAGE")}.jsonl')
