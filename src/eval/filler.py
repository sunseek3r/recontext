import json
import os
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

import sacrebleu
from openai import AsyncOpenAI
from src.create_context import create_context_for_repo

load_dotenv()

CONTEXT_LENGTH_LIMIT = 30_000
MODEL_ID = os.getenv("LLM_MODEL", "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ")

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# A semaphore prevents us from opening 200+ simultaneous OS sockets and crashing Python
MAX_CONCURRENT_JOBS = 50
semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

def truncate_context_from_left(context_str: str, limit: int = CONTEXT_LENGTH_LIMIT) -> str:
    if len(context_str) <= limit:
        return context_str
    return context_str[-limit:]

async def complete_middle_async(full_prompt):
    response = await client.completions.create(
        model=MODEL_ID,
        prompt=full_prompt,
        max_tokens=512,
        temperature=0.0,
        stop=["<|file_separator|>", "<|endoftext|>", "\n\n"]
    )
    return response.choices[0].text

async def process_single_item(line):
    """Handles the entire pipeline for a single file end-to-end."""
    async with semaphore:
        data = json.loads(line)
        repo_name = data['repo'].replace('/', '__') + '-' + data['revision']
        file_prefix = data['prefix']
        file_suffix = data['suffix']
        modified_files = data.get('modified')
        print(f"Starting work on: {repo_name}...")

        # Properly awaiting the newly native async function!
        context_str = await create_context_for_repo(
            repo_name,
            file_prefix,
            file_suffix,
            use_rag=False,
            use_modified_files=True,
            summarize_code_samples=True,
            summarize_prefix_suffix=False,
            modified_files=modified_files,
            use_random_files=False,
            verbose=False
        )
        context_str = truncate_context_from_left(context_str)

        # Build the prompt
        full_prompt = (
            f"<|fim_prefix|>{context_str}\n"
            f"// Current file:\n"
            f"{file_prefix}<|fim_suffix|>{file_suffix}<|fim_middle|>"
        )

        # Fire it off to the Coder the second the context is ready
        middle = await complete_middle_async(full_prompt)

        return middle, context_str


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

                output_data = {
                    "line_number": line_num,
                    "score": score,
                    "prediction": comp_p,
                    "reference": comp_r
                }
                f_out.write(json.dumps(output_data) + "\n")
            except Exception as e:
                print(f"Line {line_num:03d} | Error: {e}")

    if total > 0:
        print(f"\nFinal chrF Score: {sum_chf / total:.2f}")

async def evaluate_filler_async(file_path, answers_path):
    input_path = Path(file_path).name
    out_path = f'predictions/{input_path}'
    res_path = f'results/{input_path}'
    eval_res_path = f'eval_results/{input_path}'
    
    for path in [out_path, res_path, eval_res_path]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'r') as input_file:
        lines = input_file.readlines()

    print(f"🚀 Blasting {len(lines)} jobs through the Async Pipeline...")
    
    # Launch all tasks concurrently
    tasks = [process_single_item(line) for line in lines]
    
    # tqdm.gather gives us a nice progress bar for async tasks!
    results = await tqdm.gather(*tasks, desc="Processing files")

    print("\n💾 Saving and Evaluating...")
    with open(out_path, 'w') as pred_file, open(res_path, 'w') as res_file:
        for middle, context_str in results:
            pred_file.write(json.dumps({'middle': middle}) + '\n')
            res_file.write(json.dumps({"context": context_str}) + '\n')

    calculate_chrf_jsonl(
        predictions_file=out_path,
        references_file=answers_path,
        output_file=eval_res_path
    )

if __name__ == '__main__':
    stage = os.getenv("STAGE", "test")
    file_path = f'data/python-{stage}.jsonl'
    answers_path = f'data/answers-python-{stage}.jsonl'
    
    asyncio.run(evaluate_filler_async(file_path, answers_path))