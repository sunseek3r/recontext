import json
import sacrebleu



def calculate_chrf_jsonl(predictions_file: str, references_file: str, output_file: str):
    with open(predictions_file, 'r') as f_preds, \
        open(references_file, 'r') as f_refs, \
        open(output_file, 'w') as f_out:

        for line_num, (pred, ref) in enumerate(zip(f_preds, f_refs)):
            try:
                data_p = json.loads(pred)
                data_r = json.loads(ref)

                comp_p = data_p.get('completion', '')
                comp_r = data_r.get('completion', '')


                chrf_metric = sacrebleu.sentence_chrf(comp_p, [comp_r])
                score = chrf_metric.score

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