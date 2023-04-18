from rouge_score import rouge_scorer
import numpy as np



def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    if metric_name == "sacrebleu":
        decoded_labels = [[label] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)

    # Extract a few results from ROUGE
    if metric_name == "rouge":
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    else:
        result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)

    return result