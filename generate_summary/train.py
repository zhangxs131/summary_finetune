from utils.get_args import ModelArguments,DataTrainingArguments,TaskArguments
from processor.jianxun_en_dataset import Jianxun_EN_Dataset
from processor.jianxun_cn_dataset import Jianxun_CN_Dataset
from processor.title_en_dataset import Title_EN_Dataset
from processor.title_cn_dataset import Title_CN_Dataset

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)

import transformers
from transformers.trainer_utils import is_main_process
from datasets import load_dataset, load_metric

import sys
import os
import logging
import numpy as np


logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TaskArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, task_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, task_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, MBartTokenizer):
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")


    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=label_pad_token_id)

    # Metric
    metric_name = "rouge"
    metric = load_metric(metric_name)

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

    if task_args.task_name == 'jianxun_en':

        train_dataset = Jianxun_EN_Dataset(tokenizer, data_args.train_file)
        eval_dataset = Jianxun_EN_Dataset(tokenizer, data_args.validation_file)

    elif task_args.task_name == 'jianxun_zh':
        train_dataset = Jianxun_CN_Dataset(tokenizer, data_args.train_file)
        eval_dataset = Jianxun_CN_Dataset(tokenizer, data_args.validation_file)

    elif task_args.task_name == 'title_en':
        train_dataset = Title_EN_Dataset(tokenizer, data_args.train_file)
        eval_dataset = Title_EN_Dataset(tokenizer, data_args.validation_file)

    elif task_args.task_name == 'title_zh':
        train_dataset = Title_CN_Dataset(tokenizer, data_args.train_file)
        eval_dataset = Title_CN_Dataset(tokenizer, data_args.validation_file)

    else:
        print('Wrong task name')


    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:

        train_result = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

if __name__ == "__main__":
    main()


