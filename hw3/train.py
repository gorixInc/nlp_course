import argparse
from functools import partial
from datasets import load_dataset
import evaluate
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--label_column_name", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--label_all_tokens", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--return_entity_level_metrics", action="store_true")
    return parser.parse_args()


def compute_metrics(p, metric, label_list, return_entity_level_metrics):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index which is -100
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


def tokenize_and_align_labels(
        examples,
        tokenizer,
        max_length,
        tokens_col,
        label_col,
        label_all_tokens,
):
    tokenized_inputs = tokenizer(
        examples[tokens_col],  # the column with pre-tokenized text goes here
        truncation=True,
        padding=False,
        max_length=max_length,
        is_split_into_words=True,  # allows us to deal with pre-tokenized input
    )

    labels = []
    for index, label in enumerate(examples[label_col]):
        word_ids = tokenized_inputs.word_ids(batch_index=index)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored by the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    # Trainer API expects the label column to be named "labels" specifically
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def main():

    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    ds = load_dataset(args.dataset_name)
    ds = ds.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            max_length=args.max_length,
            tokens_col="tokens",
            label_col="ner_tags",
            label_all_tokens=args.label_all_tokens,
        )
    )

    label_list = ds["train"].features[args.label_column_name].feature.names
    num_labels = len(label_list)

    seqeval = evaluate.load("seqeval")
    eval_fn = partial(
        compute_metrics,
        metric=seqeval,
        label_list=label_list,
        return_entity_level_metrics=args.return_entity_level_metrics,
    )

    # Task 1.1
    # create `label2id` (NER tags to integer indices) and `id2label` (integer indices to NER tags)
    # dictionaries from the `label_list`

    label2id = {label_list[i]:i for i in range(num_labels)}
    id2label = {label2id[k]:k for k in label2id}
    
    # TODO 1.2
    # initialize the model with `args.model_name_or_path`
    # also pass `num_labels`, `id2label`, `label2id` as parameters
    # to the model with the same names
    # `num_labels` is required to determine the output dimensionality
    # while `id2label` and `label2id will be useful later on for inference
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    

    # TODO 1.3
    # Decide what evaluation and save strategies to use 
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="overall_f1" if args.return_entity_level_metrics else "f1",
        greater_is_better=True,
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=eval_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Early stopping callback implementation
    )

    trainer.train()
    test_metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")
    print(test_metrics)
    trainer.save_model()


if __name__ == "__main__":
    main()
