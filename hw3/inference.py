import argparse

from transformers import AutoModelForTokenClassification, AutoTokenizer

# Task 4
# load your trained model and its tokenizer from the local folder
model = AutoModelForTokenClassification.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("user_input")  # positional argument
    args = parser.parse_args()

    # Task: tag an arbitrary user input string
    # Example:
    #   Input: "Some Company did something"
    #   Output: [("Some Company", "ORG"), ("did", "0"), ("something", "0")]
    # 1. tokenize the input (the return format should be pytorch)
    # 2. pass the tokenized input through the model
    # 3. get the tag predictions for each token using argmax
    # 4. using `id2label` from the model's config (model.config) convert the predicted indices into labels (i.e. NER tags)
    # 5. convert indices (`input_ids`) in the tokenizer's output to tokens using `convert_ids_to_tokens`
    # 6. align the tokens and the NER tags:
    #   6.1 the output should be a list of tuples: [(word, tag), ...]
    #   6.2 the tokenizer splits the input text into subword units, so you should merge them back into words
    #       parts of a bigger word can be identified by `##` in front of the token
    #       the first part of a word consisting of subword units doesn't have `##` in front of it
    #    Example: ["Some", "Comp", "##any"], ["B-ORG", "I-ORG", "0"] -> [("Some", "B-ORG"), ("Company", "I-ORG")]
    # 7. some named entities consist of several words: the beginning is indicated by "B-" in the name of the tag,
    #    while "I-" means the continuation of the previous entity
    #    combine such sequences into a single tuple (words, tag)
    #    Example: [("Some", "B-ORG"), ("Company", "I-ORG")] -> [("Some Company", "ORG")]
    # 8. filter out "[CLS]" and "[SEP]" tokens from the output
    # 9. print out the input and the tagged output
