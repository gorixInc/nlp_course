import argparse

from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# Task 4
# load your trained model and its tokenizer from the local folder
model = AutoModelForTokenClassification.from_pretrained('output')
tokenizer = AutoTokenizer.from_pretrained('output')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("user_input")  # positional argument
    args = parser.parse_args()
    # Task: tag an arbitrary user input string
    # Example:
    #   Input: "Some Company did something"
    #   Output: [("Some Company", "ORG"), ("did", "0"), ("something", "0")]
    # 1. tokenize the input (the return format should be pytorch)
    user_input = args.user_input
    max_length = len(user_input)
    tokenized_inputs = tokenizer(user_input, return_tensors='pt')
    # 2. pass the tokenized input through the model
    model_output = model(**tokenized_inputs).logits
    # 3. get the tag predictions for each token using argmax
    predicted_token_class_ids = torch.argmax(model_output, dim=-1)
    # 4. using `id2label` from the model's config (model.config) convert the predicted indices into labels (i.e. NER tags)
    id2label = model.config.id2label
    labels = [id2label.get(int(idx)) for idx in predicted_token_class_ids[0]]
    # 5. convert indices (`input_ids`) in the tokenizer's output to tokens using `convert_ids_to_tokens`
    tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs.input_ids[0])
    # 6. align the tokens and the NER tags:
    #   6.1 the output should be a list of tuples: [(word, tag), ...]
    assert len(tokens) == len(labels)
    combined = []
    for i in range(len(tokens)):
        combined.append([tokens[i], labels[i]])
    # 8. filter out "[CLS]" and "[SEP]" tokens from the output
    combined = [el for el in combined if not el[0] in ['[CLS]', '[SEP]']]
    
    #   6.2 the tokenizer splits the input text into subword units, so you should merge them back into words
    #       parts of a bigger word can be identified by `##` in front of the token
    #       the first part of a word consisting of subword units doesn't have `##` in front of it
    combined_rev = combined[::-1]

    for i in range(len(combined_rev) - 1):
        token = combined_rev[i][0]
        if token.startswith('##') and not i == 0:
            combined_rev[i+1][0] += token.replace('##', '')
            combined_rev[i] = None
    combined_rev = [el for el in combined_rev if el is not None]
    
    #    Example: ["Some", "Comp", "##any"], ["B-ORG", "I-ORG", "0"] -> [("Some", "B-ORG"), ("Company", "I-ORG")] 
    # 7. some named entities consist of several words: the beginning is indicated by "B-" in the name of the tag,
    #    while "I-" means the continuation of the previous entity
    #    combine such sequences into a single tuple (words, tag)

    for i in range(len(combined_rev)): 
        tag = combined_rev[i][1]
        token = combined_rev[i][0]
        if tag.startswith('B-'):
            combined_rev[i][1] = tag.replace('B-', '')
            continue
        if tag.startswith('I-') and i < len(combined_rev) - 1:
            combined_rev[i+1][0] += ' ' + token
            combined_rev[i] = None

    combined_rev = [el for el in combined_rev if el is not None] 

    combined = combined_rev[::-1]
    #    Example: [("Some", "B-ORG"), ("Company", "I-ORG")] -> s[("Some Company", "ORG")]
    combined = [tuple(el) for el in combined]
    # 9. print out the input and the tagged output
    print(combined)