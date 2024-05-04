
tokens, labels = ["Some", "Comp", "##any"], ["B-ORG", "I-ORG", "0"]
combined = []
for i in range(len(tokens)):
    combined.append([tokens[i], labels[i]])
combined_rev = combined[::-1]

for i in range(len(combined_rev) - 1):
    token = combined_rev[i][0]
    if token.startswith('##'):
        combined_rev[i+1][0] += token.replace('##', '')
        combined_rev[i] = None
combined_rev = [el for el in combined_rev if el is not None]

for i in range(len(combined_rev) - 1): 
    tag = combined_rev[i][1]
    token = combined_rev[i][0]
    if tag.startswith('I-'):
        combined_rev[i+1][0] += ' ' + token
        combined_rev[i] = None
    if tag.startswith('B-'):
        combined_rev[i][1] = tag.replace('B-', '')
combined_rev = [el for el in combined_rev if el is not None] 


print(combined_rev[::-1])