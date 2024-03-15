labeltext2id={}
labelid2text={}
labels=[]
MAXLENGTH=0
path="../data/train.tsv"
with open(path, "r") as f:
    lines = f.readlines()
    i=1
    while i<len(lines):
        words = lines[i].split(None)
        if words[1].startswith('"') and not words[1].endswith('"'):
            words[1] += lines[i+1].split()[0]
            i+=1
        print(words)
        if words[0] not in labels:
            labels.append(words[0])

        i+=1
print(labels)
for i,label in enumerate(labels):
    labeltext2id[label]=i
print(labeltext2id)
for i,label in enumerate(labels):
    labelid2text[i]=label
print(labelid2text)