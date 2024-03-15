labelCount={'景点': 0, '餐馆': 0, '酒店': 0, 'thank': 0, '出租': 0, '地铁': 0, 'bye': 0, '没有意图': 0, 'greet': 0}
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
        labelCount[words[0]]+=1

        i+=1
print(labelCount)