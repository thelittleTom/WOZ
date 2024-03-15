from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, path,tokenizer,labeltext2id,pad_size=80):

        self.targets = []
        self.sentences = []
        MAXLENGTH=0
        with open(path, "r") as f:
            lines = f.readlines()
            i=1
            while i<len(lines):
                words = lines[i].split(None)
                if words[1].startswith('"') and not words[1].endswith('"'):
                    words[1] += lines[i+1].split()[0]
                    i+=1

                self.targets.append(labeltext2id[words[0]])
                self.sentences.append(words[1])
                MAXLENGTH=max(MAXLENGTH,len(words[1]))
                i+=1
        MAXLENGTH+=3
        self.texts = [tokenizer(text,
                                padding='max_length',  # 填充到最大长度
                                max_length=pad_size,  # 经过数据分析，最大长度为35  #TODO
                                truncation=True,
                                return_tensors="pt")
                      for text in self.sentences]

    def __getitem__(self, idx):

        return self.texts[idx],self.targets[idx]

    def __len__(self):
        return len(self.texts)
