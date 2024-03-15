from torch.utils.data import Dataset
import json
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
CLS='<CLS>'
import  os




class MyDataset(Dataset):

    def __init__(self, path ,labeltext2id,vocab_path,pad_size=60):

        self.targets = []
        self.texts = []
        MAXLENGTH=0
        tokenizer = lambda x: [y for y in x]  # char-level
        if os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                vocab = json.load(f)

        print(f"Vocab size: {len(vocab)}")



        with open(path, "r") as f:
            lines = f.readlines()
            i = 1
            while i < len(lines):
                words = lines[i].split(None)
                if words[1].startswith('"') and not words[1].endswith('"'):
                    words[1] += lines[i + 1].split()[0]
                    i += 1

                self.targets.append(labeltext2id[words[0]])
                words[1]=CLS+words[1]
                MAXLENGTH = max(MAXLENGTH, len(words[1]))
                i += 1

                content, label = words[1], words[0]
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                self.texts.append(words_line)

        print(f"Max length: {MAXLENGTH}")

    def __getitem__(self, idx):

        return self.texts[idx],self.targets[idx]

    def __len__(self):
        return len(self.texts)
