import matplotlib.pyplot as plt

def count_characters_in_lines( path):
    line_lengths = []

    with open(path, "r") as f:
        lines = f.readlines()
        i = 1
        while i < len(lines):
            words = lines[i].split(None)
            if words[1].startswith('"') and not words[1].endswith('"'):
                words[1] += lines[i + 1].split()[0]
                i += 1

            words[1] ='0'+ words[1]

            i += 1

            content, label = words[1], words[0]
            line_lengths.append(len(content))

    return line_lengths

def plot_line_lengths_distribution(line_lengths):
    plt.hist(line_lengths, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Number of Characters in Line')
    plt.ylabel('Frequency')
    plt.title('Distribution of Line Lengths')
    plt.show()

if __name__ == "__main__":
    file_path = '/root/yizhi/data/train.tsv'  # 将 'your_file.txt' 替换为你的文件路径
    line_lengths = count_characters_in_lines(file_path)
    plot_line_lengths_distribution(line_lengths)
