import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_repeat_rate(orig, orig_add):


    # 使用TF-IDF向量化器将文本转换为向量
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([orig, orig_add])

    # 计算两个文本之间的余弦相似度
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # 将余弦相似度转换为重复率
    repeat_rate = similarity*100

    return repeat_rate

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("命令行参数格式错误，正确格式: python main.py <orig> <orig_add> <ans>")
        sys.exit(1)

    orig = sys.argv[1]
    orig_add = sys.argv[2]
    ans = sys.argv[3]

    #读取原文
    with open(orig, 'r', encoding='utf-8') as f:
        orig_text = f.read()

    #读取抄袭文本
    with open(orig_add, 'r', encoding='utf-8') as f:
        orig_add_text = f.read()

    #计算查重率
    repeat_rate = calculate_repeat_rate(orig_text, orig_add_text)

    #将查重率输出到答案文件
    with open(ans, 'w', encoding='utf-8') as f:
        f.write(f"{repeat_rate:.2f}%\n")