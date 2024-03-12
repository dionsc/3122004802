import sys
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import pytest

@pytest.mark.benchmark(min_rounds=10)

#对中文字段进行分词
def tokenize_zh(text):
    return list(jieba.cut(text,cut_all=False))

# 计算两个文本的重复率
def calculate_repeat_rate(orig, orig_add):

    # 使用TF-IDF向量化器将文本转换为向量
    vectorizer = TfidfVectorizer(tokenizer=tokenize_zh, lowercase=False)
    tfidf_matrix = vectorizer.fit_transform([orig, orig_add])

    # 计算两个文本之间的余弦相似度
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return similarity

def solve(orig,orig_add,ans):

    # 读取原文
    with open(orig, 'r', encoding='utf-8') as f:
        orig_text = f.read()

    # 读取抄袭文本
    with open(orig_add, 'r', encoding='utf-8') as f:
        orig_add_text = f.read()

    # 计算查重率
    repeat_rate = calculate_repeat_rate(orig_text, orig_add_text)

    # 将查重率输出到答案文件
    with open(ans, 'w', encoding='utf-8') as f:
        f.write("{:.2f}".format(int(repeat_rate*100)/100))



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("命令行参数格式错误，正确格式: python main.py <orig> <orig_add> <ans>")
        sys.exit(1)
    solve(sys.argv[1],sys.argv[2],sys.argv[3])
