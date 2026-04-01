import os
import random
from rouge_score import rouge_scorer
from summarizer import summarize

ARTICLES_PATH = "data/News Articles"
SUMMARIES_PATH = "data/Summaries"

def get_all_article_paths():
    paths = []
    for category in os.listdir(ARTICLES_PATH):
        cat_path = os.path.join(ARTICLES_PATH, category)
        if os.path.isdir(cat_path):
            for filename in os.listdir(cat_path):
                if filename.endswith('.txt'):
                    article_path = os.path.join(cat_path, filename)
                    summary_path = os.path.join(SUMMARIES_PATH, category, filename)
                    if os.path.exists(summary_path):
                        paths.append((article_path, summary_path))
    return paths


def read_file(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read().strip()


def evaluate(method='tfidf', n=3, sample_size=100):
    all_paths = get_all_article_paths()
    sample = random.sample(all_paths, min(sample_size, len(all_paths)))

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    r1, r2, rl = [], [], []

    for article_path, summary_path in sample:
        article = read_file(article_path)
        reference = read_file(summary_path)
        generated = summarize(article, method=method, n=n)
        scores = scorer.score(reference, generated)
        r1.append(scores['rouge1'].fmeasure)
        r2.append(scores['rouge2'].fmeasure)
        rl.append(scores['rougeL'].fmeasure)

    print(f"\nMethod: {method.upper()} | Articles evaluated: {len(sample)}")
    print(f"ROUGE-1 : {sum(r1)/len(r1):.4f}")
    print(f"ROUGE-2 : {sum(r2)/len(r2):.4f}")
    print(f"ROUGE-L : {sum(rl)/len(rl):.4f}")


def get_random_article():
    all_paths = get_all_article_paths()
    article_path, summary_path = random.choice(all_paths)
    article = read_file(article_path)
    reference = read_file(summary_path)
    category = article_path.split(os.sep)[-2]
    filename = os.path.basename(article_path)
    return article, reference, category, filename


if __name__ == "__main__":
    evaluate(method='tfidf', n=3)
    evaluate(method='textrank', n=3)
    evaluate(method='frequency', n=3)