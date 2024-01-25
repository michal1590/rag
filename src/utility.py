import numpy as np
import matplotlib.pyplot as plt
from src.data_models import Passage


def summary_results(data: list[Passage]) -> None:
    similarity_scores = [qa.similarity for qa in data if qa.similarity]

    # best case
    best_score_idx = np.argmin(similarity_scores)
    best_qa = data[best_score_idx]
    print(f"Best pair\nQuestion: {best_qa.question}\nGenerated answer: {best_qa.generated_answer}\nTrue answer: "
          f"{best_qa.answer}")

    # average case
    if len(similarity_scores) % 2 == 0:  # for even number of points, median is not well-defined
        similarity_scores.pop(0)

    median_score = np.median(similarity_scores)
    median_score_idx = similarity_scores.index(median_score)
    median_qa = data[median_score_idx]
    print(f"\n\nMedian pair\nQuestion: {median_qa.question}\nGenerated answer: {median_qa.generated_answer}\n"
          f"True answer: {median_qa.answer}")

    correct = [qa.LMM_judgment.lower().startswith('yes') for qa in data if qa.LMM_judgment]
    print(f'\n Model ocenił {len(correct)} z {len(data)} jako poprawne odpowiedzi')

    # visualization
    counts, bins = np.histogram(similarity_scores)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.xlabel('Similarity distance')
    plt.ylabel('Count')
    plt.show()
