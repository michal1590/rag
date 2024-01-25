import sys
sys.path.append('/workdir')  # for Docker purposes

import argparse
from tqdm import tqdm

from config.general_config import MIN_WORDS_COUNT_ANSWER, N_GENERATE_PASSAGES, N_GENERATE_QUESTION
from src.data_manager import DataManager
from src.utility import summary_results
from src.qa_manager import QAManager


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='data/QA_covid.xlsx')
    parser.add_argument('--model_name', type=str, default='ai-forever/mGPT')
    parser.add_argument('--embedder_model_name', type=str, default='sentence-transformers/distiluse-base-multilingual-cased-v2')
    parser.add_argument('--cross_encoder_model_name', type=str, default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    parser.add_argument('--llm_location', type=str, choices=['local', 'chatgpt'], default='chatgpt')
    parser.add_argument('--api_key_file', type=str, default='api_key.txt')
    args = parser.parse_args()

    data_manager = DataManager(min_words_count=MIN_WORDS_COUNT_ANSWER)
    data = data_manager.prepare_data(args.file_path)

    manager = QAManager(data=data, model_name=args.model_name, emdedder_model_name=args.embedder_model_name,
                        cross_encoder_model_name=args.cross_encoder_model_name, llm_location=args.llm_location,
                        api_key_file=args.api_key_file)

    # eval
    for qa in tqdm(data[:10]):
        answer = manager.answer_question(qa.question, n_questions=N_GENERATE_QUESTION, n_passages=N_GENERATE_PASSAGES)
        qa.generated_answer = answer

        similarity = manager.measure_similarity(answer, qa.answer)
        qa.similarity = similarity

        judgment = manager.ask_for_judgment(answer, qa.answer)
        qa.LMM_judgment = judgment

    summary_results(data)
