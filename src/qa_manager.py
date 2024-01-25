import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from sentence_transformers import CrossEncoder

from src.data_models import Passage


class QAManager:
    def __init__(self, data: list[Passage], model_name: str, emdedder_model_name: str, cross_encoder_model_name: str,
                 llm_location: str = 'local', api_key_file: str = None):
        if llm_location == 'local':
            self._llm = HuggingFacePipeline.from_model_id(model_id=model_name, task='text-generation',
                                                          pipeline_kwargs={"max_new_tokens": 100}, device=0)
        else:  # chatgpt
            self._llm = self._init_openai(api_key_file)

        self._cross_encoder = CrossEncoder(model_name=cross_encoder_model_name, device='cuda')
        self._db = self._create_database(data, emdedder_model_name)
        self._llm_location = llm_location
        self._response_parser = StrOutputParser()
        self._chains = self._create_chains()

    @staticmethod
    def _init_openai(api_key_file: str) -> ChatOpenAI:
        assert api_key_file, "Api key for open AI have to be provided"
        with open(api_key_file, 'r') as f:
            api_key = f.read()
        assert api_key, "Api key for open AI have to be provided - found empty file"

        return ChatOpenAI(openai_api_key=api_key)

    def _create_chains(self):
        chains = {
            'helper_questions': self._create_helper_questions_chain(),
            'ask_llm_for_answer': self._create_ask_for_answer_chain(),
            'ask_llm_for_judgment': self._create_ask_for_judgment_chain(),
        }
        return chains

    def _create_ask_for_judgment_chain(self):
        template = """
        Decide whether these two answer contain same information. Use only 'yes' and 'no' words.

        Answer 1: {true_answer}
         
        Answer 2: {pred_answer}"
        """
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self._llm | self._response_parser

    def _create_ask_for_answer_chain(self):
        template = """
        Odpowiedz na pytanie korzystając jedynie z podanych informacji.
        
        Informacje: {top_passage} 
        
        Pytanie : {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self._llm | self._response_parser

    def _create_helper_questions_chain(self):
        template = """
        You are a patient at the doctor. Suggest up to {n_questions} additional questions, related to one provided by 
        user. Output one question per line. Do not number the questions. Do not use 'answer' or 'question' words. 
        Response language is polish. Remember to output one question per line. 
        
        User's question: {question}   
        """
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self._llm | self._response_parser

    def _create_database(self, data: list[Passage], emdedder_model_name: str) -> Collection:
        logger.info("Creating database")

        db = self._init_db(emdedder_model_name)
        self._populate_db(db, data)

        logger.info("Database created")
        return db

    @staticmethod
    def _init_db(model_name: str) -> Collection:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name, device='cuda')
        chroma_client = chromadb.Client()
        return chroma_client.create_collection("ragdb", embedding_function=embedding_function)

    @staticmethod
    def _populate_db(db: Collection, data: list[Passage]):
        ids = [str(passage.index) for passage in data]
        documents = [passage.answer for passage in data]

        db.add(ids=ids, documents=documents)

    def answer_question(self, question: str, n_questions: int, n_passages: int) -> str:
        """
        1. Ask LLM for supporting questions
        2. Retrieve similar passages for each question
        3. Re-rank passages
        4. Ask final question
        """
        logger.info(f"Answering question: {question}")

        questions = self._get_helper_questions(question, n_questions=n_questions)
        questions_to_query = questions + [question]

        passages = self._extract_passages(questions_to_query, n_passages)
        top_passage = self._get_top_passage(passages, question)

        logger.debug(f"Top passage {top_passage}")

        answer = self._ask_for_answer(question, top_passage)

        logger.info(f"Answer is {answer}")
        return answer

    def _get_helper_questions(self, question: str, n_questions: int) -> list[str]:
        response = self._chains['helper_questions'].invoke({"question": question, "n_questions": n_questions})
        return response.split('\n')

    def _extract_passages(self, questions: list[str], n_passages: int) -> list[str]:
        response = self._db.query(query_texts=questions, n_results=n_passages, include=['documents', 'embeddings'])
        unique_docs = set()
        for documents in response['documents']:
            for doc in documents:
                unique_docs.add(doc)
        return list(unique_docs)

    def _get_top_passage(self, passages: list[str], question: str) -> str:
        pairs = [[question, passage] for passage in passages]
        scores = self._cross_encoder.predict(pairs)

        max_idx = np.argmax(scores)
        return passages[max_idx]

    def _ask_for_answer(self, question: str, top_passage: str) -> str:
        return self._chains['ask_llm_for_answer'].invoke({"question": question, "top_passage": top_passage})

    def measure_similarity(self, pred_answer: str, true_answer: str) -> float:
        # Measure similarity by calculating distance from self-similarity and pred-true
        # Lower is better and 0 is best result

        pred_true_similarity = self._cross_encoder.predict([[pred_answer, true_answer]])[0]
        self_similarity = self._cross_encoder.predict([[true_answer, true_answer]])[0]
        return self_similarity - pred_true_similarity

    def ask_for_judgment(self, pred_answer: str, true_answer: str) -> str:
        return self._chains['ask_llm_for_judgment'].invoke({"true_answer": true_answer, "pred_answer": pred_answer})
