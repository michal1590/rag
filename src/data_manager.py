from loguru import logger
import pandas as pd
import spacy

from src.data_models import Passage


class DataManager:
    def __init__(self, min_words_count: int):
        self._spacy_nlp = spacy.load('pl_core_news_sm')
        self._min_words_count = min_words_count

    def prepare_data(self, path: str) -> list[Passage]:
        logger.info(f"Working on file {path}")

        df = pd.read_excel(path)
        logger.debug(f"Raw file contains {len(df)} rows")

        df = self._preprocess(df)
        logger.debug(f"Preprocessed file contains {len(df)} rows")

        collection = self._make_collection(df)
        return collection

    @staticmethod
    def _make_collection(df: pd.DataFrame) -> list[Passage]:
        collection = []
        for i, row in df.iterrows():
            collection.append(Passage(question=row['Q'],
                                      answer=row['A'],
                                      index=i))
        return collection

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # drop single word answers
        df = self._remove_short_answers(df, threshold=self._min_words_count)

        # remove "tak" and "nie" from the beginning of answers
        df['A'] = df['A'].map(lambda x: self._remove_leading_terms(x, ['tak,', 'nie,', 'tak.', 'nie.']))

        return df

    def _remove_short_answers(self, df: pd.DataFrame, threshold: int) -> pd.DataFrame:
        df['word_count'] = df['A'].map(self._count_words)
        df['to_drop'] = df['word_count'] < threshold
        df = df.loc[df['to_drop'] == False, ['Q', 'A']]
        return df

    def _count_words(self, text: str) -> int:
        doc = self._spacy_nlp(text)
        return len(doc)

    @staticmethod
    def _remove_leading_terms(text: str, terms: list[str]) -> str:
        for term in terms:
            if text.lower().startswith(term.lower()):  # space is added to selected only words
                text = text[len(term):].strip()

                # make first letter capital one
                text = text[0].upper() + text[1:]

        return text