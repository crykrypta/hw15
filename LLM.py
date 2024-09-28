import openai

from langchain_openai import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

import tiktoken

from config import load_config

# Загрузка конфигурации
config = load_config()
openai.api_key = config.openai_key


# Класс для работы с LLM
class LLM:
    # Инициализация класса
    def __init__(self, path_to_base: str,
                 system: str | None = None,
                 chunk_size: int = 1024,
                 model_name: str = 'gpt-3.5-turbo',
                 metadata_lenths=True):
        """_summary_

        Args:
            path_to_base (str): _description_
            system (str | None, optional): _description_. default: None.
            chunk_size (int, optional): _description_. default: 1024.
            model_name (str, optional): description_. def-lt: 'gpt-3.5-turbo'.
            metadata_lenths (bool, optional): _description_. default: True.
            _type_: _description_
        """

        # Загружаем текст базы
        with open(path_to_base, 'r', encoding='utf-8') as file:
            document = file.read()

        # Токенизация чанков
        def token_counter(text):
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

        # Создаем список чанков
        source_chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
        for chunk in splitter.split_text(document):
            source_chunks.append(Document(page_content=chunk, metadata={}))

        # Создаем индексную базу
        embeddings = OpenAIEmbeddings()
        self.db = FAISS.from_documents(source_chunks,
                                       embeddings)  # type:ignore
        # Сохраняем системный промпт
        self.system = system
        # Сохраняем параметры модели
        self.model_name = model_name

    # Внутренняя функция для более читабельного вывода модели
    def _format_text(self, text: str, limit: int) -> str:
        words = text.split()
        formatted_text = []
        current_line = ""

        for word in words:
            # Проверяем, поместится ли следующее слово в текущую строку
            if len(current_line) + len(word) + 1 <= limit:
                # Если текущая строка пустая, добавляем слово без пробела
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                # Добавляем строку в список и начинаем новую
                formatted_text.append(current_line)
                current_line = word

        # Добавляем последнюю строку, если она не пустая
        if current_line:
            formatted_text.append(current_line)

        # Возвращаем результат в виде строк, разделённых символом новой строки
        return "\n".join(formatted_text)

    # Функция получения ответа от ChatGPT
    def get_answer(self, query: str,
                   temperature: float = 0.0,
                   outline_limit: int = 100) -> str | None:
        # Базовый системный промпт
        system = self.system or '''Ты - нейроконсультант, ответь на
            вопрос пользователя на основе документа с информацией.\n
            Не придумывай ничего от себя, отвечай максимально по документу.\n
            Не упоминай Документ с информацией для ответа пользователю.\n
            Пользователь ничего не должен знать про
            Документ с информацией для ответа пользователю'''

        # Релевантные отрезки из базы
        docs = self.db.similarity_search(query, k=4)
        message_content = '\n'.join([f'{doc.page_content}' for doc in docs])

        user_prompt = f'''
        Вопрос пользователя: {query}\n
        Документ с информацией: {message_content}'''

        client = openai.OpenAI()

        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user_prompt}
        ]

        completion = client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type:ignore
            temperature=0
        )

        return self._format_text(completion.choices[0].message.content,
                                 outline_limit)
