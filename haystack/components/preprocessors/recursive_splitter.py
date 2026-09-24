# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from copy import deepcopy
from typing import Any, Literal

from haystack import Document, component, default_to_dict, logging
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install tiktoken'") as tiktoken_imports:
    import tiktoken

logger = logging.getLogger(__name__)


@component
class RecursiveDocumentSplitter:
    """
    Recursively chunk text into smaller chunks.

    This component is used to split text into smaller chunks, it does so by recursively applying a list of separators
    to the text.

    The separators are applied in the order they are provided, typically this is a list of separators that are
    applied in a specific order, being the last separator the most specific one.

    Each separator is applied to the text, it then checks each of the resulting chunks, it keeps the chunks that
    are within the split_length, for the ones that are larger than the split_length, it applies the next separator in the
    list to the remaining text.

    This is done until all chunks are smaller than the split_length parameter.

    Example:

    ```python
    from haystack import Document
    from haystack.components.preprocessors import RecursiveDocumentSplitter

    chunker = RecursiveDocumentSplitter(split_length=15, split_overlap=0, separators=["\\n\\n", "\\n", ".", " "])
    text = ('''Artificial intelligence (AI) - Introduction

    AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.
    AI technology is widely used throughout industry, government, and science. Some high-profile applications include advanced web search engines; recommendation systems; interacting via human speech; autonomous vehicles; generative and creative tools; and superhuman play and analysis in strategy games.''')
    doc = Document(content=text)
    doc_chunks = chunker.run([doc])
    print(doc_chunks["documents"])
    # [
    # Document(id=..., content: 'Artificial intelligence (AI) - Introduction\\n\\n', meta: {'source_id': '...', 'parent_id': '...', 'split_id': 0, 'split_idx_start': 0, '_split_overlap': None, 'page_number': 1})
    # Document(id=..., content: 'AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.\\n', meta: {'source_id': '...', 'parent_id': '...', 'split_id': 1, 'split_idx_start': 45, '_split_overlap': None, 'page_number': 1})
    # Document(id=..., content: 'AI technology is widely used throughout industry, government, and science.', meta: {'source_id': '...', 'parent_id': '...', 'split_id': 2, 'split_idx_start': 142, '_split_overlap': None, 'page_number': 1})
    # Document(id=..., content: ' Some high-profile applications include advanced web search engines; recommendation systems; interac...', meta: {'source_id': '...', 'parent_id': '...', 'split_id': 3, 'split_idx_start': 216, '_split_overlap': None, 'page_number': 1})
    # Document(id=..., content: 'vehicles; generative and creative tools; and superhuman play and analysis in strategy games.', meta: {'source_id': '...', 'parent_id': '...', 'split_id': 4, 'split_idx_start': 350, '_split_overlap': None, 'page_number': 1})
    # ]
    ```
    """  # noqa: E501

    def __init__(
        self,
        *,
        split_length: int = 200,
        split_overlap: int = 0,
        split_unit: Literal["word", "char", "token"] = "word",
        separators: list[str] | None = None,
        sentence_splitter_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes a RecursiveDocumentSplitter.

        :param split_length: The maximum length of each chunk by default in words, but can be in characters or tokens.
            See the `split_units` parameter.
        :param split_overlap: The number of overlapping units (words, characters, or tokens, per
            `split_unit`) between consecutive chunks.
        :param split_unit: The unit of the split_length parameter. It can be either "word", "char", or "token".
            If "token" is selected, the text will be split into tokens using the tiktoken tokenizer (o200k_base).
            Special-token strings in document content are encoded as ordinary text.
        :param separators: An optional list of separator strings to use for splitting the text. The string
            separators will be treated as regular expressions unless the separator is "sentence", in that case the
            text will be split into sentences using a custom sentence tokenizer based on NLTK.
            See: haystack.components.preprocessors.sentence_tokenizer.SentenceSplitter.
            If no separators are provided, the default separators ["\\n\\n", "sentence", "\\n", " "] are used.
        :param sentence_splitter_params: Optional parameters to pass to the sentence tokenizer.
            See: haystack.components.preprocessors.sentence_tokenizer.SentenceSplitter for more information.

        :raises ValueError: If the overlap is greater than or equal to the chunk size or if the overlap is negative, or
                            if any separator is not a string.
        """
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.split_units = split_unit
        self.separators = separators if separators else ["\n\n", "sentence", "\n", " "]  # default separators
        self._check_params()
        self.nltk_tokenizer = None
        self.sentence_splitter_params = (
            {"keep_white_spaces": True} if sentence_splitter_params is None else sentence_splitter_params
        )
        self.tiktoken_tokenizer: "tiktoken.Encoding" | None = None
        self._is_warmed_up = False
