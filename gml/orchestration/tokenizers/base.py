from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Counts tokens in a string for budget arithmetic.

    The ``version`` string ties a count to a specific tokenizer + encoding
    (e.g. ``"tiktoken:cl100k_base"``) so cached counts on MemoryItems can be
    invalidated when the tokenizer changes.
    """

    @property
    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def count(self, text: str) -> int: ...
