"""Base abstract class for LLM punctuators."""

import abc


class LLMPunctuator(abc.ABC):
    """Abstract base class for LLM-based punctuation models.

    This class defines the interface that all LLM punctuator implementations
    must follow.
    """

    @abc.abstractmethod
    def __init__() -> None:
        """Initialize the punctuator."""
        pass

    @abc.abstractmethod
    def add_punctuation() -> str:
        """Add punctuation to text.

        Returns:
            Text with punctuation added.
        """
        pass
