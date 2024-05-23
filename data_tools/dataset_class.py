from dataclasses import dataclass
from typing import Any, List, TypeVar, Callable, Type, cast


T = TypeVar("T")


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


@dataclass
class Annotation:
    span_text: str
    category: str
    annotator: str
    start_char: int
    end_char: int
    start_spacy_token: int
    end_spacy_token: int

    @staticmethod
    def from_dict(obj: Any) -> 'Annotation':
        assert isinstance(obj, dict)
        span_text = from_str(obj.get("span_text"))
        category = from_str(obj.get("category"))
        annotator = from_str(obj.get("annotator"))
        start_char = from_int(obj.get("start_char"))
        end_char = from_int(obj.get("end_char"))
        start_spacy_token = from_int(obj.get("start_spacy_token"))
        end_spacy_token = from_int(obj.get("end_spacy_token"))
        return Annotation(span_text, category, annotator, start_char, end_char, start_spacy_token, end_spacy_token)

    def to_dict(self) -> dict:
        result: dict = {}
        result["span_text"] = from_str(self.span_text)
        result["category"] = from_str(self.category)
        result["annotator"] = from_str(self.annotator)
        result["start_char"] = from_int(self.start_char)
        result["end_char"] = from_int(self.end_char)
        result["start_spacy_token"] = from_int(self.start_spacy_token)
        result["end_spacy_token"] = from_int(self.end_spacy_token)
        return result


@dataclass
class DatasetElement:
    id: int
    text: str
    category: str
    annotations: List[Annotation]
    spacy_tokens: str

    @staticmethod
    def from_dict(obj: Any) -> 'DatasetElement':
        assert isinstance(obj, dict)
        id = int(from_str(obj.get("id")))
        text = from_str(obj.get("text"))
        category = from_str(obj.get("category"))
        annotations = from_list(Annotation.from_dict, obj.get("annotations"))
        spacy_tokens = from_str(obj.get("spacy_tokens"))
        return DatasetElement(id, text, category, annotations, spacy_tokens)

    def to_dict(self) -> dict:
        result: dict = {}
        result["id"] = from_str(str(self.id))
        result["text"] = from_str(self.text)
        result["category"] = from_str(self.category)
        result["annotations"] = from_list(lambda x: to_class(Annotation, x), self.annotations)
        result["spacy_tokens"] = from_str(self.spacy_tokens)
        return result


def dataset_from_dict(s: Any) -> List[DatasetElement]:
    return from_list(DatasetElement.from_dict, s)


def dataset_to_dict(x: List[DatasetElement]) -> Any:
    return from_list(lambda x: to_class(DatasetElement, x), x)
