from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class GenerateRequest:
    prompt: str

@dataclass(frozen=True, slots=True)
class GenerateResponse:
    test: str
