from environs import Env
from dataclasses import dataclass


@dataclass
class Config:
    openai_key: str


def load_config() -> Config:
    env = Env()
    env.read_env()
    return Config(openai_key=env.str("OPENAI_API_KEY"))
