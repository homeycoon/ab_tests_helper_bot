from dataclasses import dataclass
from environs import Env


@dataclass
class TgBot:
    token: str


@dataclass
class GoogleSheet:
    path_to_creds: str
    email: str
    table_name: str


@dataclass
class Config:
    tg_bot: TgBot
    google_sheet: GoogleSheet


def load_config(path: str | None = None) -> Config:
    env = Env()
    env.read_env(path)
    return Config(
        tg_bot=TgBot(token=env('BOT_TOKEN')),
        google_sheet=GoogleSheet(
            path_to_creds=env('PATH_TO_CREDS'),
            email=env('EMAIL'),
            table_name=env('TABLE_NAME')
        )
    )
