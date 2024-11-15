import gspread
from oauth2client.service_account import ServiceAccountCredentials

from config_data.config import Config, load_config

config: Config = load_config()

PATH_TO_CREDS = config.google_sheet.path_to_creds
table_name = config.google_sheet.table_name


# Задаем настройки авторизации
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name(PATH_TO_CREDS, scope)
gs = gspread.authorize(credentials)


# Записываем в Google Sheet результаты Бутстрап
async def write_bootstrap_to_gs(df):
    df_values = df.values.tolist()

    sheet = gs.open(table_name)
    wish_sheet = sheet.sheet1
    wish_sheet.clear()

    wish_sheet.insert_rows(df_values, row=1)

    spreadsheet_url = sheet.url

    return spreadsheet_url


# Записываем в Google Sheet результаты тестов на нормальность распределения
async def write_normal_test_to_gs(df):
    df_values = df.values.tolist()

    sheet = gs.open(table_name)
    wish_sheet = sheet.sheet1
    wish_sheet.clear()

    wish_sheet.insert_rows(df_values, row=1)


# Записываем в Google Sheet результаты теста на равенство дисперсий
async def write_var_test_to_gs(df):
    df_values = df.values.tolist()

    sheet = gs.open(table_name)
    wish_sheet = sheet.sheet1

    wish_sheet.insert_rows(df_values, row=6)


# Записываем в Google Sheet результаты теста на статистическую значимость выборочных средних
async def write_result_test_to_gs(df):
    df_values = df.values.tolist()

    sheet = gs.open(table_name)
    wish_sheet = sheet.sheet1

    wish_sheet.insert_rows(df_values, row=11)

    spreadsheet_url = sheet.url

    return spreadsheet_url
