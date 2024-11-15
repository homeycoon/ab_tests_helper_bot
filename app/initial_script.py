import gspread
from oauth2client.service_account import ServiceAccountCredentials

from config_data.config import Config, load_config

config: Config = load_config()

PATH_TO_CREDS = config.google_sheet.path_to_creds
email = config.google_sheet.email
table_name = config.google_sheet.table_name


# Задаем настройки авторизации
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name(PATH_TO_CREDS, scope)
gs = gspread.authorize(credentials)


def create_new_table(table_name, email, perm_type, role):
    gs.create(table_name)
    sheet = gs.open(table_name)
    sheet.share(email, perm_type, role)


if __name__ == "__main__":
    create_new_table(table_name='AB_test_result', email=email,  perm_type='user', role='writer')
