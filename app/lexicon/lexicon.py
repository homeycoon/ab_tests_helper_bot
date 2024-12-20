MAIN_MENU = {
    '/start': 'Запуск/перезапуск бота',
    '/start_analysis': 'Начало анализа A/B-теста',
    '/help': 'Описание бота',
    '/cancel': 'Прервать процесс'
}

LEX_COMMANDS = {
    'start': (
        'Добро пожаловать в бота-помощника в анализе А/В-тестов.\n\n'
        'Этот бот помогает подобрать наиболее подходящий тест '
        'для оценки статистической значимости различий '
        'среднего/медиан в двух/трех независимых выборках.\n'
        'Для начала анализа А/В-теста вызовите команду /start_analysis'
    ),
    'start_analysis': (
        'Процесс анализа А/В-теста начат.\n\n'
        'Прикрепите, пожалуйста, csv-файл (с кодировкой UTF-8 '
        'и разделителем - запятой) с результатами А/В-теста.\n\n'
        'Файл должен соответствовать условиям:\n'
        '* содержать 2 или 3 столбца с любыми заголовками\n'
        '* данные в столбцах должны быть числовыми '
        '(в качестве десятичного разделителя использовать точку)\n'
        '* данные в столбцах должны быть независимы\n'
        '* в данных не должно быть пропущенных значений'
    ),
    'start_analysis_again': (
        'Процесс анализа А/В-теста перезапущен.\n\n'
        'Прикрепите, пожалуйста, csv-файл (с кодировкой UTF-8 '
        'и разделителем - запятой) с результатами А/В-теста.\n\n'
        'Файл должен соответствовать условиям:\n'
        '* содержать 2 или 3 столбца с любыми заголовками\n'
        '* данные в столбцах должны быть числовыми '
        '(в качестве десятичного разделителя использовать точку)\n'
        '* данные в столбцах должны быть независимы\n'
        '* в данных не должно быть пропущенных значений'
    ),
    'help': (
        'Этот бот помогает подобрать наиболее подходящий тест '
        'для оценки статистической значимости различий '
        'среднего/медиан в двух/трех независимых выборках.\n\n'
        'Для запуска/перезапуска бота вызовите команду /start\n'
        'Для начала анализа А/В-теста вызовите команду /start_analysis\n'
        'Для получения описания бота и его основных команд '
        'вызовите команду /help\n'
        'Для прерывания процесса анализа А/В-теста вызовите команду /cancel'
    )
}

LEX_ANSWERS = {
    'outliers_caution': (
        '\nВыбросы могут исказить результаты анализа, особенно '
        'при оценке статистической значимости выборочных средних.\n\n'
        'Если выбросов немного и они незначительно отличаются '
        'от других значений в выборке или если вы считаете, что выбросы '
        'не помешают анализу, нажмите "Продолжить".\n\n'
        'В качестве альтернативы вы можете выбрать метод "Бутстрап", '
        'который оценивает статистическую значимость медиан выборок. '
        'Этот метод более устойчив к выбросам, однако не является полностью надежным. '
        'Для применения метода "Бутстрап", нажмите "Бутстрап".\n\n'
        'В противном случае обработайте выбросы и отправьте файл снова.'
    )
}