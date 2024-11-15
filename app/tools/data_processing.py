import io
import logging

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import pingouin as pg
from aiogram import Bot

from tools.google_sheets import (write_var_test_to_gs,
                                 write_result_test_to_gs, write_normal_test_to_gs, write_bootstrap_to_gs)

logger = logging.getLogger(__name__)

# Обработка данных методом Бутстрап
async def bootstrap_processing(df, count_cols, alfa):
    test_name = 'Бутстрап'
    alfa_float = float(alfa)
    cols_l = df.columns.to_list()
    if count_cols == 2:
        data_a = df.iloc[:, 0]
        data_b = df.iloc[:, 1]
        median_diff = []
        for i in range(1000):
            sample_data_a = data_a.sample(frac=1, replace=True)
            sample_median_a = sample_data_a.median()

            sample_data_b = data_b.sample(frac=1, replace=True)
            sample_median_b = sample_data_b.median()

            sample_median_diff = sample_median_a - sample_median_b
            median_diff.append(sample_median_diff)

        if alfa_float == 0.05:
            lower_quantile = pd.Series(median_diff).quantile(0.025)
            upper_quantile = pd.Series(median_diff).quantile(0.975)
        else:
            lower_quantile = pd.Series(median_diff).quantile(0.005)
            upper_quantile = pd.Series(median_diff).quantile(0.995)

        if (np.all([x > 0 for x in [lower_quantile, upper_quantile]])
                or np.all([x < 0 for x in [lower_quantile, upper_quantile]])):
            conclusion = 'Статистически значимое различие'
        else:
            conclusion = 'Статистически незначимое различие'
        bootstrap_df = pd.DataFrame([['Бутстрап', None],
                                    ['Нижний квартиль разниц медиан', 'Верхний квартиль разниц медиан'],
                                    [lower_quantile, upper_quantile]])

        text = (f'Применен тест: {test_name}\n'
                f'Нижний квартиль разниц медиан: {lower_quantile:.4f}\n'
                f'Верхний квартиль разниц медиан: {upper_quantile:.4f}\n'
                f'Вывод: {conclusion}\n\n')
    else:
        data_a = df.iloc[:, 0]
        data_b = df.iloc[:, 1]
        data_c = df.iloc[:, 2]

        median_diff_ab = []
        median_diff_ac = []
        median_diff_bc = []
        for i in range(1000):
            sample_data_a = data_a.sample(frac=1, replace=True)
            sample_median_a = sample_data_a.median()

            sample_data_b = data_b.sample(frac=1, replace=True)
            sample_median_b = sample_data_b.median()

            sample_data_c = data_c.sample(frac=1, replace=True)
            sample_median_c = sample_data_c.median()

            sample_median_diff_ab = sample_median_a - sample_median_b
            median_diff_ab.append(sample_median_diff_ab)

            sample_median_diff_ac = sample_median_a - sample_median_c
            median_diff_ac.append(sample_median_diff_ac)

            sample_median_diff_bc = sample_median_b - sample_median_c
            median_diff_bc.append(sample_median_diff_bc)

        if alfa_float == 0.05:
            lower_quantile_ab = pd.Series(median_diff_ab).quantile(0.025)
            upper_quantile_ab = pd.Series(median_diff_ab).quantile(0.975)

            lower_quantile_ac = pd.Series(median_diff_ac).quantile(0.025)
            upper_quantile_ac = pd.Series(median_diff_ac).quantile(0.975)

            lower_quantile_bc = pd.Series(median_diff_bc).quantile(0.025)
            upper_quantile_bc = pd.Series(median_diff_bc).quantile(0.975)
        else:
            lower_quantile_ab = pd.Series(median_diff_ab).quantile(0.005)
            upper_quantile_ab = pd.Series(median_diff_ab).quantile(0.995)

            lower_quantile_ac = pd.Series(median_diff_ac).quantile(0.005)
            upper_quantile_ac = pd.Series(median_diff_ac).quantile(0.995)

            lower_quantile_bc = pd.Series(median_diff_bc).quantile(0.005)
            upper_quantile_bc = pd.Series(median_diff_bc).quantile(0.995)

        conclusion_ab = (f'Статистически значимое различие между {cols_l[0]} и {cols_l[1]} группами'
                         if (np.all([x > 0 for x in [lower_quantile_ab, upper_quantile_ab]])
                         or np.all([x < 0 for x in [lower_quantile_ab, upper_quantile_ab]]))
                         else f'Статистически незначимое различие между {cols_l[0]} и {cols_l[1]} группами')

        conclusion_ac = (f'Статистически значимое различие между {cols_l[0]} и {cols_l[2]} группами'
                         if (np.all([x > 0 for x in [lower_quantile_ac, upper_quantile_ac]])
                             or np.all([x < 0 for x in [lower_quantile_ac, upper_quantile_ac]]))
                         else f'Статистически незначимое различие между {cols_l[0]} и {cols_l[2]} группами')

        conclusion_bc = (f'Статистически значимое различие между {cols_l[1]} и {cols_l[2]} группами'
                         if (np.all([x > 0 for x in [lower_quantile_bc, upper_quantile_bc]])
                             or np.all([x < 0 for x in [lower_quantile_bc, upper_quantile_bc]]))
                         else f'Статистически незначимое различие между {cols_l[1]} и {cols_l[2]} группами')

        bootstrap_df = pd.DataFrame([['Бутстрап', None, None,
                                      None, None, None,
                                      None, None, None],
                                     [f'Нижний квартиль разниц медиан для столбцов {cols_l[0]} и {cols_l[1]}',
                                      f'Верхний квартиль разниц медиан для столбцов {cols_l[0]} и {cols_l[1]}', None,
                                      f'Нижний квартиль разниц медиан для столбцов {cols_l[0]} и {cols_l[2]}',
                                      f'Верхний квартиль разниц медиан для столбцов {cols_l[0]} и {cols_l[1]}', None,
                                      f'Нижний квартиль разниц медиан для столбцов {cols_l[1]} и {cols_l[2]}',
                                      f'Верхний квартиль разниц медиан для столбцов {cols_l[0]} и {cols_l[2]}', None],
                                     [lower_quantile_ab, upper_quantile_ab, None,
                                      lower_quantile_ac, upper_quantile_ac, None,
                                      lower_quantile_bc, upper_quantile_bc, None]])
        text = (f'Применен тест: {test_name}\n'
                f'Нижний квартиль разниц медиан для столбцов {cols_l[0]} и {cols_l[1]}: {lower_quantile_ab:.4f}\n'
                f'Верхний квартиль разниц медиан для столбцов {cols_l[0]} и {cols_l[1]}: {upper_quantile_ab:.4f}\n'
                f'Вывод: {conclusion_ab }\n\n'
                f'Нижний квартиль разниц медиан для столбцов {cols_l[0]} и {cols_l[2]}: {lower_quantile_ac:.4f}\n'
                f'Верхний квартиль разниц медиан для столбцов {cols_l[0]} и {cols_l[1]}: {upper_quantile_ac:.4f}\n'
                f'Вывод: {conclusion_ac}\n\n'
                f'Нижний квартиль разниц медиан для столбцов {cols_l[1]} и {cols_l[2]}: {lower_quantile_bc:.4f}\n'
                f'Верхний квартиль разниц медиан для столбцов {cols_l[0]} и {cols_l[2]}: {upper_quantile_bc:.4f}\n'
                f'Вывод: {conclusion_bc}\n\n')

    gs_url = await write_bootstrap_to_gs(bootstrap_df)

    text += f'Ссылка на google sheet: {gs_url}'

    return text


# Тест Шапиро-Уилка на нормальность распределения
async def shapiro_test_processing(data):
    shapiro_test = stats.shapiro(data)
    return {
        'test_name': 'Тест Шапиро-Уилка',
        'test_result': shapiro_test,
        'is_normal': shapiro_test[1] >= 0.05
    }


# Тест Колмогорова-Смирнова на нормальность распределения
async def ks_test_processing(data):
    loc, scale = norm.fit(data)
    n = norm(loc=loc, scale=scale)
    ks_test = stats.kstest(data, n.cdf)

    return {
        'test_name': 'Тест Колмогорова-Смирнова',
        'test_result': ks_test,
        'is_normal': ks_test.pvalue >= 0.05
    }


# Тест Левене на равенство дисперсий
async def levene_test_processing(data_a, data_b, data_c=None):
    if data_c is None:
        levene_test = stats.levene(data_a, data_b, center='mean')
    else:
        levene_test = stats.levene(data_a, data_b, data_c, center='mean')

    if levene_test.pvalue < 0.05:
        equal_var = False
    else:
        equal_var = True
    levene_df = pd.DataFrame([['Тест Левене', None],
                              ['Statistic', 'p-value'],
                              [levene_test.statistic, levene_test.pvalue]])
    await write_var_test_to_gs(df=levene_df)
    return equal_var


# Обработка датафрейма с двумя столбцами
async def two_cols_df(df, alfa):
    data_a = df.iloc[:, 0]
    data_b = df.iloc[:, 1]

    # Проверяем выборку А
    if data_a.count() > 5000:
        a_test = {
            "test_name": "Без теста, т.к. выборка > 5000",
            "test_result": [None, None]
        }
        a_is_normal = True
    elif 50 < data_a.count() <= 5000:
        a_test = await ks_test_processing(data_a)
        a_is_normal = a_test["is_normal"]
    else:
        a_test = await shapiro_test_processing(data_a)
        a_is_normal = a_test["is_normal"]

    # Проверяем выборку B
    if data_b.count() > 5000:
        b_test = {
            "test_name": "Без теста, т.к. выборка > 5000",
            "test_result": [None, None]
        }
        b_is_normal = True
    elif 50 < data_b.count() <= 5000:
        b_test = await ks_test_processing(data_b)
        b_is_normal = b_test["is_normal"]
    else:
        b_test = await shapiro_test_processing(data_b)
        b_is_normal = b_test["is_normal"]

    is_normal = all((a_is_normal, b_is_normal))

    test_check_df = pd.DataFrame(
        [[None, a_test["test_name"], None,
          None, b_test["test_name"], None],
         ['Statistic_a', 'p-value_a', None,
         'Statistic_b', 'p-value_b', None],
         [a_test["test_result"][0], a_test["test_result"][1], None,
          b_test["test_result"][0], b_test["test_result"][1], None]]
    )
    await write_normal_test_to_gs(df=test_check_df)

    if is_normal:
        equal_var = await levene_test_processing(data_a, data_b)
    else:
        equal_var = False

    if is_normal and equal_var:
        test_result = stats.ttest_ind(data_a, data_b)
        test_name = 't-тест для независимых выборок с равными дисперсиями'
    elif is_normal and not equal_var:
        test_result = stats.ttest_ind(data_a, data_b, equal_var=False)
        test_name = 't-тест для независимых выборок с неравными дисперсиями'
    else:
        test_result = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
        test_name = 'Тест Манна-Уитни для независимых выборок'

    result_test_df = pd.DataFrame([[test_name, None],
                                   ['Statistic', 'p-value'],
                                   [test_result.statistic, test_result.pvalue]])

    gs_url = await write_result_test_to_gs(result_test_df)
    alfa_float = float(alfa)

    conclusion = 'Статистически значимое различие' if test_result.pvalue < alfa_float else 'Статистически незначимое различие'
    text = (f'Применен тест: {test_name}\n'
            f'Полученное p-value: {test_result.pvalue:.4f}\n'
            f'Вывод: {conclusion}\n\n'
            f'Ссылка на google sheet: {gs_url}')
    return text


# Обработка датафрейма с тремя столбцами
async def three_cols_df(df, alfa):
    data_a = df.iloc[:, 0]
    data_b = df.iloc[:, 1]
    data_c = df.iloc[:, 2]

    # Проверяем выборку А
    if data_a.count() > 5000:
        a_test = {
            "test_name": "Без теста, т.к. выборка > 5000",
            "test_result": [None, None]
        }
        a_is_normal = True
    elif 50 < data_a.count() <= 5000:
        a_test = await ks_test_processing(data_a)
        a_is_normal = a_test["is_normal"]
    else:
        a_test = await shapiro_test_processing(data_a)
        a_is_normal = a_test["is_normal"]

    # Проверяем выборку B
    if data_b.count() > 5000:
        b_test = {
            "test_name": "Без теста, т.к. выборка > 5000",
            "test_result": [None, None]
        }
        b_is_normal = True
    elif 50 < data_b.count() <= 5000:
        b_test = await ks_test_processing(data_b)
        b_is_normal = b_test["is_normal"]
    else:
        b_test = await shapiro_test_processing(data_b)
        b_is_normal = b_test["is_normal"]

    # Проверяем выборку C
    if data_c.count() > 5000:
        c_test = {
            "test_name": "Без теста, т.к. выборка > 5000",
            "test_result": [None, None]
        }
        c_is_normal = True
    elif 50 < data_c.count() <= 5000:
        c_test = await ks_test_processing(data_c)
        c_is_normal = c_test["is_normal"]
    else:
        c_test = await shapiro_test_processing(data_c)
        c_is_normal = c_test["is_normal"]

    is_normal = all((a_is_normal, b_is_normal, c_is_normal))

    test_check_df = pd.DataFrame(
        [[None, a_test["test_name"], None,
          None, b_test["test_name"], None,
          None, c_test["test_name"], None],
         ['Statistic_a', 'p-value_a', None,
          'Statistic_b', 'p-value_b', None,
          'Statistic_c', 'p-value_c', None],
         [a_test["test_result"][0], a_test["test_result"][1], None,
          b_test["test_result"][0], b_test["test_result"][1], None,
          c_test["test_result"][0], c_test["test_result"][1], None]]
    )
    await write_normal_test_to_gs(df=test_check_df)

    if is_normal:
        equal_var = await levene_test_processing(data_a, data_b, data_c)
    else:
        equal_var = False

    if is_normal and equal_var:
        test_result = stats.f_oneway(data_a, data_b, data_c)
        test_name = 'Однофакторный ANOVA'
        statistic = test_result.statistic
        df_num = None
        df_den = None
        pvalue = test_result.pvalue
        result_test_df = pd.DataFrame([[test_name, None],
                                       ['Statistic', 'p-value'],
                                       [statistic, pvalue]])
    elif is_normal and not equal_var:
        cols = df.columns.to_list()
        df_long = pd.melt(df, value_vars=cols, var_name='Группа', value_name='Значение')
        test_result = pg.welch_anova(data=df_long, dv="Значение", between="Группа")
        test_name = 'Дисперсионный анализ Уэлча'
        statistic = float(test_result['F'].values[0])
        df_num = float(test_result['ddof1'].values[0])
        df_den = float(test_result['ddof2'].values[0])
        pvalue = float(test_result['p-unc'].values[0])
        result_test_df = pd.DataFrame([[test_name, None],
                                       ['Statistic', 'p-value', 'df_num', 'df_den'],
                                       [statistic, pvalue, df_num, df_den]])
    else:
        test_result = stats.kruskal(data_a, data_b, data_c)
        test_name = 'Тест Краскела-Уоллиса для независимых выборок'
        statistic = test_result.statistic
        df_num = None
        df_den = None
        pvalue = test_result.pvalue
        result_test_df = pd.DataFrame([[test_name, None],
                                       ['Statistic', 'p-value'],
                                       [statistic, pvalue]])
    gs_url = await write_result_test_to_gs(result_test_df)
    alfa_float = float(alfa)
    conclusion = 'Статистически значимое различие' if pvalue < alfa_float else 'Статистически незначимое различие'
    text = (f'Применен тест: {test_name}\n' +
            (f'Степень свободы (числитель): {round(df_num, 4)}\n' if df_num else '') +
            (f'Степень свободы (знаменатель): {round(df_den, 4)}\n' if df_den else '') +
            f'Полученное p-value: {pvalue:.4f}\n'
            f'Вывод: {conclusion}\n\n'
            f'Ссылка на google sheet: {gs_url}')
    return text


# Выбор подходящей функции для обработки данных
async def df_processing(df, count_cols, bootstrap, alfa):
    df = df.astype(float)
    if bootstrap:
        result = await bootstrap_processing(df, count_cols, alfa)
    else:
        if count_cols == 2:
            result = await two_cols_df(df, alfa)
        elif count_cols == 3:
            result = await three_cols_df(df, alfa)
        else:
            result = 'Не смог обработать - столбцов не 2 или 3'
    return result


# Обработка файла
async def file_processing(file_id, count_cols, bootstrap, alfa, bot: Bot):
    file = await bot.get_file(file_id)
    file_path = file.file_path
    try:
        file_bytesio = io.BytesIO()
        result = await bot.download_file(file_path, destination=file_bytesio)
        result.seek(0)
        file_bytes = file_bytesio.read().decode('utf-8', errors='ignore')
        df = pd.read_csv(io.StringIO(file_bytes))
        result = await df_processing(df=df, count_cols=count_cols, bootstrap=bootstrap, alfa=alfa)
        return result
    except Exception as e:
        print(e)
        raise
