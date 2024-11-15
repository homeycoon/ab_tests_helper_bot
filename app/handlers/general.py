import io
import logging

import pandas as pd
import numpy as np

from aiogram import Router, F, Bot
from aiogram.filters import CommandStart, Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import default_state, StatesGroup, State
from aiogram.fsm.storage.redis import RedisStorage, Redis
from aiogram.types import Message, CallbackQuery

from keyboards.keyboards import alfa_kb, after_outliers_kb
from lexicon.lexicon import LEX_COMMANDS, LEX_ANSWERS
from tools.data_processing import file_processing

router: Router = Router()

redis: Redis = Redis(host='localhost')
storage: RedisStorage = RedisStorage(redis=redis)

user_dict: dict[int, dict[str, str | int | bool]] = {}


class FSMQuestions(StatesGroup):
    csv_file = State()
    alfa = State()


# Вызов команды "start" в дефолтном состоянии
@router.message(CommandStart(), StateFilter(default_state))
async def process_start_command_default(message: Message):
    await message.answer(text=LEX_COMMANDS['start'])


# Вызов команды "start" в любом состоянии кроме дефолтного
@router.message(CommandStart(), ~StateFilter(default_state))
async def process_start_command_state(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(text=LEX_COMMANDS['start'])


# Вызов команды "help"
@router.message(Command(commands='help'))
async def help_process_command(message: Message):
    await message.answer(text=LEX_COMMANDS['help'])


# Вызов команды "cancel" в любом состоянии кроме дефолтного
@router.message(Command(commands='cancel'), ~StateFilter(default_state))
async def cancel_process(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(text='Процесс прерван')


# Вызов команды "cancel" в дефолтном состоянии
@router.message(Command(commands='cancel'), StateFilter(default_state))
async def not_cancel_process(message: Message):
    await message.answer(text='Отменять нечего')


# Вызов команды "start_analysis" в дефолтном состоянии
@router.message(Command(commands='start_analysis'), StateFilter(default_state))
async def process_start_test_default(message: Message, state: FSMContext):
    await state.set_state(FSMQuestions.csv_file)
    await message.answer(text=LEX_COMMANDS['start_analysis'])


# Вызов команды "start_analysis" в любом состоянии кроме дефолтного
@router.message(Command(commands='start_analysis'), ~StateFilter(default_state))
async def process_start_test_state(message: Message, state: FSMContext):
    await state.clear()
    await state.set_state(FSMQuestions.csv_file)
    await message.answer(text=LEX_COMMANDS['start_analysis_again'])


# Получение файла в состоянии csv_file и его проверка на соответствие условий
@router.message(StateFilter(FSMQuestions.csv_file),
                F.content_type.in_({'document'}))
async def process_csv_file(message: Message, state: FSMContext, bot: Bot):
    file_id = message.document.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    try:
        file_bytesio = io.BytesIO()
        result = await bot.download_file(file_path, destination=file_bytesio)
        result.seek(0)
        file_bytes = file_bytesio.read().decode('utf-8', errors='ignore')
        df = pd.read_csv(io.StringIO(file_bytes))
        if not df.isnull().values.any():
            if len(df.columns) == 2:
                count_cols = 2
            elif len(df.columns) == 3:
                count_cols = 3
            else:
                count_cols = 0
            if count_cols in (2, 3):
                try:
                    df = df.astype(float)
                    text = ''
                    flag = False
                    for i in df.columns.to_list():
                        col = df[i]
                        q1 = np.percentile(col, 25)
                        q3 = np.percentile(col, 75)
                        IQR = q3 - q1
                        lower_bound = q1 - 1.5 * IQR
                        upper_bound = q3 + 1.5 * IQR
                        outliers = [str(x) for x in col if x < lower_bound or x > upper_bound]
                        if outliers:
                            text += f'Столбец {i} имеет выбросы: {", ".join(outliers)}\n'
                            flag = True
                    await state.update_data(
                        file_unique_id=message.document.file_unique_id,
                        file_id=message.document.file_id,
                        count_cols=count_cols
                    )
                    # Оставлено на случай, если надо будет закинуть в redis
                    # df = df.astype(float)
                    # df_json = df.to_json(orient='records')
                    # await redis.set(f'{message.from_user.id}_df', df_json, ex=3600)

                    if flag:
                        await message.answer(text='В ваших данных есть выбросы.\n\n'
                                                  + text +
                                                  LEX_ANSWERS['outliers_caution'],
                                             reply_markup=after_outliers_kb)
                    else:
                        await state.update_data(bootstrap=False)
                        await message.answer(text='Какой уровень значимости установить?',
                                             reply_markup=alfa_kb)
                        await state.set_state(FSMQuestions.alfa)
                except ValueError:
                    await message.answer(text='Проверьте, что ваши данные представлены в числовом формате')
            else:
                await message.answer(text='Проверьте, что файл содержит 2 или 3 столбца')
        else:
            await message.answer(text='В ваших данных есть пропущенные значения.'
                                      'Это может повлиять на результаты анализа. '
                                      'Обработайте их, пожалуйста, и отправьте '
                                      'файл заново без пропущенных значений')
    except Exception:
        await message.answer(text='Проверьте, что формат файла csv с кодировкой UTF-8 '
                                  'и разделитем - запятой')


# Получение ответа на вопрос о дальнейших действиях при наличии выбросов
# в состоянии csv_file
@router.callback_query(StateFilter(FSMQuestions.csv_file),
                       F.data.in_(['continue', 'bootstrap']))
async def process_csv_file_with_outliers(callback: CallbackQuery,  state: FSMContext):
    await state.update_data(bootstrap=True if callback.data == 'bootstrap' else False)
    await callback.message.answer(text='Какой уровень значимости установить?',
                                  reply_markup=alfa_kb)
    await state.set_state(FSMQuestions.alfa)


# Ответ на любое действите кроме загрузки файла в состоянии csv_file
@router.message(StateFilter(FSMQuestions.csv_file))
async def not_process_csv_file(message: Message):
    await message.answer(text='Загрузите csv-файл')


# Получение уровня значимости в состоянии alfa
@router.callback_query(StateFilter(FSMQuestions.alfa),
                       F.data.in_(['0.01', '0.05']))
async def process_alfa_q(callback: CallbackQuery, state: FSMContext, bot: Bot):
    await state.update_data(alfa=callback.data)
    user_dict[callback.message.from_user.id] = await state.get_data()
    await state.clear()
    await callback.message.answer(text='Обработка. Подождите, пожалуйста')
    if callback.message.from_user.id in user_dict:
        try:
            result = await file_processing(
                file_id=user_dict[callback.message.from_user.id]['file_id'],
                count_cols=user_dict[callback.message.from_user.id]['count_cols'],
                bootstrap=user_dict[callback.message.from_user.id]['bootstrap'],
                # user_id=callback.message.from_user.id, Может понадобиться для redis
                alfa=user_dict[callback.message.from_user.id]['alfa'],
                bot=bot
            )
        except:
            await callback.message.answer(text='Возникла ошибка при обработке.')
        else:
            await callback.message.answer(text=result)
    else:
        await callback.message.answer(text='Ваши данные не удается найти')


# Ответ на  любое действие кроме нажатия на инлайн кнопки в состоянии alfa
@router.callback_query(StateFilter(FSMQuestions.alfa))
async def not_process_alfa_q(callback: CallbackQuery):
    await callback.message.edit_text(text='Используйте инлайн кнопки',
                                     reply_markup=alfa_kb)
