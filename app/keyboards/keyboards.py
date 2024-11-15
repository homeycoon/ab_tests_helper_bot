from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


alfa_001_b: InlineKeyboardButton = InlineKeyboardButton(text='0.01', callback_data='0.01')
alfa_005_b: InlineKeyboardButton = InlineKeyboardButton(text='0.05', callback_data='0.05')
alfa_kb: InlineKeyboardMarkup = InlineKeyboardMarkup(inline_keyboard=[[alfa_001_b, alfa_005_b]])

continue_b: InlineKeyboardButton = InlineKeyboardButton(text='Продолжить', callback_data='continue')
bootstrap_b: InlineKeyboardButton = InlineKeyboardButton(text='Бутстрап', callback_data='bootstrap')
after_outliers_kb: InlineKeyboardMarkup = InlineKeyboardMarkup(inline_keyboard=[[continue_b], [bootstrap_b]])
