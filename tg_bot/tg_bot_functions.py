import logging
import time
import asyncio
import os
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.enums import ParseMode
from aiogram.types import FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.default import DefaultBotProperties
import pandas as pd
import json
from depersonalization_data import DataProcessor
from get_predict_cpu import run as run_forecast
import traceback
from aiogram.exceptions import TelegramNetworkError, TelegramRetryAfter
from typing import Callable, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Замените на ваш токен
TOKEN = "8218748711:AAFj29TWIGODZIqFxwgMNTupz4m8zxWtiRU"

# Инициализация
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
data_processor = DataProcessor()

class TelegramSafe:
    """Класс для безопасных запросов к Telegram"""
    
    @staticmethod
    async def safe_request(
        func: Callable,
        *args,
        max_retries: int = 3,
        retry_delay: int = 2,
        **kwargs
    ) -> Any:
        """
        Безопасный запрос к Telegram с повторными попытками
        
        Args:
            func: Функция для выполнения
            max_retries: Максимальное количество попыток
            retry_delay: Задержка между попытками в секундах
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
                
            except TelegramRetryAfter as e:
                # Если Telegram просит подождать
                wait_time = e.retry_after
                print(f"Telegram просит подождать {wait_time} секунд...")
                await asyncio.sleep(wait_time)
                last_error = e
                
            except TelegramNetworkError as e:
                print(f"Сетевая ошибка (попытка {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    # Ждем перед повторной попыткой
                    wait_time = retry_delay * (attempt + 1)
                    print(f"Ждем {wait_time} секунд перед повторной попыткой...")
                    await asyncio.sleep(wait_time)
                last_error = e
                
            except Exception as e:
                print(f"Неожиданная ошибка: {e}")
                last_error = e
                break
        
        # Если все попытки неудачны
        raise last_error if last_error else Exception("Неизвестная ошибка")
    
    @staticmethod
    async def send_with_retry(bot, chat_id: int, text: str, **kwargs) -> bool:
        """Отправка сообщения с повторными попытками"""
        try:
            await TelegramSafe.safe_request(
                bot.send_message,
                chat_id=chat_id,
                text=text,
                **kwargs
            )
            return True
        except:
            return False
        
# FSM состояния
class ForecastStates(StatesGroup):
    waiting_period = State()
    waiting_file = State()
    waiting_scope = State()
    waiting_shops = State()
    waiting_gaps = State()
    waiting_categories = State()

# Глобальные данные пользователей
user_data_store = {}

# Инициализация данных пользователя
def init_user_data(user_id):
    if user_id not in user_data_store:
        user_data_store[user_id] = {
            "period": None,  # 30 или 60 дней
            "file": "default_05.06",  # default или custom
            "scope": None,  # all или specific
            "shops": [],  # список магазинов
            "gaps": [],  # список ГАПов
            "categories": [],  # список категорий
            "message_id": None  # ID сообщения для редактирования
        }
    return user_data_store[user_id]

# Удаление данных пользователя
def clear_user_data(user_id):
    if user_id in user_data_store:
        del user_data_store[user_id]

# Клавиатура для главного меню
def get_main_keyboard():
    keyboard = [
        [InlineKeyboardButton(text="Прогноз на 30 дней", callback_data="period_30")],
        [InlineKeyboardButton(text="Прогноз на 60 дней", callback_data="period_60")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

# Клавиатура выбора файла
def get_file_keyboard():
    keyboard = [
        [InlineKeyboardButton(text="Файл до 05.06", callback_data="file_default")],
        [InlineKeyboardButton(text="Прислать новый файл", callback_data="file_custom")],
        [InlineKeyboardButton(text="Назад", callback_data="back_to_main")],
        [InlineKeyboardButton(text="Далее", callback_data="next_to_scope")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

# Клавиатура выбора области
def get_scope_keyboard():
    keyboard = [
        [InlineKeyboardButton(text="Вся сеть", callback_data="scope_all")],
        [InlineKeyboardButton(text="Конкретные точки", callback_data="scope_specific")],
        [InlineKeyboardButton(text="Назад", callback_data="back_to_file")],
        [InlineKeyboardButton(text="Далее", callback_data="next_to_shops")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

# Клавиатура выбора магазинов
def get_shops_keyboard(user_shops=None):
    if user_shops is None:
        user_shops = []
    
    keyboard = []
    row = []
    
    for i in range(1, 17):
        shop_num = f"{i:02d}"
        shop_name = f"Магазин_{shop_num}"
        is_selected = shop_name in user_shops
        
        # Форматируем текст
        text = f"{'✅ ' if is_selected else ''}Магазин_{shop_num}"
        
        # Формируем callback
        if is_selected:
            callback = f"shop_deselect_{shop_num}"
        else:
            callback = f"shop_select_{shop_num}"
        
        row.append(InlineKeyboardButton(text=text, callback_data=callback))
        
        # 2 кнопки в строке
        if len(row) == 2:
            keyboard.append(row)
            row = []
    
    # Если осталась неполная строка
    if row:
        keyboard.append(row)
    
    # Кнопки управления
    keyboard.append([
        InlineKeyboardButton(text="Все магазины", callback_data="shop_all"),
        InlineKeyboardButton(text="Сбросить", callback_data="shop_reset")
    ])
    
    keyboard.append([
        InlineKeyboardButton(text="Назад", callback_data="back_to_scope"),
        InlineKeyboardButton(text="Далее", callback_data="next_to_gaps")
    ])
    
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

# Клавиатура выбора ГАПов
def get_gaps_keyboard(user_gaps=None):
    if user_gaps is None:
        user_gaps = []
    
    # Обезличенные ГАПы
    gap_names = {
        "gap_1": "ГруппаКатегорий_1",
        "gap_2": "ГруппаКатегорий_2",
        "gap_3": "ГруппаКатегорий_3",
        "gap_4": "ГруппаКатегорий_4",
        "gap_5": "ГруппаКатегорий_5",
        "gap_6": "ГруппаКатегорий_6"
    }
    
    keyboard = []
    for gap_id, gap_name in gap_names.items():
        is_selected = gap_id in user_gaps
        text = f"{'✅ ' if is_selected else ''}{gap_name}"
        callback = f"gap_select_{gap_id}" if not is_selected else f"gap_deselect_{gap_id}"
        keyboard.append([InlineKeyboardButton(text=text, callback_data=callback)])
    
    # Дополнительные опции
    keyboard.append([
        InlineKeyboardButton(text="Все ГАПы", callback_data="gap_all"),
        InlineKeyboardButton(text="Сбросить", callback_data="gap_reset")
    ])
    
    # Кнопки управления
    control_row = [
        InlineKeyboardButton(text="Назад", callback_data="back_to_shops"),
        InlineKeyboardButton(text="Далее", callback_data="next_to_categories")
    ]
    keyboard.append(control_row)
    
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

# Клавиатура выбора категорий
def get_categories_keyboard(user_gaps, user_categories=None):
    if user_categories is None:
        user_categories = []
    
    # Категории для каждого ГАПа
    gap_categories = {
        "gap_1": ["Кат1_ГК1", "Кат2_ГК1", "Кат3_ГК1", 
                 "Кат4_ГК1", "Кат5_ГК1", "Кат6_ГК1"],
        "gap_2": ["Кат1_ГК2"],
        "gap_3": ["Кат1_ГК3", "Кат2_ГК3", "Кат3_ГК3", 
                 "Кат4_ГК3", "Кат5_ГК3", "Кат6_ГК3",
                 "Кат7_ГК3", "Кат8_ГК3"],
        "gap_4": ["Кат1_ГК4", "Кат2_ГК4"],
        "gap_5": ["Кат1_ГК5"],
        "gap_6": ["Кат1_ГК6"]
    }
    
    # Названия ГАПов для заголовков
    gap_names = {
        "gap_1": "ГруппаКатегорий_1",
        "gap_2": "ГруппаКатегорий_2",
        "gap_3": "ГруппаКатегорий_3",
        "gap_4": "ГруппаКатегорий_4",
        "gap_5": "ГруппаКатегорий_5",
        "gap_6": "ГруппаКатегорий_6"
    }
    
    keyboard = []
    
    # Показываем категории только для выбранных ГАПов
    for gap_id in user_gaps:
        if gap_id in gap_categories:
            # Заголовок ГАПа
            gap_name = gap_names.get(gap_id, gap_id)
            keyboard.append([InlineKeyboardButton(text=f"{gap_name}:", callback_data="noop")])
        
            # Кнопки категорий этого ГАПа
            categories = gap_categories[gap_id]
            row = []
            
            for i, category in enumerate(categories):
                is_selected = category in user_categories
                
                button_text = f"{'✅ ' if is_selected else ''}{category}"
                
                # Формируем callback_data
                if is_selected:
                    callback_data = f"cat_deselect_{gap_id.split('_')[1]}_{i}"
                else:
                    callback_data = f"cat_select_{gap_id.split('_')[1]}_{i}"
                
                # Добавляем кнопку в ряд
                row.append(InlineKeyboardButton(text=button_text, callback_data=callback_data))
                
                # Перенос строки после 4 кнопок или если это последняя кнопка
                if len(row) == 4 or i == len(categories) - 1:
                    keyboard.append(row)
                    row = []
    
    # Дополнительные опции
    keyboard.append([
        InlineKeyboardButton(text="Все категории", callback_data="cat_all"),
        InlineKeyboardButton(text="Сбросить все", callback_data="cat_reset")
    ])
    
    # Кнопки управления
    keyboard.append([
        InlineKeyboardButton(text="Назад", callback_data="back_to_gaps"),
        InlineKeyboardButton(text="Начать прогноз", callback_data="start_forecast")
    ])
    
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

# Форматирование выбранных опций для отображения
def format_selection_text(user_data):
    text = "Ваш выбор:\n\n"
    
    # Период
    period_text = f"{user_data['period']} дней" if user_data['period'] else "не выбран"
    text += f"• Прогноз на: {period_text}\n"
    
    # Файл
    file_text = "Файл до 05.06" if user_data['file'] == "default_05.06" else "Новый файл"
    text += f"• Источник данных: {file_text}\n"
    
    # Область
    scope_text = "Вся сеть" if user_data['scope'] == "all" else "Конкретные точки" if user_data['scope'] else "не выбрана"
    text += f"• Область: {scope_text}\n"
    
    # Магазины
    if user_data['scope'] == "all":
        shops_text = "все магазины"
    elif user_data['shops']:
        # Проверяем не пустой ли список и не содержит ли "all"
        if user_data['shops'] == ["all"] or user_data['shops'] == []:
            shops_text = "не выбраны"
        else:
            shops_text = f"{len(user_data['shops'])} магазин(ов)"
    else:
        shops_text = "не выбраны"
    text += f"• Магазины: {shops_text}\n"
    
    # ГАПы
    if user_data['gaps']:
        gaps_text = f"{len(user_data['gaps'])} ГАП(ов)"
    else:
        gaps_text = "не выбраны"
    text += f"• Группы категорий: {gaps_text}\n"
    
    # Категории
    if user_data['categories']:
        cats_text = f"{len(user_data['categories'])} категорий"
    else:
        cats_text = "не выбраны"
    text += f"• Категории: {cats_text}\n"
    
    return text

# Команда /start
@dp.message(Command("start"))
async def start_command(message: types.Message):
    user_id = message.from_user.id
    init_user_data(user_id)
    
    # Удаляем предыдущее сообщение бота, если есть
    user_data = user_data_store[user_id]
    if user_data.get("message_id"):
        try:
            await bot.delete_message(chat_id=user_id, message_id=user_data["message_id"])
        except:
            pass
    
    # Отправляем новое сообщение
    welcome_text = (
        "Добро пожаловать в бот прогнозирования продаж!\n\n"
        "Выберите период для прогноза:"
    )
    
    msg = await message.answer(
        welcome_text,
        reply_markup=get_main_keyboard()
    )
    
    # Сохраняем ID сообщения для последующего редактирования
    user_data["message_id"] = msg.message_id

# Команда /end
@dp.message(Command("end"))
async def end_command(message: types.Message):
    user_id = message.from_user.id
    
    # Удаляем сообщение бота, если есть
    if user_id in user_data_store and user_data_store[user_id].get("message_id"):
        try:
            await bot.delete_message(
                chat_id=user_id, 
                message_id=user_data_store[user_id]["message_id"]
            )
        except:
            pass
    
    # Очищаем данные пользователя
    clear_user_data(user_id)
    
    await message.answer(
        "Спасибо за использование бота прогнозирования!\n"
        "До новых встреч!\n\n"
        "Для начала работы нажмите /start"
    )

# Обработка callback-запросов
@dp.callback_query(F.data.startswith("period_"))
async def handle_period(callback: types.CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    user_data = init_user_data(user_id)
    
    period = callback.data.split("_")[1]  # "30" или "60"
    user_data["period"] = int(period)
    
    # Обновляем сообщение
    selection_text = format_selection_text(user_data)
    new_text = f"{selection_text}\n\n Выберите источник данных:"
    
    await callback.message.edit_text(
        new_text,
        reply_markup=get_file_keyboard()
    )
    await callback.answer()

@dp.callback_query(F.data == "file_default")
async def handle_file_default(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    user_data = init_user_data(user_id)
    user_data["file"] = "default_05.06"
    
    # Обновляем сообщение
    selection_text = format_selection_text(user_data)
    new_text = f"{selection_text}\n\nВыберите область прогноза:"
    
    await callback.message.edit_text(
        new_text,
        reply_markup=get_scope_keyboard()
    )
    await callback.answer("Выбран файл до 05.06")

@dp.callback_query(F.data == "file_custom")
async def handle_file_custom(callback: types.CallbackQuery):
    await callback.answer("Временно недоступно", show_alert=True)

@dp.callback_query(F.data == "scope_all")
async def handle_scope_all(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    user_data = init_user_data(user_id)
    user_data["scope"] = "all"
    user_data["shops"] = ["all"]  # Устанавливаем специальное значение
    
    # Очищаем остальные выборы
    user_data["gaps"] = []
    user_data["categories"] = []
    
    selection_text = format_selection_text(user_data)
    new_text = f"{selection_text}\n\n Выберите группы категорий:"
    
    await callback.message.edit_text(
        new_text,
        reply_markup=get_gaps_keyboard(user_data["gaps"])
    )
    await callback.answer()

@dp.callback_query(F.data == "scope_specific")
async def handle_scope_specific(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    user_data = init_user_data(user_id)
    user_data["scope"] = "specific"
    user_data["shops"] = []  # ОЧИЩАЕМ при каждом выборе конкретных точек
    
    selection_text = format_selection_text(user_data)
    new_text = f"{selection_text}\n\n Выберите магазины (можно несколько):"
    
    await callback.message.edit_text(
        new_text,
        reply_markup=get_shops_keyboard(user_data["shops"])
    )
    await callback.answer()

# Обработка выбора магазинов
@dp.callback_query(F.data.startswith("shop_"))
async def handle_shop_selection(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    user_data = init_user_data(user_id)
    
    action = callback.data
    if action == "shop_all":
        # Выбрать все 16 магазинов
        user_data["shops"] = [f"Магазин_{i:02d}" for i in range(1, 17)]
        
    elif action == "shop_reset":
        # Сбросить выбор
        user_data["shops"] = []

    elif action.startswith("shop_select_"):
        shop_num = action.replace("shop_select_", "")
        shop_name = f"Магазин_{shop_num}"
        if shop_name not in user_data["shops"]:
            user_data["shops"].append(shop_name)
    elif action.startswith("shop_deselect_"):
        shop_num = action.replace("shop_deselect_", "")
        shop_name = f"Магазин_{shop_num}"
        if shop_name in user_data["shops"]:
            user_data["shops"].remove(shop_name)
    
    # Обновляем клавиатуру
    selection_text = format_selection_text(user_data)
    new_text = f"{selection_text}\n\n Выберите магазины (можно несколько):"
    
    await callback.message.edit_text(
        new_text,
        reply_markup=get_shops_keyboard(user_data["shops"])
    )
    await callback.answer()

# Обработка выбора ГАПов
@dp.callback_query(F.data.startswith("gap_"))
async def handle_gap_selection(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    user_data = init_user_data(user_id)
    
    action = callback.data
    
    if action == "gap_all":
        user_data["gaps"] = ["gap_1", "gap_2", "gap_3", "gap_4", "gap_5", "gap_6"]
    elif action == "gap_reset":
        user_data["gaps"] = []
    elif action.startswith("gap_select_"):
        gap_id = action.replace("gap_select_", "")
        if gap_id not in user_data["gaps"]:
            user_data["gaps"].append(gap_id)
    elif action.startswith("gap_deselect_"):
        gap_id = action.replace("gap_deselect_", "")
        if gap_id in user_data["gaps"]:
            user_data["gaps"].remove(gap_id)
    
    # Обновляем сообщение
    selection_text = format_selection_text(user_data)
    new_text = f"{selection_text}\n\n Выберите группы категорий:"
    
    await callback.message.edit_text(
        new_text,
        reply_markup=get_gaps_keyboard(user_data["gaps"])
    )
    await callback.answer()

@dp.callback_query(F.data == "ignore")
async def handle_ignore(callback: types.CallbackQuery):
    """Обработчик для игнорируемых кнопок (заголовки)"""
    await callback.answer()

@dp.callback_query(F.data.startswith("cat_"))
async def handle_category_selection(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    user_data = init_user_data(user_id)
    
    print(f"DEBUG: Нажата кнопка с callback_data: {callback.data}")
    
    # Игнорируем заголовки
    if callback.data == "ignore":
        await callback.answer()
        return
    
    action = callback.data
    
    # Маппинг категорий
    gap_categories = {
        "gap_1": ["Кат1_ГК1", "Кат2_ГК1", "Кат3_ГК1", 
                 "Кат4_ГК1", "Кат5_ГК1", "Кат6_ГК1"],
        "gap_2": ["Кат1_ГК2"],
        "gap_3": ["Кат1_ГК3", "Кат2_ГК3", "Кат3_ГК3", 
                 "Кат4_ГК3", "Кат5_ГК3", "Кат6_ГК3",
                 "Кат7_ГК3", "Кат8_ГК3"],
        "gap_4": ["Кат1_ГК4", "Кат2_ГК4"],
        "gap_5": ["Кат1_ГК5"],
        "gap_6": ["Кат1_ГК6"]
    }
    
    if action == "cat_all":
        # ВЫБРАТЬ ВСЕ КАТЕГОРИИ
        user_data["categories"] = []
        for gap_id in user_data["gaps"]:
            if gap_id in gap_categories:
                user_data["categories"].extend(gap_categories[gap_id])
        print(f"DEBUG: Выбраны ВСЕ категории: {user_data['categories']}")
        
    elif action == "cat_reset":
        # СБРОСИТЬ ВСЕ
        user_data["categories"] = []
        print("DEBUG: Категории сброшены")
        
    elif action.startswith("cat_select_"):
        # ВЫБРАТЬ КАТЕГОРИЮ
        try:
            # cat_select_1_0 -> ["cat", "select", "1", "0"]
            parts = action.split("_")
            if len(parts) >= 4:
                gap_num = parts[2]  # "1"
                cat_idx = int(parts[3])  # 0
                gap_id = f"gap_{gap_num}"  # "gap_1"
                
                if gap_id in gap_categories and 0 <= cat_idx < len(gap_categories[gap_id]):
                    category = gap_categories[gap_id][cat_idx]
                    if category not in user_data["categories"]:
                        user_data["categories"].append(category)
                        print(f"DEBUG: Добавлена категория: {category}")
        except Exception as e:
            print(f"DEBUG: Ошибка при выборе: {e}")
            await callback.answer("Ошибка выбора", show_alert=True)
            return
            
    elif action.startswith("cat_deselect_"):
        # ОТМЕНИТЬ ВЫБОР КАТЕГОРИИ
        try:
            parts = action.split("_")
            if len(parts) >= 4:
                gap_num = parts[2]
                cat_idx = int(parts[3])
                gap_id = f"gap_{gap_num}"
                
                if gap_id in gap_categories and 0 <= cat_idx < len(gap_categories[gap_id]):
                    category = gap_categories[gap_id][cat_idx]
                    if category in user_data["categories"]:
                        user_data["categories"].remove(category)
                        print(f"DEBUG: Удалена категория: {category}")
        except Exception as e:
            print(f"DEBUG: Ошибка при отмене: {e}")
            await callback.answer("Ошибка отмены", show_alert=True)
            return
    
    # ВСЕГДА обновляем сообщение
    try:
        selection_text = format_selection_text(user_data)
        new_text = f"{selection_text}\n\nВыберите категории товаров:"
        
        await callback.message.edit_text(
            new_text,
            reply_markup=get_categories_keyboard(user_data["gaps"], user_data["categories"])
        )
        print(f"DEBUG: Сообщение обновлено. Категории: {user_data['categories']}")
    except Exception as e:
        print(f"DEBUG: Ошибка при обновлении сообщения: {e}")
        # Если сообщение не изменилось, это нормально
    
    await callback.answer()

def reset_user_data_from_step(user_data, step):
    """
    Сбрасывает данные пользователя с определенного шага
    
    Args:
        user_data: данные пользователя
        step: с какого шага сбрасывать
            'main' - все
            'file' - все кроме периода
            'scope' - все кроме периода и файла
            'shops' - все кроме периода, файла, области
            'gaps' - все кроме периода, файла, области, магазинов
            'categories' - только категории
    """
    if step == 'main':
        user_data["period"] = None
        user_data["file"] = "default_05.06"
        user_data["scope"] = None
        user_data["shops"] = []
        user_data["gaps"] = []
        user_data["categories"] = []
    elif step == 'file':
        user_data["scope"] = None
        user_data["shops"] = []
        user_data["gaps"] = []
        user_data["categories"] = []
    elif step == 'scope':
        user_data["shops"] = []
        user_data["gaps"] = []
        user_data["categories"] = []
    elif step == 'shops':
        user_data["gaps"] = []
        user_data["categories"] = []
    elif step == 'gaps':
        user_data["categories"] = []

# Навигационные кнопки
@dp.callback_query(F.data == "back_to_main")
async def handle_back_to_main(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    user_data = init_user_data(user_id)
    reset_user_data_from_step(user_data, 'main')
    
    await callback.message.edit_text(
        "Добро пожаловать в бот прогнозирования продаж!\n\n"
        "Выберите период для прогноза:",
        reply_markup=get_main_keyboard()
    )
    await callback.answer()

@dp.callback_query(F.data == "back_to_file")
async def handle_back_to_file(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    user_data = init_user_data(user_id)
    reset_user_data_from_step(user_data, 'file')
    
    selection_text = format_selection_text(user_data)
    new_text = f"{selection_text}\n\n Выберите источник данных:"
    
    await callback.message.edit_text(
        new_text,
        reply_markup=get_file_keyboard()
    )
    await callback.answer()

@dp.callback_query(F.data == "back_to_scope")
async def handle_back_to_scope(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    user_data = init_user_data(user_id)
    reset_user_data_from_step(user_data, 'scope')
    
    selection_text = format_selection_text(user_data)
    new_text = f"{selection_text}\n\n Выберите область прогноза:"
    
    await callback.message.edit_text(
        new_text,
        reply_markup=get_scope_keyboard()
    )
    await callback.answer()

@dp.callback_query(F.data == "back_to_shops")
async def handle_back_to_shops(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    user_data = init_user_data(user_id)
    
    # Всегда сбрасываем магазины при возврате
    user_data["shops"] = []
    reset_user_data_from_step(user_data, 'shops')
    
    selection_text = format_selection_text(user_data)
    
    if user_data["scope"] == "specific":
        new_text = f"{selection_text}\n\n Выберите магазины (можно несколько):"
        await callback.message.edit_text(
            new_text,
            reply_markup=get_shops_keyboard(user_data["shops"])
        )
    else:
        new_text = f"{selection_text}\n\n Выберите область прогноза:"
        await callback.message.edit_text(
            new_text,
            reply_markup=get_scope_keyboard()
        )
    await callback.answer()

@dp.callback_query(F.data == "back_to_gaps")
async def handle_back_to_gaps(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    user_data = init_user_data(user_id)
    reset_user_data_from_step(user_data, 'gaps')
    
    selection_text = format_selection_text(user_data)
    new_text = f"{selection_text}\n\n Выберите группы категорий:"
    
    await callback.message.edit_text(
        new_text,
        reply_markup=get_gaps_keyboard(user_data["gaps"])
    )
    await callback.answer()

# Кнопки "Далее"
@dp.callback_query(F.data == "next_to_scope")
async def handle_next_to_scope(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    user_data = init_user_data(user_id)
    
    selection_text = format_selection_text(user_data)
    new_text = f"{selection_text}\n\n Выберите область прогноза:"
    
    await callback.message.edit_text(
        new_text,
        reply_markup=get_scope_keyboard()
    )
    await callback.answer()

@dp.callback_query(F.data == "next_to_shops")
async def handle_next_to_shops(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    user_data = init_user_data(user_id)
    
    if user_data["scope"] == "all":
        # Пропускаем выбор магазинов
        selection_text = format_selection_text(user_data)
        new_text = f"{selection_text}\n\n Выберите группы категорий:"
        
        await callback.message.edit_text(
            new_text,
            reply_markup=get_gaps_keyboard(user_data["gaps"])
        )
    else:
        selection_text = format_selection_text(user_data)
        new_text = f"{selection_text}\n\n Выберите магазины (можно несколько):"
        
        await callback.message.edit_text(
            new_text,
            reply_markup=get_shops_keyboard(user_data["shops"])
        )
    await callback.answer()

@dp.callback_query(F.data == "next_to_gaps")
async def handle_next_to_gaps(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    user_data = init_user_data(user_id)
    
    if user_data["scope"] == "specific" and not user_data["shops"]:
        await callback.answer("Пожалуйста, выберите хотя бы один магазин", show_alert=True)
        return
    
    selection_text = format_selection_text(user_data)
    new_text = f"{selection_text}\n\nВыберите группы категорий:"
    
    await callback.message.edit_text(
        new_text,
        reply_markup=get_gaps_keyboard(user_data["gaps"])
    )
    await callback.answer()

@dp.callback_query(F.data == "next_to_categories")
async def handle_next_to_categories(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    user_data = init_user_data(user_id)
    
    if not user_data["gaps"]:
        await callback.answer("Пожалуйста, выберите хотя бы одну группу категорий", show_alert=True)
        return
    
    selection_text = format_selection_text(user_data)
    new_text = f"{selection_text}\n\n Выберите категории товаров:"
    
    await callback.message.edit_text(
        new_text,
        reply_markup=get_categories_keyboard(user_data["gaps"], user_data["categories"])
    )
    await callback.answer()

@dp.callback_query(F.data == "retry_forecast")
async def handle_retry_forecast(callback: types.CallbackQuery):
    """Обработчик повторного запуска прогноза"""
    user_id = callback.from_user.id
    user_data = init_user_data(user_id)
    
    # Просто вызываем основной обработчик
    await handle_start_forecast(callback)

# Запуск прогноза
@dp.callback_query(F.data == "start_forecast")
async def handle_start_forecast(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    user_data = init_user_data(user_id)
    
    # 1. СРАЗУ пытаемся ответить на callback
    try:
        await callback.answer("Начинаю прогноз...")
    except Exception as e:
        logger.warning(f"Не удалось ответить на callback: {e}")
        # Пробуем отправить сообщение об ошибке сети
        try:
            await bot.send_message(
                callback.message.chat.id,
                "**Прошу прощения, проблемы с интернет-соединением...**\n\n"
                "Пожалуйста, подождите минуту, затем нажмите кнопку 'Начать прогноз' повторно."
            )
        except:
            pass  # Если не удалось и это, то совсем плохо
        return
    
    # Проверяем, что все необходимые данные выбраны
    if not user_data["period"]:
        try:
            await callback.message.edit_text("Не выбран период прогноза")
        except:
            try:
                await bot.send_message(callback.message.chat.id, "Не выбран период прогноза")
            except:
                pass
        return
    
    # Проверяем, что выбраны категории
    if not user_data["categories"]:
        selection_text = format_selection_text(user_data)
        try:
            await callback.message.edit_text(
                f"{selection_text}\n\nНе выбраны категории товаров.\nПожалуйста, выберите хотя бы одну категорию."
            )
        except:
            try:
                await bot.send_message(
                    callback.message.chat.id,
                    f"{selection_text}\n\nНе выбраны категории товаров"
                )
            except:
                pass
        return
    
    # 2. Обновляем сообщение
    selection_text = format_selection_text(user_data)
    
    try:
        await callback.message.edit_text(
            f"{selection_text}\n\nПодготовка данных..."
        )
    except Exception as e:
        logger.warning(f"Не удалось обновить сообщение: {e}")
        # Пробуем отправить новое сообщение
        try:
            await bot.send_message(
                callback.message.chat.id,
                f"{selection_text}\n\nПодготовка данных..."
            )
        except Exception as e2:
            logger.error(f"Критическая ошибка связи: {e2}")
            # Последняя попытка - сообщение о проблеме сети
            try:
                await bot.send_message(
                    callback.message.chat.id,
                    "**Прошу прощения, проблемы с интернет-соединением...**\n\n"
                    "Пожалуйста, подождите минуту, затем нажмите кнопку 'Начать прогноз' повторно."
                )
            except:
                pass
            return
    
    try:
        # 3. Конвертируем выбор пользователя
        real_data = data_processor.convert_user_selection_to_real(
            anon_shops=user_data["shops"] if user_data["scope"] == "specific" else [],
            anon_gaps=user_data["gaps"],
            anon_categories=user_data["categories"],
            scope=user_data["scope"]
        )
        
        # 4. Проверяем товары
        if not real_data["real_items"]:
            error_text = (
                f"{selection_text}\n\n"
                "Не найдены товары для выбранных категорий.\n"
                "Попробуйте выбрать другие категории или 'Всю сеть'."
            )
            try:
                await callback.message.edit_text(error_text)
            except:
                try:
                    await bot.send_message(callback.message.chat.id, error_text)
                except:
                    pass
            return
        
        # 5. Обновляем статус
        try:
            await callback.message.edit_text(
                f"{selection_text}\n\n"
                f"Запуск прогнозирования..."
            )
        except:
            try:
                await bot.send_message(
                    callback.message.chat.id,
                    f"Найдено {len(real_data['real_items'])} товаров\nЗапуск прогнозирования..."
                )
            except:
                pass
        
        # 6. Запускаем прогноз с ОГРАНИЧЕНИЕМ количества товаров
        logger.info(f"Запуск прогноза: horizon={user_data['period']}, shops={len(real_data['real_shops'])}, items={len(real_data['real_items'])}")
        
        # ОГРАНИЧИВАЕМ количество товаров до 50 для стабильности
        # limited_items = real_data['real_items']
        # print(f"Ограничиваем количество товаров с {len(real_data['real_items'])} до {len(limited_items)} для стабильности")
        
        loop = asyncio.get_event_loop()
        result_df = await loop.run_in_executor(
            None,
            lambda: run_forecast(
                horizon=user_data["period"],
                shops=real_data["real_shops"],
                items=real_data["real_items"],  # Используем ОГРАНИЧЕННЫЙ список
                batch_size=64  # Уменьшаем для стабильности
            )
        )
        
        if result_df.empty:
            try:
                await callback.message.edit_text(
                    f"{selection_text}\n\nНет данных для прогноза с выбранными фильтрами"
                )
            except:
                try:
                    await bot.send_message(
                        callback.message.chat.id,
                        "Нет данных для прогноза с выбранными фильтрами"
                    )
                except:
                    pass
            return
        
        # 7. ФИЛЬТРУЕМ по дате: оставляем только прогнозы начиная с 5 июня 2025
        cutoff_date = pd.Timestamp('2025-06-05')
        
        # Проверяем какие даты есть в прогнозе
        min_date = result_df['Дата'].min()
        max_date = result_df['Дата'].max()
        print(f"Прогноз с {min_date.date()} по {max_date.date()}")
        
        # Фильтруем только даты >= 5 июня 2025
        result_df = result_df[result_df['Дата'] >= cutoff_date]
        
        print(f"После фильтрации с {cutoff_date.date()}: {len(result_df)} записей")
        
        if result_df.empty:
            await callback.message.answer(
                f"Нет прогнозов на период с {cutoff_date.date()}.\n"
                f"Попробуйте выбрать другой период."
            )
            return
        
        # 8. Дополнительно: группируем по неделям для уменьшения объема
        # Можно добавить опционально
        if len(result_df) > 200000:  # Если все еще много записей
            print(f"Слишком много записей ({len(result_df)}), группируем по неделям...")
            
            # Создаем недельный индекс
            result_df['Неделя'] = result_df['Дата'].dt.to_period('W').dt.start_time
            
            # Группируем по неделям, магазинам и товарам
            grouped = result_df.groupby(
                ['Магазин', 'НоменклатураКод', 'Неделя']
            ).agg({
                'pred_Количество': 'sum'
            }).reset_index()
            
            # Переименовываем колонки обратно
            grouped = grouped.rename(columns={'Неделя': 'Дата'})
            result_df = grouped
            
            print(f"После группировки по неделям: {len(result_df)} записей")
    
        # 7. Обезличиваем результат
        try:
            await callback.message.edit_text(
                f"{selection_text}\n\n"
                f"Прогноз выполнен! Записей: {len(result_df):,}\n"
                f"Обезличивание..."
            )
        except:
            try:
                await bot.send_message(
                    callback.message.chat.id,
                    f"Прогноз выполнен! Обезличивание..."
                )
            except:
                pass
        
        anonymized_df = data_processor.anonymize_forecast_result(
            result_df,
            selected_anon_shops=user_data["shops"] if user_data["scope"] == "specific" else []
        )
        
        # 10. УДАЛЯЕМ старые записи из файла перед отправкой
        # (уже сделано на шаге 7, но на всякий случай еще раз)
        anonymized_df = anonymized_df[anonymized_df['Дата'] >= cutoff_date]

        # 8. Сохраняем и отправляем файл
        csv_filename = f"forecast_{user_data['period']}_days_{user_id}.csv"
        anonymized_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        file_size_mb = os.path.getsize(csv_filename) / (1024 * 1024)
        print(f"Размер файла: {file_size_mb:.2f} MB")
        
        # Если файл все еще огромный (>50MB), используем сжатие
        if file_size_mb > 50:
            gz_filename = csv_filename + '.gz'
            
            # Пересохраняем со сжатием
            anonymized_df.to_csv(gz_filename, index=False, encoding='utf-8-sig', compression='gzip')
            
            gz_size_mb = os.path.getsize(gz_filename) / (1024 * 1024)
            print(f"Сжатый размер: {gz_size_mb:.2f} MB (сжатие в {file_size_mb/gz_size_mb:.1f} раз)")
            
            # Отправляем сжатый файл
            document = FSInputFile(gz_filename, filename=f"прогноз_сжатый.gz")
            
            # Удаляем несжатый файл
            try:
                os.remove(csv_filename)
            except:
                pass
            
            csv_filename = gz_filename
        else:
            document = FSInputFile(csv_filename)
        
        caption = (
            f"Прогноз на {user_data['period']} дней готов!\n\n"
            f"Параметры:\n"
            f"• Магазины: {'вся сеть' if user_data['scope'] == 'all' else str(len(user_data['shops']))}\n"
            f"• Категории: {len(user_data['categories'])}\n"
            f"• Товаров: {anonymized_df['КодТовара'].nunique()}\n\n"
            f"Статистика:\n"
            f"• Всего записей: {len(anonymized_df):,}\n"
            f"• Период: {anonymized_df['Дата'].min().date()} - {anonymized_df['Дата'].max().date()}\n"
            f"• Средний прогноз: {anonymized_df['ПрогнозПродаж'].mean():.1f} ед./день\n"
            f"• Размер файла: {file_size_mb:.1f} MB"
        )
        
        try:
            await TelegramSafe.safe_request(
                callback.message.answer_document,
                document=document,
                caption=caption,
                read_timeout=60,  # 60 секунд на чтение
                write_timeout=60,  # 60 секунд на запись
                connect_timeout=30  # 30 секунд на соединение
            )
            
            print(f"Файл успешно отправлен!")
            
        except Exception as e:
            logger.error(f"Ошибка при отправке файла: {e}")
            
            # Пробуем отправить без документа, только статистику
            await callback.message.answer(
                f"{caption}\n\n"
                f"Не удалось отправить файл ({file_size_mb:.1f} MB).\n"
                f"Файл сохранен локально: {csv_filename}"
            )
        
            # Удаляем временный файл
            try:
                os.remove(csv_filename)
            except:
                pass
            
        except Exception as e:
            logger.error(f"Ошибка при отправке файла: {e}")
            try:
                await callback.message.edit_text(
                    f"{selection_text}\n\n"
                    f"Прогноз выполнен, но не удалось отправить файл.\n"
                    f"Ошибка: {str(e)[:100]}"
                )
            except:
                try:
                    await bot.send_message(
                        callback.message.chat.id,
                        f"Прогноз выполнен, но не удалось отправить файл."
                    )
                except:
                    pass
        
        # 9. Завершаем
        try:
            await callback.message.answer(
                "**Прогноз успешно завершен!**\n\n"
                "Хотите сделать новый прогноз? Нажмите /start\n"
                "Для выхода нажмите /end"
            )
        except:
            pass
        
    except Exception as e:
        logger.error(f"Ошибка при прогнозе: {e}\n{traceback.format_exc()}")
        
        # Определяем тип ошибки
        error_msg = str(e)
        if "out of bounds" in error_msg or "index" in error_msg:
            error_text = (
                f"{selection_text}\n\n"
                "**Ошибка при обработке данных**\n\n"
                "Проблема: слишком много товаров с недостаточной историей.\n"
                "Система автоматически ограничила выборку.\n\n"
                "Пожалуйста, нажмите кнопку 'Начать прогноз' повторно."
            )
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            error_text = (
                f"{selection_text}\n\n"
                "**Прошу прощения, проблемы с интернет-соединением...**\n\n"
                "Пожалуйста, подождите минуту, затем нажмите кнопку 'Начать прогноз' повторно."
            )
        else:
            error_text = f"{selection_text}\n\nОшибка при выполнении прогноза:\n{error_msg[:150]}..."
        
        try:
            await callback.message.edit_text(error_text)
        except:
            try:
                await bot.send_message(callback.message.chat.id, error_text)
            except:
                pass
    
    # Очищаем данные пользователя
    clear_user_data(user_id)

# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    print("Бот запущен...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен")

# D:\python_envs\forecasting_bot\Scripts\activate