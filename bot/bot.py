import asyncio
import logging
import numpy as np
import torch
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from transformers import BertTokenizer, BertForSequenceClassification

# Загрузка базы данных ответов
data = np.load('database.npz', allow_pickle=True)
questions = data['answer_class']
answers = data['answers']

# Инициализация модели
model = BertForSequenceClassification.from_pretrained("./fine-tune-bert", num_labels=30)
tokenizer = BertTokenizer.from_pretrained("./fine-tune-bert")

logging.basicConfig(level=logging.INFO)
bot = Bot(token="6511904496:AAEKLGUscMmnUFO-Zfa6I1zlvduZLzSrgH0")
dp = Dispatcher()

@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Привет! Для того чтобы задать вопрос, используй команду /ask.")

@dp.message(Command("ask"))
async def ask_question(message: types.Message):
    input_text = message.text.replace('/ask', '').strip()
    def bert_apply(text, model, tokenizer):
        model.eval()
        t = tokenizer(text, truncation=True, return_tensors='pt', max_length=50)
        with torch.no_grad():
            model_output = model(**{k: v.to(model.device) for k, v in t.items()})
        return model_output
    logits = bert_apply(input_text, model, tokenizer).logits
    predictions = logits.argmax().item()
    max_logit = logits.max().item()
    
    best_answer = None
    if (max_logit > 3.5):
        for i in range(len(questions)):
            if predictions == i:
                best_answer = answers[i]
                break
    else:
        best_answer = "Не понял Вашего вопроса. Пожалуйста, свяжитесь с куратором."
    await message.answer(best_answer)
    
@dp.message()
async def answer(message: types.Message):
    await message.answer("Для того чтобы задать вопрос, используй команду /ask.")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())