import json
import logging
import os
from datetime import datetime, timedelta

import psycopg2
import speech_recognition as sr
from pydub import AudioSegment
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# --------- Logging setup ---------
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --------- Load tokenizer from JSON ---------
with open('tokenizer_clean.json', 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)

# --------- Load model ---------
model = load_model(r'toxic_classifier_lstm.h5')
MAX_LEN = 100

# --------- PostgreSQL setup ---------
try:
    conn = psycopg2.connect(
    dbname="toxicusers",         # your database name on Render
    user="toxicusers_user",                   # your Render DB user
    password="P2fnJbxr45IByiahCi3Smbco7I1FKgxO",           # your Render DB password
    host="dpg-d121a0juibrs73end2rg-a.render.com",  # your Render DB host
    port="5432",
    sslmode='require'                          # important for Render PostgreSQL
)
    cursor = conn.cursor()
    logger.info("PostgreSQL connected successfully.")
except Exception as e:
    logger.error(f"Failed to connect to PostgreSQL: {e}")
    raise e

# --------- Database functions ---------
def get_user_record(user_id):
    try:
        cursor.execute("SELECT toxic_count, blocked_until FROM toxic_users WHERE user_id = %s", (user_id,))
        return cursor.fetchone()
    except Exception as e:
        logger.error(f"DB error in get_user_record: {e}")
        return None

def update_user_record(user_id, username, toxic_count, blocked_until):
    try:
        cursor.execute("""
            INSERT INTO toxic_users (user_id, username, toxic_count, blocked_until)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id)
            DO UPDATE SET toxic_count = EXCLUDED.toxic_count,
                          blocked_until = EXCLUDED.blocked_until,
                          username = EXCLUDED.username
        """, (user_id, username, toxic_count, blocked_until))
        conn.commit()
    except Exception as e:
        logger.error(f"DB error in update_user_record: {e}")

# --------- Toxicity detection ---------
def detect_toxicity(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    prediction = model.predict(padded)[0][0]
    logger.info(f"Toxicity prediction for '{text}': {prediction:.4f}")
    return 1 if prediction > 0.5 else 0  # 1 = toxic, 0 = non-toxic

# --------- Simple explainability ---------
TOXIC_KEYWORDS = {'hate', 'stupid', 'idiot', 'dumb', 'kill', 'trash', 'ugly'}

def explain_toxicity(text):
    words = text.split()
    highlighted = []
    for w in words:
        if w.lower() in TOXIC_KEYWORDS:
            highlighted.append(f"**{w}**")  # Bold toxic words
        else:
            highlighted.append(w)
    return ' '.join(highlighted)

# --------- Speech to text ---------
def speech_to_text(file_path):
    wav_path = file_path.replace('.ogg', '.wav')
    try:
        sound = AudioSegment.from_file(file_path)
        sound.export(wav_path, format="wav")
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return ""

    r = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
    except sr.UnknownValueError:
        logger.warning("Google Speech Recognition could not understand audio")
        text = ""
    except sr.RequestError as e:
        logger.error(f"Could not request results from Google Speech Recognition service; {e}")
        text = ""
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
    return text

# --------- Telegram bot handlers ---------
def start(update: Update, context: CallbackContext):
    update.message.reply_text("👋 Welcome! Send me text or voice messages and I'll check for toxicity.")

def handle_text(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    username = update.message.from_user.username or update.message.from_user.first_name
    text = update.message.text
    now = datetime.now()
    chat_id = update.message.chat_id
    message_id = update.message.message_id

    record = get_user_record(user_id)
    toxic_count = 0
    blocked_until = None

    if record:
        toxic_count, blocked_until = record
        if blocked_until and now < blocked_until:
            try:
                context.bot.delete_message(chat_id=chat_id, message_id=message_id)
            except BadRequest:
                pass
            update.message.reply_text(f"⛔ You're blocked until {blocked_until.strftime('%Y-%m-%d %H:%M:%S')}")
            return

    prediction = detect_toxicity(text)

    if prediction == 1:
        toxic_count += 1

        try:
            context.bot.delete_message(chat_id=chat_id, message_id=message_id)
        except BadRequest:
            pass

        explanation = explain_toxicity(text)
        update.message.reply_text(f"🚫 Toxic message removed. Explanation:\n{explanation}")

        if toxic_count == 8:
            update.message.reply_text(f"⚠️ Warning @{username}: 8 toxic messages detected.")
        elif toxic_count >= 10:
            blocked_until = now + timedelta(days=2)
            update.message.reply_text(f"⛔ You are blocked for 2 days until {blocked_until.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        update.message.reply_text("✅ Not Toxic")

    update_user_record(user_id, username, toxic_count, blocked_until)

def handle_voice(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    username = update.message.from_user.username or update.message.from_user.first_name
    now = datetime.now()
    chat_id = update.message.chat_id
    message_id = update.message.message_id

    record = get_user_record(user_id)
    toxic_count = 0
    blocked_until = None

    if record:
        toxic_count, blocked_until = record
        if blocked_until and now < blocked_until:
            try:
                context.bot.delete_message(chat_id=chat_id, message_id=message_id)
            except BadRequest:
                pass
            update.message.reply_text(f"⛔ You're blocked until {blocked_until.strftime('%Y-%m-%d %H:%M:%S')}")
            return

    voice_file = update.message.voice.get_file()
    ogg_path = f"{user_id}_{message_id}.ogg"
    voice_file.download(ogg_path)

    text = speech_to_text(ogg_path)

    try:
        context.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except BadRequest:
        pass

    if os.path.exists(ogg_path):
        os.remove(ogg_path)

    if not text:
        update.message.reply_text("❓ Could not understand your voice message.")
        return

    prediction = detect_toxicity(text)

    if prediction == 1:
        toxic_count += 1

        explanation = explain_toxicity(text)
        update.message.reply_text(f"🚫 Toxic voice message removed. Explanation:\n{explanation}")

        if toxic_count == 8:
            update.message.reply_text(f"⚠️ Warning @{username}: 8 toxic messages detected.")
        elif toxic_count >= 10:
            blocked_until = now + timedelta(days=2)
            update.message.reply_text(f"⛔ You are blocked for 2 days until {blocked_until.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        update.message.reply_text(f"✅ Voice message transcription: {text}\n✅ Not Toxic")

    update_user_record(user_id, username, toxic_count, blocked_until)

# --------- Main ---------
def main():
    TOKEN = "8117423761:AAHajq68kw5uGvm9KVhyEK937DKvsOxPNLo"  # Replace with your token

    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text))
    dp.add_handler(MessageHandler(Filters.voice, handle_voice))

    logger.info("Bot started. Waiting for messages...")
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
