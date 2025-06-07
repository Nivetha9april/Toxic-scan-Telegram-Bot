import pickle
import numpy as np
import psycopg2
from datetime import datetime, timedelta
import speech_recognition as sr
from pydub import AudioSegment
import os
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from telegram.error import BadRequest
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------- Load tokenizer and model ---------
with open(r'tokenizer_clean.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model(r'toxic_classifier_lstm.h5')
MAX_LEN = 100

# --------- PostgreSQL setup ---------
conn = psycopg2.connect(
    dbname="toxic_comment_detection",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

def get_user_record(user_id):
    cursor.execute("SELECT toxic_count, blocked_until FROM toxic_users WHERE user_id = %s", (user_id,))
    return cursor.fetchone()

def update_user_record(user_id, username, toxic_count, blocked_until):
    cursor.execute("""
        INSERT INTO toxic_users (user_id, username, toxic_count, blocked_until)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (user_id)
        DO UPDATE SET toxic_count = EXCLUDED.toxic_count,
                      blocked_until = EXCLUDED.blocked_until,
                      username = EXCLUDED.username
    """, (user_id, username, toxic_count, blocked_until))
    conn.commit()

# --------- Toxicity detection ---------
def detect_toxicity(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    prediction = model.predict(padded)[0][0]
    return 1 if prediction > 0.5 else 0  # 1 = toxic, 0 = non-toxic

# --------- Simple explainability: highlight toxic keywords ---------
# For demo, define a small list of toxic keywords your model likely flags (expand as needed)
TOXIC_KEYWORDS = {'hate', 'stupid', 'idiot', 'dumb', 'kill', 'trash', 'ugly'}

def explain_toxicity(text):
    words = text.split()
    highlighted = []
    for w in words:
        # Simple lowercase check if word is toxic keyword
        if w.lower() in TOXIC_KEYWORDS:
            highlighted.append(f"**{w}**")  # Bold toxic words
        else:
            highlighted.append(w)
    return ' '.join(highlighted)

# --------- Speech to text from voice message ---------
def speech_to_text(file_path):
    # Convert ogg/opus to wav
    wav_path = file_path.replace('.ogg', '.wav')
    sound = AudioSegment.from_file(file_path)
    sound.export(wav_path, format="wav")

    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = r.record(source)

    try:
        text = r.recognize_google(audio)
    except sr.UnknownValueError:
        text = ""
    except sr.RequestError:
        text = ""

    # Clean up wav file
    os.remove(wav_path)
    return text

# --------- Bot handlers ---------
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
            # Delete message from blocked user
            try:
                context.bot.delete_message(chat_id=chat_id, message_id=message_id)
            except BadRequest:
                pass
            update.message.reply_text(f"⛔ You're blocked until {blocked_until.strftime('%Y-%m-%d %H:%M:%S')}")
            return

    prediction = detect_toxicity(text)

    if prediction == 1:
        toxic_count += 1

        # Delete toxic message
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

    # Download voice file
    voice_file = update.message.voice.get_file()
    ogg_path = f"{user_id}_{message_id}.ogg"
    voice_file.download(ogg_path)

    # Convert speech to text
    text = speech_to_text(ogg_path)

    # Remove the original voice message (optional)
    try:
        context.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except BadRequest:
        pass

    # Clean up ogg file
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
    TOKEN = '8117423761:AAHajq68kw5uGvm9KVhyEK937DKvsOxPNLo'  # Replace here

    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text))
    dp.add_handler(MessageHandler(Filters.voice, handle_voice))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
