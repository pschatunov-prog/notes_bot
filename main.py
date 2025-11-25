import os
import logging
import tempfile
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters, ConversationHandler
from dotenv import load_dotenv

from db import Database
from llm import summarize_and_tag, semantic_search, analyze_notes, transcribe_audio

# Load environment variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Database instance
db = Database()

# States for ConversationHandler (if needed, but for MVP we can use stateless or simple states)
# For this MVP, we'll process messages directly for adding notes, and use commands for others.
# However, to avoid confusion between "Search query" and "New note", we can use a simple state machine or just slash commands.
# User request: "Add notes: User sends text or voice".
# Let's make it so:
# - Text/Voice (no command) -> Add Note
# - /search <query> -> Search
# - /analyze -> Analyze

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to your AI Note Bot!\n\n"
        "Commands:\n"
        "- Send text or voice to add a note.\n"
        "- /search <query>: Search your notes.\n"
        "- /analyze: Analyze all your notes.\n"
        "- /help: Show this help message."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text
    
    if not text:
        return

    # Process as new note
    status_msg = await update.message.reply_text("Processing note...")
    
    # 1. Summarize and Tag
    result = await summarize_and_tag(text)
    summary = result.get("summary", "No summary")
    tags = ", ".join(result.get("tags", []))
    
    # 2. Save to DB
    db.add_note(user_id, text, summary, tags)
    
    await status_msg.edit_text(
        f"✅ Note saved!\n\n"
        f"📝 **Summary**: {summary}\n"
        f"🏷 **Tags**: {tags}"
    , parse_mode="Markdown")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    voice = update.message.voice
    
    status_msg = await update.message.reply_text("🎧 Processing voice message...")
    
    # Download voice file
    file = await context.bot.get_file(voice.file_id)
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_audio:
        temp_path = temp_audio.name
    
    await file.download_to_drive(temp_path)
    
    # Transcribe
    await status_msg.edit_text("📝 Transcribing...")
    text = await transcribe_audio(temp_path)
    
    # Clean up temp file
    os.remove(temp_path)
    
    if not text:
        await status_msg.edit_text("❌ Could not transcribe audio.")
        return

    await status_msg.edit_text(f"🗣 **Transcribed**: {text}\n\n🤖 Generating summary...")
    
    # Summarize and Tag
    result = await summarize_and_tag(text)
    summary = result.get("summary", "No summary")
    tags = ", ".join(result.get("tags", []))
    
    # Save to DB
    db.add_note(user_id, text, summary, tags)
    
    await update.message.reply_text(
        f"✅ Note saved!\n\n"
        f"📝 **Summary**: {summary}\n"
        f"🏷 **Tags**: {tags}"
    , parse_mode="Markdown")

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    query = " ".join(context.args)
    
    if not query:
        await update.message.reply_text("Usage: /search <query>")
        return

    status_msg = await update.message.reply_text("🔍 Searching...")
    
    notes = db.get_notes(user_id)
    result = await semantic_search(query, notes)
    
    await status_msg.edit_text(result)

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    status_msg = await update.message.reply_text("🧠 Analyzing your notes...")
    
    notes_text = db.get_all_notes_text(user_id)
    result = await analyze_notes(notes_text)
    
    await status_msg.edit_text(result)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logging.error(msg="Exception while handling an update:", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text("An error occurred while processing your request.")

if __name__ == '__main__':
    if not TELEGRAM_TOKEN:
        print("Error: TELEGRAM_TOKEN not found in environment variables.")
        exit(1)

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    
    # Message handlers
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    
    # Error handler
    application.add_error_handler(error_handler)

    print("Bot is running...")
    application.run_polling()
