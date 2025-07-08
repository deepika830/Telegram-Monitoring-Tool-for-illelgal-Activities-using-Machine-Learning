# Further enhanced scam detection with keyword triggering and improved thresholds
from telethon import TelegramClient, events
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import json
import hashlib
import logging
import re
from datetime import datetime
from config import api_id, api_hash
from pathlib import Path
import csv
from telethon.tl import types 
LOG_PATH = Path("threat_logs.csv")
###################################################################################
# Configure logging
logging.basicConfig(
    filename='scam_detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
###################################################################################
# Critical scam keywords that should trigger higher scrutiny
SCAM_KEYWORDS = {
    'financial_scam': ['fund', 'money', 'transfer', 'account', 'bank', 'payment', 'refund', 'tax', 'ssn', 'credit card'],
    'urgency_threats': ['arrest', 'warrant', 'police', 'urgent', 'immediately', 'threat', 'extortion', 'legal action', 'hacked', 'virus'],
    'credential_phishing': ['login', 'password', 'verify', 'account', 'security', 'update', 'reset', 'authentication', 'suspension'],
    'fake_offers': ['free', 'offer', 'prize', 'won', 'reward', 'gift', 'guaranteed', 'investment', 'secret method'],
    'otherscams': ['covid', 'relief', 'government', 'support', 'aid', 'fbi', 'irs', 'blackmail', 'seizure', 'romance']
}
SUSPICIOUS_LINK_PATTERNS = ["http", "https", ".com", ".net", ".xyz", ".ru", ".cn", "://", "t.me/", "bit.ly", "tinyurl"]
SUSPICIOUS_EMOJIS = [
    "💰", "💸", "🤑", "💵", "💳", "💲",
    "🎁", "🏆", "🎉", "🎊",
    "⏰", "⚠️", "🚨", "😱", "😰", "😨", "❗", "‼️", "❌", "⛔",
    "🔒", "🔐", "🛡️", "🔑",

]
NORMAL_PHRASES = {
    'hi', 'hello', 'hey', 'how are you', 'good morning', 
    'good afternoon', 'good night', 'whats up', 'how are u guys'
}
#############################################################################################
# Load model and tokenizer
try:
    model_path = "scam_type_detector"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise
#######################################################################################################
# Load category mapping
try:
    with open("category_map.json") as f:
        category_map = json.load(f)
    logger.info(f"Loaded {len(category_map)} scam categories")
except Exception as e:
    logger.error(f"Category map loading failed: {str(e)}")
    raise

client = TelegramClient('test', api_id, api_hash)
client.start()
print("✅ Connected successfully")


#Enhanced keyword detection with partial matches
def contains_scam_keywords(text):
    """Check for scam keywords with partial matches and obfuscations"""
    text_lower = text.lower()
    for scam_type, keywords in SCAM_KEYWORDS.items():
        for keyword in keywords:
            # Match keywords with word boundaries or common obfuscations
            pattern = re.compile(
                rf'(^|\W){re.escape(keyword)}[s]?($|\W)|'  # Basic match with plurals
                rf'{re.escape(keyword.replace(" ", "/s?"))}',  # Handle spaces
                re.IGNORECASE
            )
            if pattern.search(text_lower):
                return scam_type
    return None
#######################################################################################################33
def classify_message(text, sender=None):
    try:
        keyword_match = contains_scam_keywords(text)

        # Step 1: Short message skip logic
        if not keyword_match and len(text.split()) < 1:
            text_lower = text.lower()

            has_link = any(link_part in text_lower for link_part in SUSPICIOUS_LINK_PATTERNS)
            has_emoji = any(emoji in text for emoji in SUSPICIOUS_EMOJIS)

            if not (has_link or has_emoji):
                return 0, 0.0, None  # Skip unimportant short message

        # Step 2: Tokenize and classify
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        # Step 3: Adjust confidence using keyword_match
        if keyword_match:
            if keyword_match == category_map[str(pred)]:
                confidence = min(confidence * 1.3, 1.0)
            else:
                confidence = max(confidence, 0.75)

        # Step 4: Optional audit logging
        if sender:
            log_model_decision(sender,text, outputs.logits, probs, pred, confidence)

        return pred, confidence, probs

    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        return 0, 0.0, None
###########################################################################################################
AUDIT_LOG_PATH = Path("model_audit_log.csv")
def log_model_decision(sender,text, logits, probs, pred, confidence):
    class_labels = [
        "safe", "financial_scam", "credential_phishing",
        "fake_offers", "urgency_threats", "otherscams"
    ]
    sender= sender.id

    write_header = not AUDIT_LOG_PATH.exists()
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "Sender ID","Text", "Predicted Class", "Confidence",
                "Probabilities", "Logits"
            ])
        writer.writerow([
            text[:300],  # Truncate for readability
            class_labels[pred] if pred < len(class_labels) else f"Unknown({pred})",
            round(confidence, 4),
            [round(p, 4) for p in probs[0].tolist()],
            [round(l, 4) for l in logits[0].tolist()]
        ])
###############################################################################################
###############################################################################################
def log_to_csv(sender, chat, chat_title, scam_type, confidence, text):
    links = re.findall(r'(https?://\S+|www\.\S+|t\.me/\S+)', text)
    fields = [
        datetime.now().isoformat(),
        scam_type,
        round(confidence, 3),
        chat_title,
        sender.username if sender and sender.username else sender.id,
        text[:200],
        ";".join(links) if links else "none",  # New: Store all URLs
        sender.id,
        chat.id if hasattr(chat, 'id') else 'private',
        'user' if isinstance(chat, types.User) else 
        'group' if isinstance(chat, types.Chat) else 
        'channel' if isinstance(chat, types.Channel) else 'unknown'
    ]
    
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "type", "confidence", "chat", "user", "message", 
                "urls", "user_id", "chat_id", "chat_type"  # Updated header
            ])
        writer.writerow(fields)
##################################################################################################
#Listens for new incoming messages.
@client.on(events.NewMessage)
async def handler(event):
    try:
        text = event.message.text or ""
        text = text.strip()
        if not text:
            print("not text")

        sender = await event.get_sender()
        chat = await event.get_chat()
        chat_title = getattr(chat, "title", f"Private Chat with {sender.username or sender.id}")

        # 🚨 Predict Scam Type
        pred, confidence, probs = classify_message(text, sender)
        scam_type = category_map[str(pred)]

        # 🧠 Special handling for urgency
        is_urgency_threat = scam_type == 'urgency_threats' and contains_scam_keywords(text)

        # 🔽 Dynamic thresholds by scam type
        min_confidence = {
            'normal': 0.0,
            'financial_scam': 0.48,
            'credential_phishing': 0.52,
            'fake_offers': 0.55,
            'urgency_threats': 0.60,
            'otherscams': 0.75
        }.get(scam_type, 0.75)

        # 🧠 High-confidence match
        if pred != 0 and confidence > min_confidence:
            prob_percent = confidence * 100
            log_to_csv(sender, chat, chat_title, scam_type, confidence, text)

            report = (
                f"🚨 **Suspicious Message Detected**\n"
                f"**Type**: {scam_type.upper()} ({prob_percent:.1f}%)\n"
                f"**Chat**: {chat_title}\n"
                f"**User**: {sender.username or sender.id}\n"
                f"**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"**Message**: {text[:300]}{'...' if len(text) > 300 else ''}"
            )

            logger.info(f"SCAM DETECTED: {scam_type} (Confidence: {prob_percent:.1f}%)")
            print("\n" + "!"*60)
            print(report)
            print("!"*60)

            # Optional alert message to the user
            if confidence > 0.6 or is_urgency_threat:
                hashed_id = hashlib.sha256(str(sender.id).encode()).hexdigest()[:12]
                verification_msg = (
                    f"⚠️ **URGENT SECURITY ALERT** ⚠️\n\n"
                    f"Potential {scam_type.replace('_', ' ')} detected:\n"
                    f"'{text[:100]}...'\n\n"
                    f"Please verify your identity:\n"
                    f"🔗 [Click for secure verification](http://clickme.replit.app)"
                )
                try:
                    await client.send_message(
                        sender.id,
                        verification_msg,
                        parse_mode='md',
                        link_preview=False
                    )
                    logger.info(f"Sent verification to {sender.id}")
                except Exception as e:
                    logger.error(f"Failed to warn {sender.id}: {str(e)}")
        else:
            log_msg = f"[Normal] {chat_title} | {sender.username or sender.id}: {text[:100]}..."
            logger.debug(log_msg)
            print(log_msg)

    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
#############################################################################################
if __name__ == "__main__":
    client.start()
    print("[*] Advanced Scam Detection Active")
    print(f"Monitoring for: {', '.join([v for k,v in category_map.items() if k != '0'])}")
    client.run_until_disconnected()