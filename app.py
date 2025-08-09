from flask import Flask, request, jsonify, render_template
import imaplib
import email
from email.header import decode_header
from flask_cors import CORS
import openai
import os
import time
import logging
from dotenv import load_dotenv
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from supabase import create_client
import hashlib

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)
CORS(app)

def generate_stable_email_id(sender: str, subject: str, date: str) -> str:
    raw = f"{sender}|{subject}|{date}"
    return hashlib.sha256(raw.encode()).hexdigest()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Constants
EMAIL = "alihamzasultanacc3@gmail.com"
APP_PASSWORD = "ijwd wmln bcbd vsql"
VALID_CATEGORIES = ['urgent', 'support', 'sales', 'complaint', 'newsletter', 'other']
BATCH_SIZE = 10

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)

# Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/summarize', methods=['POST'])
def summarize_email():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        email_body = data.get("body", "")
        if not email_body.strip():
            return jsonify({"error": "Email body content is required"}), 400

        subject = data.get("title", "")
        sender = data.get("from", "")
        email_id = data.get("id")  # Must be passed from frontend

        prompt = f"""You're an email assistant. Read the following email and make a concise summary of it.

        From: {sender}
        Subject: {subject}

        Email Content:
        {email_body}
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful email assistant. Reply professionally and concisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )

        summary = response.choices[0].message.content.strip()

        # ✅ Update Supabase with summary
        if email_id:
            supabase.table("emails").update({"summary": summary}).eq("id", email_id).execute()
            logger.info(f"✅ Summary saved to Supabase for email ID {email_id}")
        else:
            logger.warning("⚠️ No email ID provided to update summary.")

        return jsonify({"reply": summary})

    except Exception as e:
        logger.error(f"Error in /api/summarize: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to generate summary", "details": str(e)}), 500



@app.route("/api/reply", methods=["POST"])
def generate_reply():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        email_body = data.get("body", "")
        if not email_body.strip():
            return jsonify({"error": "Email body content is required"}), 400

        email_body = data.get("body", "")
        subject = data.get("title", "")
        sender = data.get("from", "")

        prompt = f"""You're an email assistant. Read the following email and generate a concise, professional reply.

        From: {sender}
        Subject: {subject}

        Email Content:
        {email_body}

        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful email assistant. Reply professionally and concisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )

        reply = response.choices[0].message.content.strip()
        return jsonify({"reply": reply})

    except Exception as e:
        logger.error(f"Error in /api/reply: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to generate reply", "details": str(e)}), 500


@app.route("/api/reply-all", methods=["POST"])
def handle_reply_all():
    try:
        emails = request.get_json()
        if not isinstance(emails, list):
            return jsonify({"error": "Invalid data format"}), 400

        all_responses = []

        for i, email_data in enumerate(emails, start=1):
            print(f"\n--- Processing Email {i} ---")
            for key, value in email_data.items():
                print(f"{key}: {value}")

            # 1. Generate reply using internal call to /api/reply
            with app.test_client() as client:
                reply_resp = client.post("/api/reply", json=email_data)
                reply_json = reply_resp.get_json()

            if reply_resp.status_code != 200 or "reply" not in reply_json:
                print(f"❌ Failed to generate reply for email {i}")
                all_responses.append({
                    "email_index": i,
                    "status": "failed",
                    "error": reply_json.get("error", "Unknown error")
                })
                continue

            ai_reply = reply_json["reply"]
            print(f"\n✅ AI Reply:\n{ai_reply}")

            # 2. Send the reply via internal call to /api/send
            # 2. Send the reply via internal call to /api/send
            send_payload = {
                "to": email_data.get("from"),
                "reply": ai_reply,
                "subject": f"Re: {email_data.get('title', 'No Subject')}"
            }

            with app.test_client() as client:
                send_resp = client.post("/api/send", json=send_payload)
                send_json = send_resp.get_json()

            print(f"📤 Email Send Response: {send_json}")

            # 3. Delete email from Supabase if sending was successful
            if send_resp.status_code == 200:
                try:
                    email_id = email_data.get("id")
                    if email_id:
                        supabase.table("emails").delete().eq("id", email_id).execute()
                        logger.info(f"🗑️ Deleted email ID {email_id} from Supabase after replying.")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to delete email ID {email_id}: {e}")

            all_responses.append({
                "email_index": i,
                "to": send_payload["to"],
                "subject": send_payload["subject"],
                "reply": ai_reply,
                "send_status": send_json.get("status", "failed"),
                "send_response": send_json
            })


        return jsonify({
            "status": "completed",
            "total_emails": len(emails),
            "results": all_responses
        })

    except Exception as e:
        logger.exception("Error in /api/reply-all")
        return jsonify({"error": str(e)}), 500


@app.route("/api/send", methods=["POST"])
def send_email():
    data = request.get_json()
    recipient = data.get("to")
    reply_content = data.get("reply")
    subject = data.get("subject", "Re: Your email")

    if not recipient or not reply_content:
        return jsonify({"error": "Recipient and reply are required"}), 400

    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(reply_content, 'plain'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL, APP_PASSWORD)
            server.send_message(msg)
            logger.info(f"Email successfully sent to {recipient}")

        return jsonify({"status": "sent"})

    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP Authentication Error - Check your credentials")
        return jsonify({"error": "SMTP authentication failed"}), 401
    except Exception as e:
        logger.error(f"SMTP Error: {str(e)}")
        return jsonify({"error": f"Failed to send email: {str(e)}"}), 500

def connect_to_gmail():
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    imap.login(EMAIL, APP_PASSWORD)
    return imap

def mark_as_seen(email_uid):
    try:
        imap = connect_to_gmail()
        imap.select("inbox")
        imap.store(str(email_uid), '+FLAGS', '\\Seen')
        imap.logout()
        logger.info(f"Email ID {email_uid} marked as seen.")
    except Exception as e:
        logger.warning(f"Failed to mark email ID {email_uid} as seen: {e}")

def classify_email(email_id: int, email_content: str):
    try:
        # Just return classification from OpenAI without storing in Supabase
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Categorize this email with one word: urgent, support, sales, complaint, newsletter, or other"},
                {"role": "user", "content": email_content[:8000]}
            ],
            temperature=0.3,
            max_tokens=10
        )
        category = response.choices[0].message.content.lower().strip()
        return category if category in VALID_CATEGORIES[:-1] else 'other'
    except Exception as e:
        logger.error(f"Error processing email ID {email_id}: {e}")
        return 'other'
import threading

def check_and_upload_emails():
    while True:
        try:
            imap = connect_to_gmail()
            imap.select("inbox")

            status, messages = imap.search(None, "UNSEEN")
            if status != "OK":
                time.sleep(10)
                continue

            all_email_ids = messages[0].split()
            sorted_email_ids = sorted(all_email_ids, key=lambda x: int(x), reverse=True)
            recent_email_ids = sorted_email_ids[:10]

            for num in recent_email_ids:
                try:
                    res, msg = imap.fetch(num, "(RFC822 FLAGS)")
                    if res != "OK" or not msg:
                        continue

                    for response in msg:
                        if isinstance(response, tuple):
                            msg_data = email.message_from_bytes(response[1])
                            
                            subject_raw = msg_data.get("Subject", "")
                            subject, encoding = decode_header(subject_raw)[0]
                            if isinstance(subject, bytes):
                                subject = subject.decode(encoding or "utf-8", errors="ignore")

                            date = msg_data["Date"]
                            from_ = msg_data.get("From", "")
                            sender_name, sender_email = email.utils.parseaddr(from_)


                            body = ""
                            if msg_data.is_multipart():
                                for part in msg_data.walk():
                                    content_type = part.get_content_type()
                                    content_disposition = str(part.get("Content-Disposition"))
                                    if content_type == "text/plain" and "attachment" not in content_disposition:
                                        try:
                                            body = part.get_payload(decode=True).decode()
                                            break
                                        except Exception:
                                            continue
                            else:
                                try:
                                    body = msg_data.get_payload(decode=True).decode()
                                except Exception:
                                    body = ""

                            if not body.strip():
                                logger.info(f"⛔ Skipped email from {sender_email} - Empty body")
                                continue

                            unique_email_id = generate_stable_email_id(sender_email, subject, date)
                            category = classify_email(0, body)


                            # Insert only if not already present
                            exists = supabase.table("emails").select("id").eq("id", unique_email_id).execute()
                            sender_image = get_gravatar_url(sender_email)
                            if not exists.data:
                                supabase.table("emails").insert({
                                    "id": unique_email_id,
                                    "from": sender_email,
                                    "title": subject,
                                    "date": date,
                                    "read": "unread",
                                    "replied": "unknown",
                                    "classification": category,
                                    "body": body.strip() if body else "No content",
                                    "email": sender_email,
                                    "sender_name": sender_name,
                                    "sender_image": sender_image,
                                }).execute()
                                logger.info(f"✅ Uploaded email from {sender_email} to Supabase")
                            else:
                                logger.info(f"⏭️ Email already exists: {unique_email_id}")

                except Exception as e:
                    logger.error(f"❌ Error processing unseen email: {e}")

            imap.logout()

        except Exception as e:
            logger.exception("🔥 Exception in background email checker")

        time.sleep(10)
def get_gravatar_url(email):
    hash = hashlib.md5(email.strip().lower().encode()).hexdigest()
    return f"https://www.gravatar.com/avatar/{hash}?d=identicon"

@app.route("/api/emails", methods=["GET"])
def fetch_emails():
    try:
        imap = connect_to_gmail()
        imap.select("inbox")

        status, messages = imap.search(None, "UNSEEN")
        if status != "OK":
            return jsonify({"error": "Unable to fetch emails"}), 500

        all_email_ids = messages[0].split()
        sorted_email_ids = sorted(all_email_ids, key=lambda x: int(x), reverse=True)
        recent_email_ids = sorted_email_ids[:10]

        for num in recent_email_ids:
            try:
                res, msg = imap.fetch(num, "(RFC822 FLAGS)")
                if res != "OK":
                    continue

                for response in msg:
                    if isinstance(response, tuple):
                        flags = msg[0][0] if msg and msg[0] else b""
                        is_read = b"\\Seen" in flags

                        msg_data = email.message_from_bytes(response[1])

                        # Subject
                        subject_raw = msg_data.get("Subject", "")
                        subject, encoding = decode_header(subject_raw)[0]
                        if isinstance(subject, bytes):
                            subject = subject.decode(encoding or "utf-8", errors="ignore")

                        # From & Date
                        date = msg_data.get("Date", "")
                        from_ = msg_data.get("From", "")
                        sender_name, sender_email = email.utils.parseaddr(from_)


                        # Body
                        body = ""
                        if msg_data.is_multipart():
                            for part in msg_data.walk():
                                content_type = part.get_content_type()
                                content_disposition = str(part.get("Content-Disposition"))
                                if content_type == "text/plain" and "attachment" not in content_disposition:
                                    try:
                                        body = part.get_payload(decode=True).decode()
                                        break
                                    except Exception:
                                        continue
                        else:
                            try:
                                body = msg_data.get_payload(decode=True).decode()
                            except Exception:
                                body = ""

                        if not body.strip():
                            logger.info(f"⛔ Skipped email from {sender_email} - Empty body")
                            continue

                        # Now that we have all values
                        unique_email_id = generate_stable_email_id(sender_email, subject, date)
                        category = classify_email(int(num.decode()), body)

                        # Upload if not already exists
                        exists = supabase.table("emails").select("id").eq("id", unique_email_id).execute()
                        sender_image = get_gravatar_url(sender_email)
                        if not exists.data:
                            supabase.table("emails").insert({
                                "id": unique_email_id,
                                "from": sender_email,
                                "title": subject,
                                "date": date,
                                "read": "read" if is_read else "unread",
                                "replied": "unknown",
                                "classification": category,
                                "body": body.strip(),
                                "email": sender_email,
                                "sender_image": sender_image,
                                "sender_name": sender_name,
                            }).execute()
                            logger.info(f"✅ Uploaded email from {sender_email} to Supabase")
                           

                        else:
                            logger.info(f"⏭️ Email already exists: {unique_email_id}")

            except Exception as e:
                logger.error(f"Error processing email ID {num}: {e}")
                continue

        imap.logout()

        # Fetch all stored emails
        response = supabase.table("emails").select("*").order("date", desc=True).limit(100).execute()
        return jsonify(response.data)

    except Exception as e:
        logger.exception("Failed to fetch emails")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    threading.Thread(target=check_and_upload_emails, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
