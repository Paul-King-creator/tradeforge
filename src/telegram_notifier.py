"""
Telegram Notifier für TradeForge Reports
Sendet automatisch Reports an Telegram
"""
import os
import requests
from datetime import datetime

class TelegramNotifier:
    """Sendet TradeForge Reports via Telegram Bot"""
    
    def __init__(self, bot_token: str = None, chat_id: int = None):
        # Token aus Umgebungsvariable oder Parameter
        self.bot_token = bot_token or os.getenv('TRADEFORGE_BOT_TOKEN')
        self.chat_id = chat_id
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if self.enabled:
            print(f"✅ Telegram Notifier aktiviert (Chat: {chat_id})")
        else:
            print("⚠️  Telegram Notifier deaktiviert (kein Token/Chat ID)")
    
    def send_report(self, report_text: str, photo_path: str = None) -> bool:
        """Sendet Report als Telegram Nachricht"""
        if not self.enabled:
            return False
        
        try:
            # Text aufteilen wenn zu lang (max 4096 Zeichen)
            max_length = 4000  # Safety margin
            
            if len(report_text) > max_length:
                # In chunks aufteilen
                chunks = []
                current_chunk = ""
                
                for line in report_text.split('\n'):
                    if len(current_chunk) + len(line) + 1 > max_length:
                        chunks.append(current_chunk)
                        current_chunk = line + '\n'
                    else:
                        current_chunk += line + '\n'
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Ersten Chunk senden
                self._send_message(chunks[0])
                
                # Restliche Chunks senden
                for i, chunk in enumerate(chunks[1:], 2):
                    self._send_message(f"... (Fortsetzung {i}/{len(chunks)})\n\n{chunk}")
            else:
                self._send_message(report_text)
            
            # Optional: Chart senden
            if photo_path and os.path.exists(photo_path):
                self._send_photo(photo_path)
            
            return True
            
        except Exception as e:
            print(f"❌ Telegram Send Error: {e}")
            return False
    
    def _send_message(self, text: str) -> bool:
        """Sendet einzelne Nachricht"""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        payload = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True
        }
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            return True
        else:
            print(f"⚠️  Telegram API Error: {response.status_code} - {response.text}")
            return False
    
    def _send_photo(self, photo_path: str) -> bool:
        """Sendet Foto/Chart"""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        
        with open(photo_path, 'rb') as photo:
            files = {'photo': photo}
            data = {'chat_id': self.chat_id}
            
            response = requests.post(url, files=files, data=data, timeout=30)
            
            if response.status_code != 200:
                print(f"⚠️  Telegram Photo Error: {response.status_code}")
                return False
        
        return True
    
    def send_simple_message(self, message: str) -> bool:
        """Sendet einfache Status-Nachricht"""
        if not self.enabled:
            return False
        
        try:
            # Emoji escaping für HTML
            message = message.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            return self._send_message(message)
        except Exception as e:
            print(f"❌ Telegram Error: {e}")
            return False


# Einfache Funktion für schnellen Test
def send_test_message(bot_token: str, chat_id: int) -> bool:
    """Testet Telegram Verbindung"""
    notifier = TelegramNotifier(bot_token, chat_id)
    return notifier.send_simple_message(
        "🤖 TradeForge Telegram Test\n"
        "✅ Verbindung erfolgreich!\n"
        "Du wirst jetzt tägliche Reports erhalten."
    )