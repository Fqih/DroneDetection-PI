import requests
import json
from datetime import datetime
import streamlit as st

class TelegramNotifier:
    """Class untuk mengirim notifikasi ke Telegram"""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            st.error(f"Error sending Telegram message: {e}")
            return False

    def send_drone_alert(self, drone_count: int = 1) -> bool:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        urgency = "⚠️ PERINGATAN" if drone_count == 1 else "🚨 URGENT"
        emoji = "🛸" * min(drone_count, 5)

        message = f"""
{urgency} - DRONE TERDETEKSI!

{emoji} <b>⚠️LAPOR ICIK BOSS ADA DRONE NIHH⚠️</b>
📊 Jumlah Drone: <b>{drone_count}</b>
🕐 Waktu: <b>{current_time}</b>
📍 Status: <b>Aktif Terdeteksi</b>

🔍 Mohon periksa area sekitar untuk memastikan keamanan

#DroneAlert #Security 
        """.strip()

        return self.send_message(message)

    def send_test_message(self) -> bool:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"""
✅ <b>Test Notifikasi Berhasil!</b>

🤖 Bot Telegram Connected
🕐 Waktu Test: <b>{current_time}</b>
📡 Status: <b>Siap Mendeteksi</b>

Sistem notifikasi drone siap digunakan!
        """.strip()

        return self.send_message(message)

    def send_session_start(self) -> bool:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"""
🟢 <b>Sistem Deteksi Dimulai</b>

📹 Kamera: <b>Aktif</b>
🕐 Mulai: <b>{current_time}</b>
🎯 Mode: <b>Real-time Detection</b>

Sistem pemantauan drone sedang aktif...
        """.strip()
        return self.send_message(message)

    def send_session_end(self, total_detections: dict) -> bool:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_parts = []
        icons = {'Pesawat': '✈️', 'Burung': '🐦', 'Drone': '🛸', 'Helikopter': '🚁'}
        total_count = sum(total_detections.values())

        for class_name, count in total_detections.items():
            if count > 0:
                icon = icons.get(class_name, '❓')
                summary_parts.append(f"{icon} {class_name}: {count}")

        summary_text = "\n".join(summary_parts) if summary_parts else "Tidak ada deteksi"

        message = f"""
🔴 <b>Sesi Deteksi Berakhir</b>

🕐 Berakhir: <b>{current_time}</b>
📊 Total Deteksi: <b>{total_count}</b>

<b>Ringkasan Deteksi:</b>
{summary_text}

Sistem deteksi telah dihentikan.
        """.strip()

        return self.send_message(message)

    def test_connection(self) -> tuple[bool, str]:
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                bot_info = response.json()
                if bot_info.get('ok'):
                    bot_name = bot_info['result'].get('first_name', 'Unknown')
                    return True, f"Connected to bot: {bot_name}"
                else:
                    return False, "Invalid bot token"
            else:
                return False, f"HTTP Error: {response.status_code}"
        except requests.exceptions.Timeout:
            return False, "Connection timeout"
        except requests.exceptions.ConnectionError:
            return False, "Connection error"
        except Exception as e:
            return False, f"Error: {str(e)}"

# ========== Streamlit UI ==========

st.title("🛸 Notifikasi Deteksi Drone ke Telegram")

with st.sidebar:
    st.header("🔐 Konfigurasi Telegram")
    bot_token = st.text_input("Bot Token", type="password")
    chat_id = st.text_input("Chat ID")

if bot_token and chat_id:
    notifier = TelegramNotifier(bot_token, chat_id)

    if st.button("🔌 Tes Koneksi"):
        success, msg = notifier.test_connection()
        st.success(msg) if success else st.error(msg)

    if st.button("✅ Kirim Tes Notifikasi"):
        if notifier.send_test_message():
            st.success("Pesan tes berhasil dikirim!")
        else:
            st.error("Gagal mengirim pesan tes.")

    if st.button("🚨 Kirim Peringatan Drone"):
        if notifier.send_drone_alert(drone_count=3):
            st.success("Peringatan drone berhasil dikirim!")
        else:
            st.error("Gagal mengirim peringatan.")
else:
    st.warning("Masukkan Bot Token dan Chat ID terlebih dahulu di sidebar.")
