import streamlit as st
import cv2
import time
from detector import DroneDetector
from utils import get_class_colors, get_class_icons

st.set_page_config(
    page_title="Sistem Deteksi Drone Real-time",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fps-counter {
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
    .detection-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🚁 Sistem Deteksi Drone Real-time</h1>
        <p>Deteksi real-time Pesawat, Burung, Drone, dan Helikopter dengan YOLOv8</p>
    </div>
    """, unsafe_allow_html=True)

    detector = DroneDetector()
    class_colors = get_class_colors()
    class_icons = get_class_icons()

    st.sidebar.title("Pengaturan Deteksi")
    st.sidebar.markdown("### 🎯 Kelas Deteksi")
    for class_name, icon in class_icons.items():
        color = class_colors[class_name]
        st.sidebar.markdown(f"""
        <div style="background-color: {color}; padding: 0.5rem; margin: 0.2rem 0; border-radius: 5px; color: white;">
            {icon} {class_name}
        </div>
        """, unsafe_allow_html=True)

    confidence = st.sidebar.slider("Batas Confidence", 0.1, 1.0, 0.5, 0.1)
    camera_index = st.sidebar.selectbox("Pilih Kamera", [0, 1, 2], help="Coba indeks kamera berbeda jika default tidak berfungsi")
    st.sidebar.markdown("### 📱 Notifikasi Telegram")
    enable_telegram = st.sidebar.checkbox("Aktifkan Notifikasi Drone", value=False)

    bot_token = ""
    chat_id = ""
    if enable_telegram:
        bot_token = st.sidebar.text_input("Bot Token", type="password", help="Token bot Telegram Anda")
        chat_id = st.sidebar.text_input("Chat ID", help="ID chat Telegram Anda")

        if st.sidebar.button("🧪 Test Notifikasi"):
            if bot_token and chat_id:
                from telegram_notifier import TelegramNotifier
                notifier = TelegramNotifier(bot_token, chat_id)
                success = notifier.send_test_message()
                if success:
                    st.sidebar.success("✅ Test berhasil!")
                else:
                    st.sidebar.error("❌ Test gagal, periksa token/chat ID")
            else:
                st.sidebar.error("Masukkan Bot Token dan Chat ID")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_detection = st.button("▶️ Mulai", type="primary", use_container_width=True)
    with col2:
        stop_detection = st.button("⏹️ Berhenti", use_container_width=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Live Camera Feed")
        frame_placeholder = st.empty()
        fps_placeholder = st.empty()

    with col2:
        st.subheader("Status & Info")
        status_placeholder = st.empty()
        total_detection_placeholder = st.empty()

    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False
    if 'cap' not in st.session_state:
        st.session_state.cap = None

    if start_detection and not st.session_state.detection_active:
        try:
            cap = cv2.VideoCapture(camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)

            if not cap.isOpened():
                st.error(f"❌ Tidak dapat mengakses kamera {camera_index}")
                st.info("💡 Coba solusi berikut:\n- Periksa izin kamera\n- Coba indeks kamera berbeda\n- Pastikan tidak ada aplikasi lain yang menggunakan kamera")
            else:
                st.session_state.cap = cap
                st.session_state.detection_active = True
                status_placeholder.success("🟢 Kamera aktif - Deteksi berjalan")

        except Exception as e:
            st.error(f"Error inisialisasi kamera: {e}")

    if stop_detection and st.session_state.detection_active:
        if st.session_state.cap:
            st.session_state.cap.release()
        st.session_state.detection_active = False
        st.session_state.cap = None
        status_placeholder.info("🔴 Deteksi dihentikan")
        frame_placeholder.empty()
        fps_placeholder.empty()
        total_detection_placeholder.empty()
        st.sidebar.empty()

    if st.session_state.detection_active and st.session_state.cap:
        run_detection_loop(
            st.session_state.cap,
            detector,
            confidence,
            frame_placeholder,
            fps_placeholder,
            total_detection_placeholder,
            status_placeholder,
            enable_telegram,
            bot_token,
            chat_id
        )

def run_detection_loop(cap, detector, confidence, frame_placeholder, fps_placeholder, total_detection_placeholder, status_placeholder, enable_telegram, bot_token, chat_id):
    telegram_notifier = None
    if enable_telegram and bot_token and chat_id:
        from telegram_notifier import TelegramNotifier
        telegram_notifier = TelegramNotifier(bot_token, chat_id)

    fps_counter = 0
    fps_start_time = time.time()
    fps_display = 0
    total_detection_stats = {'Pesawat': 0, 'Burung': 0, 'Drone': 0, 'Helikopter': 0}
    last_drone_notification = 0
    notification_cooldown = 10

    try:
        while st.session_state.detection_active:
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("❌ Gagal membaca dari kamera")
                break

            frame_start_time = time.time()
            results = detector.detect(frame, confidence)
            annotated_frame, detections = detector.process_results(results)

            drone_detected = any(d['class_name'] == 'Drone' for d in detections)
            for detection in detections:
                class_name = detection['class_name']
                if class_name in total_detection_stats:
                    total_detection_stats[class_name] += 1

            current_time = time.time()
            if (drone_detected and telegram_notifier and 
                current_time - last_drone_notification > notification_cooldown):
                drone_count = sum(1 for d in detections if d['class_name'] == 'Drone')
                success = telegram_notifier.send_drone_alert(drone_count)
                if success:
                    last_drone_notification = current_time
                    st.sidebar.success("🚨 Notifikasi drone terkirim!")

            frame_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)

            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps_display = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()

            fps_placeholder.markdown(f"""
            <div class="fps-counter">
                📊 FPS: {fps_display:.1f}
            </div>
            """, unsafe_allow_html=True)

            frame_time = time.time() - frame_start_time
            if frame_time < 0.015:
                time.sleep(0.015 - frame_time)

    except Exception as e:
        status_placeholder.error(f"❌ Error dalam deteksi: {e}")
    finally:
        if st.session_state.detection_active:
            st.session_state.detection_active = False

if __name__ == "__main__":
    main()
