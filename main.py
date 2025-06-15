import streamlit as st
import cv2
import time
import numpy as np
from PIL import Image
import tempfile
import os
from detector import DroneDetector
from utils import get_class_colors, get_class_icons

st.set_page_config(
    page_title="Sistem Deteksi Drone",
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
    .upload-section {
        background: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def is_running_on_cloud():
    """Detect if running on cloud platform"""
    # Check common cloud environment variables
    cloud_indicators = [
        'STREAMLIT_SHARING',
        'RAILWAY_PROJECT_ID',
        'HEROKU_APP_NAME',
        'RENDER_SERVICE_ID',
        'FLY_APP_NAME'
    ]
    return any(os.getenv(indicator) for indicator in cloud_indicators)

def process_image(image, detector, confidence):
    """Process single image for detection"""
    # Convert PIL to OpenCV format
    image_array = np.array(image)
    if len(image_array.shape) == 3:
        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        image_cv = image_array
    
    # Run detection
    results = detector.detect(image_cv, confidence)
    annotated_frame, detections = detector.process_results(results)
    
    return annotated_frame, detections

def process_video(video_path, detector, confidence, progress_bar, frame_placeholder):
    """Process video file for detection"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    all_detections = []
    frame_count = 0
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, 
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect(frame, confidence)
            annotated_frame, detections = detector.process_results(results)
            
            if annotated_frame is not None:
                # Convert back to BGR for video writing
                annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                out.write(annotated_bgr)
                
                # Show current frame
                if frame_count % 10 == 0:  # Update display every 10 frames
                    frame_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)
            
            all_detections.extend(detections)
            frame_count += 1
            
            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            
    finally:
        cap.release()
        out.release()
    
    return temp_output.name, all_detections

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🚁 Sistem Deteksi Drone</h1>
        <p>Deteksi Pesawat, Burung, Drone, dan Helikopter dengan YOLOv8</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize detector
    detector = DroneDetector()
    
    if not detector.is_model_loaded():
        st.error("❌ Model YOLO tidak dapat dimuat. Pastikan file model tersedia.")
        st.info("💡 Untuk deployment cloud, pastikan file model ada di repository dan path benar.")
        st.stop()

    class_colors = get_class_colors()
    class_icons = get_class_icons()

    # Sidebar
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
    
    # Telegram notifications
    st.sidebar.markdown("### 📱 Notifikasi Telegram")
    enable_telegram = st.sidebar.checkbox("Aktifkan Notifikasi Drone", value=False)
    
    bot_token = ""
    chat_id = ""
    if enable_telegram:
        bot_token = st.sidebar.text_input("Bot Token", type="password", help="Token bot Telegram Anda")
        chat_id = st.sidebar.text_input("Chat ID", help="ID chat Telegram Anda")

        if st.sidebar.button("🧪 Test Notifikasi"):
            if bot_token and chat_id:
                try:
                    from telegram_notifier import TelegramNotifier
                    notifier = TelegramNotifier(bot_token, chat_id)
                    success = notifier.send_test_message()
                    if success:
                        st.sidebar.success("✅ Test berhasil!")
                    else:
                        st.sidebar.error("❌ Test gagal, periksa token/chat ID")
                except ImportError:
                    st.sidebar.error("❌ Module telegram_notifier tidak tersedia")
            else:
                st.sidebar.error("Masukkan Bot Token dan Chat ID")

    # Check if running on cloud
    is_cloud = is_running_on_cloud()
    
    # Main content
    if is_cloud:
        st.info("🌐 Terdeteksi berjalan di cloud. Mode kamera real-time tidak tersedia. Gunakan upload file.")
        
        # File upload mode
        st.markdown("""
        <div class="upload-section">
            <h3>📁 Upload File untuk Deteksi</h3>
            <p>Upload gambar atau video untuk dianalisis</p>
        </div>
        """, unsafe_allow_html=True)
        
        upload_type = st.radio("Pilih tipe file:", ["Gambar", "Video"], horizontal=True)
        
        if upload_type == "Gambar":
            uploaded_file = st.file_uploader(
                "Upload gambar", 
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Format yang didukung: JPG, JPEG, PNG, BMP"
            )
            
            if uploaded_file is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Gambar Asli")
                    image = Image.open(uploaded_file)
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.subheader("Hasil Deteksi")
                    with st.spinner("Memproses deteksi..."):
                        annotated_frame, detections = process_image(image, detector, confidence)
                        
                        if annotated_frame is not None:
                            st.image(annotated_frame, channels="RGB", use_column_width=True)
                            
                            # Show detection results
                            if detections:
                                st.success(f"✅ Terdeteksi {len(detections)} objek")
                                
                                # Send Telegram notification if drone detected
                                drone_count = len([d for d in detections if d['class_name'] == 'Drone'])
                                if drone_count > 0 and enable_telegram and bot_token and chat_id:
                                    try:
                                        from telegram_notifier import TelegramNotifier
                                        notifier = TelegramNotifier(bot_token, chat_id)
                                        success = notifier.send_drone_alert(drone_count)
                                        if success:
                                            st.success("🚨 Notifikasi drone terkirim!")
                                    except:
                                        st.warning("⚠️ Gagal mengirim notifikasi Telegram")
                                
                                # Detection summary
                                summary = detector.get_detection_summary(detections)
                                for class_name, count in summary.items():
                                    icon = class_icons.get(class_name, '❓')
                                    st.write(f"{icon} **{class_name}**: {count} objek")
                            else:
                                st.info("ℹ️ Tidak ada objek terdeteksi")
                        else:
                            st.error("❌ Gagal memproses gambar")
        
        else:  # Video
            uploaded_file = st.file_uploader(
                "Upload video", 
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Format yang didukung: MP4, AVI, MOV, MKV"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name
                
                st.subheader("Hasil Deteksi Video")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    frame_placeholder = st.empty()
                    progress_bar = st.progress(0)
                
                with col2:
                    st.subheader("Status")
                    status_placeholder = st.empty()
                
                try:
                    with st.spinner("Memproses video..."):
                        status_placeholder.info("🔄 Memproses video...")
                        output_path, all_detections = process_video(
                            temp_path, detector, confidence, progress_bar, frame_placeholder
                        )
                        
                        if all_detections:
                            status_placeholder.success(f"✅ Selesai! Total deteksi: {len(all_detections)}")
                            
                            # Send Telegram notification if drones detected
                            drone_detections = [d for d in all_detections if d['class_name'] == 'Drone']
                            if drone_detections and enable_telegram and bot_token and chat_id:
                                try:
                                    from telegram_notifier import TelegramNotifier
                                    notifier = TelegramNotifier(bot_token, chat_id)
                                    success = notifier.send_drone_alert(len(drone_detections))
                                    if success:
                                        st.success("🚨 Notifikasi drone terkirim!")
                                except:
                                    st.warning("⚠️ Gagal mengirim notifikasi Telegram")
                            
                            # Show summary
                            summary = detector.get_detection_summary(all_detections)
                            st.subheader("📊 Ringkasan Deteksi")
                            for class_name, count in summary.items():
                                if count > 0:
                                    icon = class_icons.get(class_name, '❓')
                                    st.write(f"{icon} **{class_name}**: {count} deteksi")
                            
                            # Provide download link for processed video
                            with open(output_path, 'rb') as f:
                                st.download_button(
                                    label="📥 Download Video Hasil Deteksi",
                                    data=f.read(),
                                    file_name="detected_video.mp4",
                                    mime="video/mp4"
                                )
                        else:
                            status_placeholder.info("ℹ️ Tidak ada objek terdeteksi dalam video")
                            
                except Exception as e:
                    status_placeholder.error(f"❌ Error memproses video: {str(e)}")
                finally:
                    # Cleanup
                    try:
                        os.unlink(temp_path)
                        if 'output_path' in locals():
                            os.unlink(output_path)
                    except:
                        pass
    
    else:
        # Local mode with camera support
        st.warning("🖥️ Mode lokal terdeteksi. Fitur kamera real-time tersedia.")
        
        # Camera selection
        camera_index = st.sidebar.selectbox("Pilih Kamera", [0, 1, 2], 
                                          help="Coba indeks kamera berbeda jika default tidak berfungsi")
        
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

        # Camera detection logic (original code)
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
        try:
            from telegram_notifier import TelegramNotifier
            telegram_notifier = TelegramNotifier(bot_token, chat_id)
        except ImportError:
            st.sidebar.error("❌ Module telegram_notifier tidak tersedia")

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

            if annotated_frame is not None:
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