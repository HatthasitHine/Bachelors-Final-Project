# 1
from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
import time
from werkzeug.utils import secure_filename
import tensorflow as tf
import librosa
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from collections import Counter
import shutil

# 2
import json
import subprocess
import wave
from pydub import AudioSegment
from google.cloud import speech
from pythainlp.util import Trie
from spleeter.separator import Separator
from pythainlp.tokenize import word_tokenize
import gc

#=============================================================================================================

# ปิดคำเตือน ที่ไม่ได้ช่วยอะไร....
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ปิด oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#=============================================================================================================

app = Flask(__name__)

# uploads เอาไว้รับไฟล์ที่อัพโหลดมาแล้วเอาไปเช็คคำร้อง
app.config['UPLOAD_FOLDER'] = 'uploads'

# uploads_check เอาไว้ก๊อปไฟล์ที่อัพโหลดมาแล้วเอาไปเช็คทำนอง
app.config['CHECK_FOLDER'] = 'uploads_check'

# เอาไว้เช็คนามสกุลไฟล์
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav'}

# ฟังก์ชั่นสร้างโฟลเดอร์สำหรับรันโปรแกรม/รับไฟล์ ถ้าไม่มีก็สร้างขึ้นมา
def create_folders():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['CHECK_FOLDER']):
        os.makedirs(app.config['CHECK_FOLDER'])

# ฟังก์ชันตรวจสอบว่าไฟล์เป็นประเภทที่อนุญาต(mp3/wav)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

#=============================================================================================================

# โหลดโมเดลและตัวปรับสเกล
model = load_model('Music_6_Genre_Full_Song_CNN_Model.h5')

with open('Label_Encoder_6_Genre_Full_Song.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('Scaler_6_Genre_Full_Song.pkl', 'rb') as file:
    scaler = pickle.load(file)

#=============================================================================================================


# แปลชื่อประเภทเพลง
genre_translation = {
    "Jazz": "แจ๊ส",
    "HipHop": "แรป",
    "Classical": "คลาสสิก",
    "Rock": "ร็อก",
    "Metal": "เมทัล",
    "Country": "ลูกทุ่ง"
}

# ฟังก์ชันแปลชื่อประเภทเพลง
def translate_genre(data):
    for item in data:
        # แปลงชื่อประเภทข้อมูลใน predicted_genre
        if item['predicted_genre'] in genre_translation:
            item['predicted_genre'] = genre_translation[item['predicted_genre']]
        
        # แปลงชื่อประเภทข้อมูลใน genre_percentages
        translated_percentages = {}
        for genre, percentage in item['genre_percentages'].items():
            if genre in genre_translation:
                translated_percentages[genre_translation[genre]] = percentage
        item['genre_percentages'] = translated_percentages

# ฟังก์ชันดึงฟีเจอร์จากไฟล์เสียง
def extract_features(file_name, sr=44100, duration=30):
    y, sr = librosa.load(file_name, sr=sr)
    num_segments = len(y) // (sr * duration)
    features_list = []

    for i in range(num_segments):
        start = i * sr * duration
        end = start + sr * duration
        y_segment = y[start:end]

        features = {}
        features['mfcc'] = np.mean(librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=13).T, axis=0).tolist()
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y_segment, sr=sr).T, axis=0).tolist()
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y_segment).T, axis=0).tolist()
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y_segment, sr=sr).T, axis=0).tolist()
        features['chroma_stft'] = np.mean(librosa.feature.chroma_stft(y=y_segment, sr=sr).T, axis=0).tolist()
        mel_spectrogram = librosa.feature.melspectrogram(y=y_segment, sr=sr)
        features['mel_spectrogram'] = np.mean(librosa.power_to_db(mel_spectrogram).T, axis=0).tolist()

        features_list.append(features)

    return features_list

# ฟังก์ชั่นเตรียมฟีเจอร์
def prepare_features(features_list):
    X = []
    for features in features_list:
        feature_array = []
        for feature in features.values():
            feature_array.extend(feature)
        X.append(feature_array)

    X = np.array(X)
    X = scaler.transform(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X

# ฟังก์ชันทำนายประเภทของเพลง
def predict_genre(file_name):
    features_list = extract_features(file_name)
    X = prepare_features(features_list)
    predictions = model.predict(X)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_genres = label_encoder.inverse_transform(predicted_classes)

    # P สร้างตัวแปรไว้เขียนข้อมูล
    results = []
    # H:pred_class มีไว้ทำไมไม่รู้ แต่ต้องมี P:WTF
    for i, (pred, pred_class) in enumerate(zip(predictions, predicted_classes)):
        genre_probabilities = {label_encoder.classes_[j]: prob for j, prob in enumerate(pred)}
        genre_probabilities = dict(sorted(genre_probabilities.items(), key=lambda item: item[1], reverse=True))

        genre_percentages = {genre: f"{prob*100:.2f}%" for genre, prob in genre_probabilities.items()}


        # P ใช้appendมาเขียนใส่result
        results.append({
            'segment': i+1,
            'predicted_genre': predicted_genres[i],
            'genre_percentages': genre_percentages
        })

    #translate_genre(results)
    for result in results:
        print(result)    
        
    return results

# ฟังก์ชั่นmp3->wav
def convert_mp3_to_wav(file_path):
    if file_path.endswith('.mp3'):
        new_file_path = file_path[:-4] + '.wav' # เปลี่ยนนามสกุลไฟล์

        # ตรวจสอบว่าไฟล์ .wav มีหรือไม่
        if not os.path.exists(new_file_path):
            # ใช้ ffmpeg เพื่อแปลงไฟล์
            command = f'ffmpeg -i "{file_path}" -acodec pcm_s16le -ar 44100 "{new_file_path}"'
            result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode != 0:
                print(f"Error converting {file_path} to {new_file_path}:\n{result.stderr.decode('utf-8')}")
                return None

            print(f'Converted: {file_path} to {new_file_path}')
        else:
            print(f'File already exists: {new_file_path}')
        return new_file_path
    return file_path

# ฟังก์ชั่นแยกเสียงร้อง/ดนตรี spleeter
def separate_vocals(input_file):
    output_dir = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    separator = Separator('spleeter:2stems')
    separator.separate_to_file(input_file, output_dir)

    vocals_path = os.path.join(output_dir, base_name, 'vocals.wav')
    new_vocals_path = os.path.join(output_dir, f'{base_name}_vocals.wav')

    if not os.path.exists(vocals_path):
        print(f"Failed to separate vocals for file: {input_file}")
        return None

    os.rename(vocals_path, new_vocals_path)

    # ลบไฟล์ต้นฉบับและไฟล์ที่ไม่ต้องการ
    accompaniment_path = os.path.join(output_dir, base_name, 'accompaniment.wav')
    os.remove(accompaniment_path)
    os.rmdir(os.path.join(output_dir, base_name))

    print(f"Separated vocals to {new_vocals_path}")
    return new_vocals_path

# ฟังก์ชั่นแบ่ง30วิ
def split_wav_file(file_path, segment_length_ms, output_format, max_segments):
    if not file_path or not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return []

    # โหลดไฟล์เสียง
    audio = AudioSegment.from_file(file_path)
    if audio is None:
        print(f"Failed to load audio file: {file_path}")
        return []

    # คำนวณจำนวนไฟล์ที่จะได้รับจากการตัด
    num_segments = min(len(audio) // segment_length_ms, max_segments)
    segment_files = []

    # กำหนดความยาวของหมายเลข segment ตามจำนวน segments ที่มีสูงสุด
    num_length = len(str(num_segments))

    # ตัดและบันทึกแต่ละส่วน
    for i in range(num_segments):
        start_ms = i * segment_length_ms
        end_ms = start_ms + segment_length_ms
        segment = audio[start_ms:end_ms]

        # สร้างชื่อไฟล์สำหรับแต่ละส่วน
        segment_file_name = f"{file_path[:-4]}_segment_{str(i+1).zfill(num_length)}.{output_format}"

        # บันทึกส่วนของไฟล์เสียง
        segment.export(segment_file_name, format=output_format)
        print(f"Exported {segment_file_name}")
        segment_files.append(segment_file_name)

    return segment_files

# ฟังก์ชั่นเปลี่ยนเสียงเป็นMono
def convert_stereo_to_mono(file_path):
    if not file_path or not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    if file_path.endswith(".wav"):
        temp_file_path = file_path[:-4] + "_mono.wav"

        # ใช้ ffmpeg เพื่อแปลงไฟล์จาก stereo เป็น mono
        command = ["ffmpeg", "-i", file_path, "-ac", "1", temp_file_path]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            print(f"Error converting {file_path} to mono:\n{result.stderr.decode('utf-8')}")

        # แทนที่ไฟล์เดิมด้วยไฟล์ที่แปลงแล้ว
        os.remove(file_path)
        os.rename(temp_file_path, file_path)

        print(f"Converted {file_path} to mono.")

# ฟังก์ชั่นเช็คMono
def check_mono_conversion(file_path):
    if not file_path or not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return False

    if file_path.endswith(".wav"):
        with wave.open(file_path, 'r') as wav_file:
            channels = wav_file.getnchannels()
            if channels != 1:
                print(f"File {file_path} has {channels} channels. Conversion failed.")
                return False
            else:
                print(f"File {file_path} is mono.")
                return True

# ฟังก์ชั่นปรับค่าไฟล์ให้เข้ากับโค้ด
def process_single_file(file_path):
    # mp3->wav
    new_file_path = convert_mp3_to_wav(file_path)

    if not new_file_path:
        print("Conversion from mp3 to wav failed.")
        return

    # แยกร้อง/ดนตรี
    vocal_file_path = separate_vocals(new_file_path)

    # แบ่ง30วิ
    segment_length_ms = 30 * 1000 # 30 วินาทีเป็นมิลลิวินาที
    output_format = 'wav' # หรือ 'mp3'
    max_segments = 20 # สร้างไฟล์ตัดแยกได้สูงสุด 20 ไฟล์ต่อไฟล์ต้นฉบับ
    segment_files = split_wav_file(vocal_file_path, segment_length_ms, output_format, max_segments)

    # เช็คMono
    for segment_file in segment_files:
        convert_stereo_to_mono(segment_file)
        check_mono_conversion(segment_file)

    # ลบไฟล์เก่า
    if file_path.endswith('.mp3') and os.path.exists(file_path):
        os.remove(file_path)
        print(f"Original file {file_path} has been removed.")

    if os.path.exists(new_file_path):
        os.remove(new_file_path)
        print(f"Original file {new_file_path} has been removed.")

    if os.path.exists(vocal_file_path):
        os.remove(vocal_file_path)
        print(f"Original file {vocal_file_path} has been removed.")

# ฟังก์ชั่นถอดคำ Google cloud speech to text api
def transcribe_audio(audio_file_path):
    """ os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'api-for-music-424713-b03aee08ba61.json' """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'try4-gcs2t-7eeb22676230.json'
    client = speech.SpeechClient()

    with open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    # ตรวจสอบว่าไฟล์เสียงไม่ว่างเปล่า
    if not content:
        print(f"Audio file is empty: {os.path.basename(audio_file_path)}")
        return None

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100, # 16000 44100 22050 48000
        language_code="th-TH",
    )

    try:
        response = client.recognize(config=config, audio=audio)
    except Exception as e:
        print(f"Error transcribing audio file: {os.path.basename(audio_file_path)}\n{str(e)}")
        return None

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript

    if not transcript:
        print(f"Failed to transcribe audio file: {os.path.basename(audio_file_path)}")
        return None

    return transcript

# ฟังก์ชั่นโหลดJSON
def load_words(json_file_path):
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"The file {json_file_path} does not exist.")

    with open(json_file_path, mode='r', encoding='utf-8-sig') as file:
        words = json.load(file)
    return words

# ตรวจสอบการมีอยู่ของไฟล์ JSON
#negative_words_path = 'Negative_word_Cleaned.json'
negative_words_path = 'Negative_word_Cleaned_New_Ref_Word.json'
positive_words_path = 'Positive_Word_Cleaned.json'

if not os.path.exists(negative_words_path):
    raise FileNotFoundError(f"{negative_words_path} not found.")

if not os.path.exists(positive_words_path):
    raise FileNotFoundError(f"{positive_words_path} not found.")

# โหลดข้อมูลคำลบและคำบวกจากไฟล์ JSON
negative_words = load_words(negative_words_path)["Negative_word"]
positive_words = load_words(positive_words_path)["Positive_word"]

# ฟังก์ชั่นหาคำที่ส่งผลเชิงลบ/ทางที่ไม่ดี ต่อผู้ฟัง
def evaluate_sentiment(transcript, negative_words, positive_words):
    transcript_words = word_tokenize(transcript, engine="newmm")

    # สร้าง bigrams และ trigrams
    bigrams = [''.join(transcript_words[i:i+2]) for i in range(len(transcript_words)-1)]
    trigrams = [''.join(transcript_words[i:i+3]) for i in range(len(transcript_words)-2)]

    combined_words = transcript_words + bigrams + trigrams

    negative_count = 0
    positive_count = 0

    # ลบการคำนวณคะแนน และเก็บเพียงจำนวนคำ
    for word in combined_words:
        if word in negative_words:
            negative_count += 1
        elif word in positive_words:
            positive_count += 1

    # ไม่คำนวณ total_score, negative_score, positive_score
    return negative_count, positive_count, combined_words, transcript  # เพิ่ม transcript ในการคืนค่า


# ฟังก์ชั่นลบไฟล์เสียง
def delete_all_audio_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete file: {file_path}. Reason: {str(e)}")

# ฟังก์ชั่นคำนวนเชิงลบทั้งเพลง
def process_all_wav_files(directory):
    total_negative_count = 0
    total_positive_count = 0
    total_transcripts = []
    total_words_count = 0

    segment_results = []  # เก็บผลลัพธ์ของแต่ละsegment

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                audio_file_path = os.path.join(root, file)
                print(f"Processing file: {file}")

                transcript = transcribe_audio(audio_file_path)
                print('ข้อมูลดิบ') #แก้บัค1
                print(transcript) #แก้บัค1
                if not transcript:
                    print(f"Skipping file due to transcription failure: {file}")
                    transcript = '-' #แก้บัค1
                    print('เช็คบัค1') #แก้บัค1
                    print(transcript) #แก้บัค1
                    #continue
                print('เช็คบัค2') #แก้บัค1
                print(transcript) #แก้บัค1


                negative_count, positive_count, combined_words, segment_transcript = evaluate_sentiment(
                    transcript, negative_words, positive_words)

                if not transcript == '-': # บัคอยู่
                    total_transcripts.append(segment_transcript) # บัคอยู่
                total_negative_count += negative_count
                total_positive_count += positive_count
                total_words_count += len(combined_words)

                # เก็บผลลัพธ์ของแต่ละ segment
                segment_results.append({
                    "transcript": segment_transcript,
                    "negative_count": negative_count,
                    "positive_count": positive_count,
                    "combined_words": combined_words
                })

                print(f"Tokenized Words: {combined_words}")
                print(f"Transcript: {segment_transcript}")
                print(f"Negative Words Count: {negative_count}")
                print()
                os.remove(audio_file_path)

    total_transcripts_combined = ' '.join(total_transcripts)
    negative_percentage_total_words = (total_negative_count / total_words_count) * 100 if total_words_count > 0 else 0
    negative_percentage_positive_words = (total_negative_count / total_positive_count) * 100 if total_positive_count > 0 else 0

    negative_percentage_total_words = round(negative_percentage_total_words, 2)

    is_negative_overall = (
        negative_percentage_total_words >= 30 or
        negative_percentage_positive_words >= 10
    )

    print("\n--- Summary ---")
    print(f"Total Negative Words Count: {total_negative_count}")
    print(f"Total Positive Words Count: {total_positive_count}")
    print(f"Total Words Count: {total_words_count}")
    print(f"Total Negative Words Compared To Positive Words: {negative_percentage_positive_words:.2f}%")
    print(f"Total Negative Words Compared To Total Words: {negative_percentage_total_words:.2f}%")
    print(f"Is the overall sentiment negative? {'Yes' if is_negative_overall else 'No'}")
    print()

    delete_all_audio_files(directory)

    bad_word_summary = {
        "total_negative_count": total_negative_count,
        "total_words_count": total_words_count,
        "negative_percentage_total_words": negative_percentage_total_words,
        "transcript": total_transcripts_combined,
        "segment_results": segment_results  # เพิ่ม segment results ในการคืนค่า
    }

    return bad_word_summary



# หน้าแรกสำหรับอัพโหลดไฟล์
@app.route('/')
def index():
    return render_template('home.html')

# ฟังก์ชันจัดการการอัพโหลดไฟล์
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'status': 'No selected file'})

    if file and allowed_file(file.filename):
        name = secure_filename(file.filename)
        print(name) # เช็คชื่อเฉยๆ
        # เปลี่ยนจาก abcd.wav เป็น song.wav
        ext = file.filename.rsplit('.', 1)[1].lower()
        if ext == 'mp3':
            filename = 'song.mp3'
        elif ext == 'wav':
            filename = 'song.wav'

        # กำหนด path ของไฟล์ที่อัพโหลดมา ไปที่โฟลเดอร์ uploads
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # จับเวลาทำงาน
        start_all_time = time.time()

        # อัพโหลด
        start_upload_time = time.time()
        file.save(file_path)
        end_upload_time = time.time()
        upload_time = round(end_upload_time - start_upload_time, 2)

        # ก๊อปปี้ไฟล์ที่อัพโหลด
        file_path_2 = os.path.join(app.config['CHECK_FOLDER'], filename)
        shutil.copy(file_path, file_path_2)

        # พรีดิกประเภทเพลงจากไฟล์ก๊อปปี้
        start_predict_time = time.time()
        results = predict_genre(file_path_2)
        print("ดิบ")
        print(results)
        translate_genre(results) #
        print("ไทย")
        print(results) # อยากเช็คข้อมูลเฉยๆ
        os.remove(file_path_2) # ลบไฟล์ก๊อปปี้
        end_predict_time = time.time()
        predict_time = round(end_predict_time - start_predict_time, 2)

        """ breakpoint() """

        # ปรับไฟล์+spleeter
        start_convert_time = time.time()
        process_single_file(file_path)
        end_convert_time = time.time()
        convert_time = round(end_convert_time - start_convert_time, 2)

        # คำเชิงลบ
        start_point_time = time.time()
        directory = 'uploads'
        bad_word_summary = process_all_wav_files(directory)
        print('bad_word_summary') # เช็คเฉยๆ
        print(bad_word_summary) # เช็คเฉยๆ
        print('transcript') # เช็คเฉยๆ
        print(bad_word_summary['transcript']) # เช็คเฉยๆ
        end_point_time = time.time()
        point_time = round(end_point_time - start_point_time, 2)

        end_all_time = time.time()
        all_time = round(end_all_time - start_all_time, 2)

        # นับ segment ใน results เผื่อมี model ใหม่มา
        segment_genre_counts = []
        for result in results:
            genre_counts = {}
            for genre, percentage in result['genre_percentages'].items():
                genre_counts[genre] = float(percentage.strip('%'))
            segment_genre_counts.append(genre_counts)

        """breakpoint() """


        return jsonify({'results': results, # ผลลัพธ์ (ประเภทเพลงส่วนใหญ่,ประเภทเพลงแยก%)
                        'bad_word_summary': bad_word_summary, # ข้อมูลคำเชิงลบ
                        'transcript': bad_word_summary['transcript'], # คำร้องที่จับได้
                        'upload_time': upload_time, # เวลาที่ใช้ช่วงอัปโหลด
                        'predict_time': predict_time, # เวลาที่ใช้ช่วงพรีดิก
                        'convert_time': convert_time, # เวลาที่ใช้ช่วงปรับไฟล์
                        'point_time': point_time, # เวลาที่ใช้ช่วงคำเชิงลบ
                        'all_time': all_time, # เวลาที่ใช้ทั้งหมด
                        'segment_genre_counts': segment_genre_counts  # จำนวนประเภทเพลง เอาไว้วนลูป /result
                        })
    else:
        return jsonify({'status': 'ประเภทไฟล์ไม่ถูกต้อง ควรอัพโหลด ไฟล์ประเภท mp3, wav'})

# หน้าแสดงผลลัพธ์
@app.route('/result')
def result():
    results = request.args.get('results')
    bad_word_summary = request.args.get('bad_word_summary')
    transcript = request.args.get('transcript')
    upload_time = request.args.get('upload_time')
    predict_time = request.args.get('predict_time')
    convert_time = request.args.get('convert_time')
    point_time = request.args.get('point_time')
    all_time = request.args.get('all_time')
    segment_genre_counts = request.args.get('segment_genre_counts')

    # แปลง results กลับเป็นออบเจกต์ Python
    results = json.loads(results) # อันนี้ต้องใช้
    bad_word_summary = json.loads(bad_word_summary) # อันนี้ไม่มั่นใจ แต่ทำไว้ก่อน เป็นข้อมูลwordเหมือนกัน

    return render_template('result.html',
                           results=results,
                           bad_word_summary=bad_word_summary,
                           transcript=transcript,
                           upload_time=upload_time,
                           predict_time=predict_time,
                           convert_time=convert_time,
                           point_time=point_time,
                           all_time=all_time,
                           segment_genre_counts=segment_genre_counts,
                           zip=zip
                           )

# คณะผู้จัดทำ
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # ฟังก์ชั่นสร้างโฟลเดอร์
    create_folders()
    gc.collect()
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
