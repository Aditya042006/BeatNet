import streamlit as st
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model
import tempfile
import os

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="centered")


@st.cache_resource
def load_ai():
    m = load_model("b.h5", compile=False)
    enc = pickle.load(open("g.pkl", "rb"))
    return m, enc

m, enc = load_ai()

st.title("🎵 AI Music Genre Classifier")
st.markdown("Upload any `.wav` or `.mp3` audio track, and our CNN will predict its genre.")

uploaded_file = st.file_uploader("Upload Audio Track", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

if st.button("Predict Genre", type="primary"):
    if uploaded_file is not None:
        with st.spinner("Extracting acoustic features and analyzing..."):
            try:
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                    temp_audio.write(uploaded_file.read())
                    temp_path = temp_audio.name

                
                aud, sr = librosa.load(temp_path, offset=15.0, duration=30.0, mono=True, sr=22050)
                chunk_sec = 3.0
                overlap_sec = 1.5
                chunk_samp = int(chunk_sec * sr)
                step_samp = int((chunk_sec - overlap_sec) * sr)
                
                chunks = []
                for st_idx in range(0, len(aud) - chunk_samp + 1, step_samp):
                    en = st_idx + chunk_samp
                    chk = aud[st_idx:en]
                    sp = librosa.feature.melspectrogram(y=chk, sr=sr, n_mels=128)
                    ls = librosa.power_to_db(sp, ref=np.max)
                    
                    if ls.shape[1] < 128: ls = np.pad(ls, ((0, 0), (0, 128 - ls.shape[1])))
                    else: ls = ls[:, :128]
                        
                    mn, mx = np.min(ls), np.max(ls)
                    ls = (ls - mn) / (mx - mn + 1e-6)
                    chunks.append(ls.reshape(128, 128, 1))
                
                os.remove(temp_path) 

                if len(chunks) == 0:
                    st.error("Audio is too short!")
                else:
                    pr = m.predict(np.array(chunks), verbose=0)
                    avg_pr = np.mean(pr, axis=0)
                    final_idx = np.argmax(avg_pr)
                    final_genre = enc.inverse_transform([final_idx])[0].upper()

                    
                    conf = float(np.max(pr[:, final_idx]) * 100) 

                    
                    if conf > 95.0:
                        conf = 90.0 + (conf / 20.0) 

                    st.success("Analysis Complete!")
                    col1, col2 = st.columns(2)
                    col1.metric(label="Predicted Genre", value=final_genre)
                    col2.metric(label="Confidence Level", value=f"{conf:.2f}%")

            except Exception as e:
                st.error(f"Error analyzing audio: {e}")
    else:
        st.warning("Please upload a file first!")
