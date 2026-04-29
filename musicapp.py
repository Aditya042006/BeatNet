import streamlit as st
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model
import tempfile
import os

st.set_page_config(page_title="BeatNet", layout="centered")

@st.cache_resource
def loadai():
    m = load_model("b.h5", compile=False)
    with open("g.pkl", "rb") as f:
        enc = pickle.load(f)
    return m, enc

m, enc = loadai()

st.markdown("<h1 style='text-align: center; font-family: Impact, sans-serif; color: #FF4B4B; font-size: 65px; letter-spacing: 3px;'>BeatNet</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px; color: gray;'>Upload any .wav or .mp3 audio track, and our CNN will predict its genre.</p>", unsafe_allow_html=True)

upf = st.file_uploader("Upload Audio Track", type=["wav", "mp3"])

if upf is not None:
    st.audio(upf, format='audio/wav')

if st.button("Predict Genre", type="primary"):
    if upf is None:
        st.warning("Please upload a file first!")
    else:
        if upf.size > 26214400:
            st.error("File is too heavy! Please upload a track smaller than 25MB.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(upf.read())
                        path = tmp.name

                    aud, sr = librosa.load(path, offset=15.0, duration=30.0, mono=True, sr=22050)
                    wsec = 3.0
                    osec = 1.5
                    wsamp = int(wsec * sr)
                    hsamp = int((wsec - osec) * sr)
                    
                    chks = []
                    for pos in range(0, len(aud) - wsamp + 1, hsamp):
                        en = pos + wsamp
                        slc = aud[pos:en]
                        
                        mel = librosa.feature.melspectrogram(y=slc, sr=sr, n_mels=128)
                        dbmel = librosa.power_to_db(mel, ref=np.max)
                        
                        if dbmel.shape[1] < 128: 
                            dbmel = np.pad(dbmel, ((0, 0), (0, 128 - dbmel.shape[1])))
                        else: 
                            dbmel = dbmel[:, :128]
                            
                        mn = np.min(dbmel)
                        mx = np.max(dbmel)
                        norm = (dbmel - mn) / (mx - mn + 1e-6)
                        
                        chks.append(norm.reshape(128, 128, 1))
                    
                    os.remove(path) 

                    if len(chks) == 0:
                        st.error("Audio is too short!")
                    else:
                        inp = np.array(chks)
                        pr = m.predict(inp, verbose=0)
                        
                        mpr = np.mean(pr, axis=0)
                        widx = np.argmax(mpr)
                        pgen = enc.inverse_transform([widx])[0].upper()
                        
                        pconf = float(np.max(pr[:, widx]) * 100) 
                        if pconf > 95.0:
                            pconf = 90.0 + (pconf / 20.0) 

                        st.success("Analysis Complete!")
                        c1, c2 = st.columns(2)
                        c1.metric(label="Predicted Genre", value=pgen)
                        c2.metric(label="Confidence Level", value=f"{pconf:.2f}%")

                except Exception as err:
                    st.error(f"Error details: {err}")
