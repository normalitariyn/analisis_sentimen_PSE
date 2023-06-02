from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics import accuracy_score
import googletrans
from googletrans import Translator
from tqdm.auto import tqdm
import streamlit as st
from streamlit_option_menu import option_menu
import altair as alt
from PIL import Image
import pandas as pd
import numpy as np
import regex as re
import json
import nltk
import pickle
nltk.download('stopwords')
nltk.download('punkt')


st.markdown("""<h1 style='text-align: center;'> Analisis Sentiment PSE dengan menggunakan Metode Logistic Regression </h1> """, unsafe_allow_html=True)
# 1. as sidevar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Analisis Sentiment", #required
        options=["Beranda", "Deskripsi", "Dataset", "Implementasi", "Tentang Kami"], #required
        icons=["house-door-fill", "book-half",  "folder-fill", "play-fill", "person-fill"], #optional
        menu_icon="cast", #optional
        default_index=0, #optional    
    styles={
        "container": {"padding": "0!important", "background-color":"white"},
        "icon": {"color": "black", "font-size": "17px"},
        "nav-link": {
            "font-size": "17px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#4169E1",
        },
        "nav-link-selected": {"background-color": "Royalblue"}
    }
    )

if selected == "Beranda":
    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        st.write("")

    with col2:
        img = Image.open('PSE.jpg')
        st.image(img, use_column_width=False, width=300)

    with col3:
        st.write("")

    st.write(""" """)
    
    
    st.write(""" """)
    
    st.write("""

    Keamanan terkait data pribadi merupakan hal yang sangat penting serta perlu di perhatikan
    oleh semua masyarakat dalam menggunakan internet. Kebocoran data pribadi pengguna dapat dimanfaatkan oleh
    orang yang tidak berkepentingan untuk berbagai macam tindakan kriminal 
    

    Dalam rangka memfasilitasi keamanan data masyarakat Indonesia, pemerintah melalui Kementerian
    Komunikasi dan Informasi (Kemkominfo) bertugas untuk membuat dan menerapkan beberapa kebijakan-kebijakan
    dalam bidang komunikasi serta informatika
    """)

if selected == "Deskripsi":
    st.subheader("Pengertian")
    st.write(""" Penyelenggara Sistem Elektronik adalah setiap Orang, penyelenggara negara, Badan Usaha, dan masyarakat yang menyediakan, mengelola, dan/atau mengoperasikan Sistem Elektronik secara sendiri-sendiri maupun bersama-sama kepada Pengguna Sistem Elektronik untuk keperluan dirinya dan/atau keperluan pihak lain""")
    st.subheader("Tujuan")
    st.write(""" 
    Penyelenggaraan Sistem Elektronik (PSE), dimana kebijakan
    tersebut memiliki tujuan untuk melindungi dari
    penyalahgunaan data pribadi penggunanya pada suatu layanan
    sistem elektronik  """)
    st.subheader("Pernyataan")
    st.write("""
    Masyarakat Indonesia mempunyai berbagai macam opini
terkait kebijakan tersebut ada yang sifatnya negatif, positif
maupun netral. Penggunaan platform media sosial sudah
menjadi salah satu sarana bagi masyarakat untuk berdiskusi
maupun berkomunikasi  Media sosial twitter merupakan
sebuah aplikasi yang banyak digunakan oleh masyarakat
Indonesia dalam menyampaikan sudut pandang terhadap suatu
topik tertentu""")

if selected == "Dataset":
    st.subheader("Dataset")
    st.write("""dataset tentang opini
    masyarakat mengenai PSE dari media sosial twitter yang
    nantinya kumpulan dari tweet tersebut akan diklasifikasikan
    ke dalam tiga kategori sentimen yaitu negatif netral serta
    positif dan kemudian dilakukan penerapan algoritma Logistic
    Regression untuk mengetahui nilai akurasinya. """)

    #st.subheader("""Pengolahan Data""")
    #st.image('psse.png', use_column_width=False, width=250)


    st.subheader("""Pengolahan Data""")
    st.write("""Preprocessing data merupakan proses dalam
mengganti teks tidak teratur supaya teratur yang nantinya
dapat membantu pada proses pengolahan data""")

    st.write(""" 
     1. Case folding merupakan tahap untuk mengganti keseluruhan kata kapital pada dataset agar berubah
menjadi tidak kapital.

     2. cleansing yaitu merupakan proses untuk menghilangkan semua simbol, mention, hastag,
retweet, url beserta emoticon pada dataset.

     3. tokenization yaitu proses untuk memisahkan suatu kalimat menjadi beberapa kata untuk
    memudahkan proses stemword serta stopword.

     4. steaming yaitu menghilangkan semua kata imbuhan menjadi kata dasar pada
dataset.

     5. stopword yaitu menghilangkan semua kata hubung serta kata yang tidak
diperlukan pada dataset
""")

    st.subheader(""" Pelabelan Dataset """)
    st.write("""Sebelum melakukan pelabelan dataset dengan library
textbloob teks yang berbahasa Indonesia harus dilakukan
penerjemahan terlebih dahulu kedalam bahasa inggris
menggunakan library google_trans. """)
  
    st.markdown("""<h2 style='text-align: center; color:grey;'> Dataset Tweet PSE </h1> """, unsafe_allow_html=True)
    df = pd.read_csv('https://raw.githubusercontent.com/aisyaturradiah/pba/main/PSE.csv')
    c1, c2, c3 = st.columns([1,5,1])

    with c1:
        st.write("")

    with c2:
        df

    with c3:
        st.write("")

if selected == "Implementasi":
    # user interface
    st.title("""Aplikasi Analisis Sentimen PSE""")
    st.subheader('Input Teks')
    new_data = st.text_area('Masukkan kalimat yang akan dianalisis :')
    submit = st.button("submit")

    if submit:
        #preprocessing
        def preprocessing(word):
            lower_case = new_data.lower()
            clean_tweet = re.sub("@[A-Za-z0-9_]+", "",lower_case)  #clenasing mention
            clean_tweet1 = re.sub("#[A-Za-z0-9_]+", "",clean_tweet)  #clenasing hashtag
            clean_tweet2 = re.sub(r'http\S+', '', clean_tweet1) #cleansing url link
            clean_tweet3 = re.sub("[^a-zA-Z ]+", " ", clean_tweet2) # cleansing character
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            stem = stemmer.stem(clean_tweet3)
            tokens = word_tokenize(stem)
            words = []
            temp = []
            for i in range(len(stem)):
                tokens = word_tokenize(stem)
                temp.append(tokens)
                listStopword = set(stopwords.words('indonesian'))
                removed = []
                for t in tokens:
                    if t not in listStopword:
                        removed.append(t)
                words.append(removed)
                kalimat = ' '.join(removed)
            penerjemah = Translator()
            hasil = penerjemah.translate(kalimat)
            translated = hasil.text
            lower_case2 = translated.lower()
            clean_char = re.sub("[^a-zA-Z ]+", " ", lower_case2)
            return lower_case, clean_tweet, clean_tweet1, clean_tweet2, clean_tweet3, stem, tokens, removed, kalimat, clean_char

        # Inputan
        st.subheader('Hasil Preprocessing Teks')
        lower_case, clean_tweet, clean_tweet1, clean_tweet2, clean_tweet3, stem, tokens, removed, kalimat, clean_char = preprocessing(
            new_data)
        st.write("Case Folding:", lower_case)
        st.write("Cleansing :", clean_tweet3)
        st.write("Steaming :", stem)
        st.write("Tokenizing :", tokens)
        st.write("Stopword :", removed)
        st.write("Kalimat :", kalimat)
        st.write("Translate :", clean_char)

        # Dataset
        df = pd.read_csv(
            'https://raw.githubusercontent.com/normalitariyn/dataset/main/dataset_PSE%20(1).csv')

        names = []
        with open(r'C:\Users\litas\hh.txt', 'r') as fp:
            for line in fp:
                x = line[:-1]
                names.append(x)

        #save model dengan pickle
        with open('model.pkl', 'rb') as file:
            model_pkl = pickle.load(file)

        #ekstraksi fitur
        tfidfvectorizer = TfidfVectorizer(analyzer='word')
        tfidf_wm = tfidfvectorizer.fit_transform(names)
        tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    
        #split data menjadi training data(80%) dan testing data(20%)
        training, test = train_test_split(tfidf_wm, test_size=0.2, random_state=1)
        training_label, test_label = train_test_split(
            df['Sentiment'], test_size=0.2, random_state=1)  # Nilai Y training dan Nilai Y testing

        #modelling dengan logistic regression 
        model_LR = model_pkl.fit(training, training_label)

        y_pred = model_LR.predict(test)

        # Prediksi
        tfidf_inputan = tfidfvectorizer.transform([stem]).toarray()  # vectorizing
        pred_text = model_pkl.predict(tfidf_inputan)
        st.subheader('Prediksi Kelas Teks')
        st.info(pred_text)

        # akurasi
        y_pred = model_pkl.predict(test)
        akurasi = accuracy_score(test_label, y_pred)
        st.subheader('Akurasi Model')
        st.success(akurasi)

if selected == "Tentang Kami":
    st.subheader("Tentang Kami")
    st.markdown("<h1 style='font-size:15px;text-align: left; color: black;'>Kelompok 4</h1>", unsafe_allow_html=True)
    st.markdown("1. Normalita Eka Ariyanti (200411100084)", unsafe_allow_html=True)
    st.markdown("2. Niken Amalia (200411100109)", unsafe_allow_html=True)
    st.markdown("3. Aisyatur Radiah (200411100116) ", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size:15px;text-align: left; color: black;'>Mata Kuliah : Pemrosesan Bahasa Alami - A </h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size:15px;text-align: left; color: black;'>Dosen Pengampu : Fika Hastarita Rachman, S.T., M.Eng.</h1>", unsafe_allow_html=True)

   