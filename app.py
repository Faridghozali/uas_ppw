import streamlit as st
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import re

# Fungsi untuk membersihkan dan normalisasi teks
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in custom_stopwords]
    cleaned_text = ' '.join(words)
    return cleaned_text

# Fungsi untuk modeling dan evaluasi
def model_and_evaluate(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Load data
data = pd.read_csv("https://gist.githubusercontent.com/Faridghozali/58d026e57e0e682b7dcaef9a3ce26607/raw/c8a9acebf8ec2937807c0ae2109e336b870a2829/data_berita.csv")
data['Content'].fillna("", inplace=True)

# Membuat list custom stop words dalam bahasa Indonesia
custom_stopwords = ["yang", "dan", "di", "dengan", "untuk", "pada", "adalah", "ini", "itu", "atau", "juga"]

# Fungsi untuk pembersihan dan normalisasi data
df = data.dropna(subset=['Label', 'Label'])
df = df.fillna(0)  # Gantilah nilai-nilai yang hilang dengan 0 atau nilai lain yang sesuai

# Pisahkan fitur dan label
X = df.drop(columns=['Label']).values
y = df['Label'].values

# Latih model LDA
topik = st.number_input("Masukkan Jumlah Topik yang Diinginkan", 1, step=1, value=5)
lda_model = LatentDirichletAllocation(n_components=topik, doc_topic_prior=0.2, topic_word_prior=0.1, random_state=42, max_iter=1)
lda_top = lda_model.fit_transform(X)

# Bobot setiap topik terhadap dokumen
nama_clm = [f"Topik {i+1}" for i in range(topik)]
U = pd.DataFrame(lda_top, columns=nama_clm)
data_with_lda = pd.concat([U, df['Label']], axis=1)

# Bagian Data
st.title("UAS Pencarian & Penambangan Web A")
st.text("Farid Ghozali - 210411100119")
st.subheader("Deskripsi Data")
st.write("Dimana Fitur yang ada di dalam data tersebut diantaranya:")
st.text("1) Date\n2) Title\n3) Content\n4) Label")
st.subheader("Data")
st.write(data)

# Bagian LDA
with st.expander("LDA"):
    st.write("Jumlah Topik yang Anda Gunakan : " + str(topik))
    st.write("Jika pada menu LDA tidak menentukan jumlah topiknya maka proses modelling akan di default dengan jumlah topik = 5")
    st.dataframe(data_with_lda)

# Bagian Modeling
with st.expander("Modelling"):
    st.write("Pilih metode yang ingin anda gunakan :")
    met1 = st.checkbox("KNN")
    met2 = st.checkbox("Naive Bayes")
    met3 = st.checkbox("Decision Tree")
    submit2 = st.button("Pilih")

    if submit2:
        if met1:
            st.write("Metode yang Anda gunakan Adalah KNN")
            model = KNeighborsClassifier(5)
            accuracy = model_and_evaluate(X_train, X_test, y_train, y_test, model)
            st.write("Akurasi: {:.2f}%".format(accuracy * 100))
        elif met2:
            st.write("Metode yang Anda gunakan Adalah Naive Bayes")
            model = MultinomialNB()
            accuracy = model_and_evaluate(X_train, X_test, y_train, y_test, model)
            st.write("Akurasi: {:.2f}%".format(accuracy * 100))
        elif met3:
            st.write("Metode yang Anda gunakan Adalah Decision Tree")
            model = DecisionTreeClassifier()
            accuracy = model_and_evaluate(X_train, X_test, y_train, y_test, model)
            st.write("Akurasi: {:.2f}%".format(accuracy * 100))
        else:
            st.write("Anda Belum Memilih Metode")

# Bagian Implementasi
with st.expander("Implementasi"):
    st.write("Masukkan Berita yang Ingin Dianalisis:")
    user_abstract = st.text_area("Abstrak", "")

    if user_abstract:
        preprocessed_user_abstract = preprocess_text(user_abstract)
        user_tf = count_vectorizer.transform([preprocessed_user_abstract])

        if lda_model is None:
            lda_model = LatentDirichletAllocation(n_components=topik, doc_topic_prior=0.2, topic_word_prior=0.1, random_state=42, max_iter=1)
            lda_top = lda_model.fit_transform(user_tf)
            st.write("Model LDA telah dilatih.")

        user_topic_distribution = lda_model.transform(user_tf)

        st.write(user_topic_distribution)
        st.write("Hasil Prediksi:")
        
        if met1:
            predicted_label = model1.predict(user_topic_distribution)
        elif met2:
            predicted_label = model2.predict(user_topic_distribution)
        elif met3:
            predicted_label = model3.predict(user_topic_distribution)
        else:
            predicted_label = None

        if predicted_label is not None:
            st.write("Label Kelas: {}".format(predicted_label[0]))
        else:
            st.write("Anda belum memilih metode.")
