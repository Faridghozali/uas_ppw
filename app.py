import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

Data, lda, Model, Implementasi = st.tabs(['Data', 'LDA', 'Modelling', 'Implementasi'])

with Data:
   st.title("Impelementasi Latent Dirichlet Allocation (LDA) ")
   st.text("Arbil Shofiyurrahman - 210411100016")
   st.subheader("Deskripsi Data")
   st.write("Fitur Fitur yang ada diantaranya:")
   st.text("Fitur-fitur data:")
   st.text("1) Judul")
   st.text("2) Isi")
   st.text("3) Label")
   st.subheader("Data")
   data = pd.read_csv("berita.csv")
   st.write(data)
