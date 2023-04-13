import pandas as pd
import numpy as np
from underthesea import word_tokenize
import regex
import pickle
from gensim import corpora
import warnings
warnings.filterwarnings('ignore')
import jieba
import re
import swifter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from scipy import sparse
from PIL import Image



# Stop word
with open('vietnamese-stopwords.txt', 'r', encoding='utf-8') as file:
    stop_words = file.read()
stop_words = stop_words.split('\n')
print('Done')
# Gensim
@st.cache_data(show_spinner = False)
def load_data():
    data_load = pd.read_excel('data.xlsx',sheet_name = 0,engine = 'openpyxl')
    products_gem = [[text for text in x.split()] for x in data_load['process']]
    dictionary = corpora.Dictionary(products_gem)
    index_vs_productid = data_load[['product_id','process']]
    tfidf_matrix_load = sparse.load_npz("cosine_similarity_matrix.npz")
    tf_model = pickle.load(open('tf_model.sav', 'rb'))
    category = data_load['sub_category'].unique()
    image_no = Image.open('no-image-available.jpg')
    ALS_df = pd.read_excel('ALS_prediction.xlsx',sheet_name = 0,engine = 'openpyxl')
    return (data_load,products_gem,dictionary,index_vs_productid,tfidf_matrix_load,tf_model,category,image_no,ALS_df)
#Load data
data_load,products_gem,dictionary,index_vs_productid,tfidf_matrix_load,tf_model,category,image_no,ALS_df = load_data()

# Load Function

def text_process(series):
    test = series.str.lower()

    test = test.swifter.apply(lambda x: regex.sub(r'\.+', " ", x))

    test = test.swifter.apply(lambda x: str(x).replace('\n', ' '))

    test = test.swifter.apply(lambda x: str(x).replace('/', ' '))

    test = test.swifter.apply(
        lambda x: re.sub('[^A-Za-z\s+áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+', '', x))

    test = test.swifter.apply(lambda x: ' '.join([i if len(i) > 1 and len(i) < 8 else '' for i in x.split(' ')]))

    test = test.swifter.apply(lambda x: regex.sub(r'\s+', ' ', x).strip())

    test = test.swifter.apply(lambda x: ' '.join('' if word in stop_words else word for word in x.split()))

    test = test.swifter.apply(lambda x: regex.sub(r'\s+', ' ', x).strip())

    test = test.swifter.apply(lambda x: word_tokenize(x, format="text"))

    test = test.swifter.apply(lambda x: regex.sub(r'\s+', ' ', x).strip())

    return test

def cornvert_str_to_tuple(a):
  a = a.replace('(',"").replace(')',"").split(";")
  b = [i.split(",") for i in a]
  c = [[int(i) for i in j] for j in b]
  return c

def suggest_product2(search_data, top=5, search_type='product_id'):
    if search_type == 'product_id':
        if search_data not in data_load['product_id'].to_list():
            return False
        index_ = data_load[data_load['product_id'] == search_data].index
        array_ = tfidf_matrix_load[index_]
        cosine_similarities = cosine_similarity(array_, tfidf_matrix_load)
        result = pd.DataFrame({'data': cosine_similarities[0]}).merge(index_vs_productid, left_index=True,
                                                                      right_index=True, how='left')
        result = result.sort_values(['data'], ascending=False).iloc[1:top + 1]
    if search_type == 'comment':
        df = pd.DataFrame({'text': [search_data]})
        search = text_process(df['text'])[0]
        search = search.split(' ')
        search = tf_model.transform(search)
        cosine_similarities = cosine_similarity(search, tfidf_matrix_load)
        result = pd.DataFrame({'data': cosine_similarities[0]}).merge(index_vs_productid, left_index=True,
                                                                      right_index=True, how='left')
        result = result.sort_values(['data'], ascending=False).iloc[1:top + 1]
        pass

    return result

def ALS_suggest(user_id):

    user_ALS = ALS_df[ALS_df['user_id'] == user_id]

    return user_ALS[['product_id','rating']]


# GUI
menu = ["Business Objective", "Content base filtering", "Colaborative filtering"]
choice = st.sidebar.selectbox('Menu', menu)

if choice == menu[0]:
    st.subheader(menu[0])
    st.image("cac-san-thuong-mai-dien-tu.jpg.webp")
    st.write('''+ Shopee là một hệ sinh thái thương mại “all in one”, trong đó có shopee.vn,
     là một website thương mại điện tử đứng top 1 của Việt Nam và khu vực Đông Nam Á.Trên trang này đã triển khai nhiều tiện ích hỗ trợ nâng cao trải nghiệm người dùng và họ muốn xây dựng nhiều tiện ích hơn nữa.Giả sử công ty này chưa triển khai Recommender System và bạn được yêu cầu triển khai hệ thống này, bạn sẽ làm gì?''')
    st.write('''+ Báo cáo gồm 2 phần :
    + Contentbase filtering sử dụng thuật toán Cosine
    + Colaborative sử dụng thuật toán ALS''')
elif choice == menu[1]:
    st.subheader(menu[1])
    type = st.radio("PLEASE INPUT COMMENT OR PRODUCT ID", options=("COMMENTS", "PRODUCT"))
    if type == 'COMMENTS':
        with st.form("my_form"):
            slider = st.slider('How many suggest product do you need?', 5, 10, 5)
            input_data = st.text_area(label="input your content")
            submitted = st.form_submit_button("Submit")
        if submitted:
            if input_data != '':
                st.divider()
                st.write("PRODUCT YOU SHOULD BUY :")
                predict = suggest_product2(input_data, top=slider, search_type='comment')
                suggest_id = predict['product_id']
                with st.columns(3)[1]:
                    num = 1
                    for i in suggest_id:
                        image = data_load[data_load['product_id'] == i]['image']
                        link = data_load[data_load['product_id'] == i]['link'].iloc[0]
                        image = image.fillna(0).iloc[0]
                        if image == 0:
                            st.write('Product number {}'.format(num))
                            st.image(image_no, caption=st.write("[Product link from shoppee]({0})".format(link)))
                            st.divider()
                        else:
                            st.write('Product number {}'.format(num))
                            st.image(image, caption=st.write("[Product link from shoppee]({0})".format(link)))
                            st.divider()
                        num += 1



    elif type == 'PRODUCT':
        category_box = st.selectbox(label="Choose your category", options=category)
        with st.form("my_form"):
            product_id = data_load[data_load['sub_category']==category_box]['product_id'].unique()
            productid_box = st.selectbox(label = "Choose your product_id",options= product_id)
            image = data_load[data_load['product_id']==productid_box]['image'].fillna(0).iloc[0]
            link = data_load[data_load['product_id']==productid_box]['link'].iloc[0]
            slider = st.slider('How many suggest product do you need?', 5, 10, 5)
            submitted = st.form_submit_button("Submit")
        if submitted:
            st.divider()
            st.write("YOUR PRODUCT IS :")
            with st.columns(3)[1]:
                if image == 0:
                    st.image(image_no, caption=st.write("[Product link from shoppee]({0})".format(link)))
                else:
                    st.image(image, caption=st.write("[Product link from shoppee]({0})".format(link)))
            st.divider()
            predict = suggest_product2(productid_box, top=slider, search_type='product_id')
            suggest_id = predict['product_id']
            st.write("PRODUCT YOU SHOULD BUY :")
            with st.columns(3)[1]:
                num = 1
                for i in suggest_id:
                    image = data_load[data_load['product_id'] == i]['image']
                    link = data_load[data_load['product_id'] == i]['link'].iloc[0]
                    image = image.fillna(0).iloc[0]
                    if image == 0:
                        st.write('Product number {}'.format(num))
                        st.image(image_no, caption=st.write("[Product link from shoppee]({0})".format(link)))
                        st.divider()
                    else :
                        st.write('Product number {}'.format(num))
                        st.image(image, caption=st.write("[Product link from shoppee]({0})".format(link)))
                        st.divider()
                    num += 1

elif choice == menu[2]:
    st.subheader(menu[2])
    with st.form("my_form"):
        user_id = st.selectbox("INPUT YOUR CUSTOMER ID:",ALS_df['user_id'].unique())
        submitted = st.form_submit_button("Submit")
    if submitted:
        st.write('Product you should buy')
        suggest = ALS_suggest(user_id)
        st.dataframe(suggest)

