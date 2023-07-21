import streamlit as st
import helper
import pickle
from streamlit_option_menu import option_menu
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

model = pickle.load(open('model.pkl','rb'))
df = pd.read_csv('data.csv')




with st.sidebar:
    user_input = option_menu(None, ["Home", "Prediction",  "Dashboard"], 
        icons=['house', 'clipboard-data', "bar-chart-line"], 
        menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "green"},
        }
    )

if user_input == 'Home':
    st.header("Duplicate Question Pair Detection Model")
    st.markdown("There are various platform for Q&A, just like StackOverflow , Quora etc. One of the many problems that these platforms face is the duplication of questions. Duplication of question ruins the experience for both the questioner and the answerer. This is the model to solve the problem ")

if user_input == 'Prediction':
    st.header('Duplicate Question Pairs')

    q1 = st.text_input('Enter question 1')
    q2 = st.text_input('Enter question 2')

    if st.button('Find'):
        query = helper.query_point_creator(q1,q2)
        result = model.predict(query)[0]

        if result:
            st.header('Duplicate')
        else:
            st.header('Not Duplicate')


if user_input == 'Dashboard':
    st.title("EDA of advanced features")

    st.header("Target value Distribution")
    ax = sns.countplot(x='is_duplicate', data=df)
    st.pyplot(ax.get_figure())

    st.header("Analysis of ctc_min, cwc_min, csc_min")
    fig = sns.pairplot(df[['ctc_min', 'cwc_min', 'csc_min', 'is_duplicate']],hue='is_duplicate')
    st.pyplot(fig)

    st.header("Analysis of ctc_max, cwc_max, csc_max")
    fig = sns.pairplot(df[['ctc_max', 'cwc_max', 'csc_max', 'is_duplicate']],hue='is_duplicate')
    st.pyplot(fig)

    st.header("Analysis of last_word_eq and first_word_eq")
    fig = sns.pairplot(df[['last_word_eq', 'first_word_eq', 'is_duplicate']],hue='is_duplicate')
    st.pyplot(fig)

    st.header("Analysis of mean_len, abs_len_diff, longest_substr_ratio")
    sns.pairplot(df[['mean_len', 'abs_len_diff','longest_substr_ratio', 'is_duplicate']],hue='is_duplicate')
    st.pyplot(fig)

    st.header("Analysis of fuzz_ratio, fuzz_partial_ratio, token_sort__ratio")
    fig = sns.pairplot(df[['fuzz_ratio', 'fuzz_partial_ratio','token_sort_ratio','token_set_ratio', 'is_duplicate']],hue='is_duplicate')
    st.pyplot(fig)