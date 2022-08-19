import streamlit as st

st.set_page_config(
    page_title="About the App",
    page_icon="ℹ️",
    layout="wide"
)

with st.expander("ABOUT THE CREATOR", expanded=True):
    st.markdown("""
    <h1 id="name">AYUSHMAAN DAS</h1>
    """, True)

    col1, col2 = st.columns([2.5,4])
    col1.image(".\\pages\\dpfrk.png", width=250)

    col2.markdown("""
    - **STREAM :** B.Tech CSE - Artificial Intelligence and Machine Learning (<b>AI & ML</b>)
    <hr/>

    - **INSTITUTE :** Sri Ramachandra Engineering and Technology, Sri Ramachandra Institute of Higher Education and Research (<b>SRIHER</b>), Porur, Chennai - 600116
    <hr/>

    - **PERIOD OF WORK :** First Year Internship (INT 200 - Internship 1)
    """, True)

    st.markdown("""<hr/>This app/project has been made under the guidance of <b><i><u>PROF. JAYANTHI G</u></i></b> and <b><i><u>DR. RAMYA M</u></i></b>
    of SRET (SRIHER) - Chennai.
    <hr/>
    """,True)

with st.expander("LIBRARIES USED"):
    st.markdown("""
    The following Python Libraries have been used to build this app:
    - **TweePy** - Fetching of Tweets from Twitter API
    - **Pandas** - Datframes and datasets
    - **Scikit-Learn** - Model Buidling and Classification
    - **Matplotlib & Seaborn** - Data visualisation
    - **NLTK** - Preprocessing of text
    - This application has been designed using **STREAMLIT**
    """)


workflow = st.expander("WORKFLOW and ARCHITECTURE")
workflow.image(".\\pages\\WorkFlow.png", output_format='PNG')