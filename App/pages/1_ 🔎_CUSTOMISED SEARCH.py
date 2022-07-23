import streamlit as st
import search
import matplotlib.pyplot  as plt

st.set_page_config(
    page_title="Search by City name",
    page_icon="🔎",
    layout="wide"
)

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def input_check(city):
    if city == "":
        st.warning("You can't leave the city name blank !!!")
        st.stop()


st.title("CUSTOMISED SEARCH - Search the way you want !!!")

st.markdown("""
<hr>

### Please follow the instructions stated below :
- Enter the **correct spelling** of the city
- After entering the name, do not forget to **press enter**
- Select the Model you want to be used from the dropdown
- Press the submit button to check the results

<hr>
""", True)

city = st.text_input("TYPE IN THE NAME OF THE CITY", placeholder="Example : Seattle")

split_size = st.slider("DATASET SPLITTING SIZE (Training : Testing)", min_value=0.2, max_value=0.8, step=0.1, value=0.3)

col_opt1, col_opt2 = st.columns([1,1])
vector_type = col_opt1.selectbox("CHOOSE THE VECTORIZER", options=['COUNT VECTORIZER','TFIDF VECTORIZER'])
model_type = col_opt2.selectbox("CHOOSE THE MODEL TO BE USED", options=['LOGISTIC REGRESSION','NAIVE - BAYES MODEL', 'RANDOM FOREST CLASSIFIER'])

city_but = st.button("SEARCH FOR TWEETS")

st.markdown("<hr>", True)



###################################################


if city_but and (vector_type == 'COUNT VECTORIZER'):
    input_check(city)
    if model_type == 'LOGISTIC REGRESSION':
        df, scores, df_size, new_df_size, r_auc, logreg_auc, nb_auc, rf_auc, r_fpr, r_tpr, logreg_fpr, logreg_tpr, nb_fpr, nb_tpr, rf_fpr, rf_tpr = search.tweet_search(city, 1, 1, split_size)
    elif model_type == 'NAIVE - BAYES MODEL':
        df, scores, df_size, new_df_size, r_auc, logreg_auc, nb_auc, rf_auc, r_fpr, r_tpr, logreg_fpr, logreg_tpr, nb_fpr, nb_tpr, rf_fpr, rf_tpr = search.tweet_search(city, 1, 2, split_size)
    else:
        df, scores, df_size, new_df_size, r_auc, logreg_auc, nb_auc, rf_auc, r_fpr, r_tpr, logreg_fpr, logreg_tpr, nb_fpr, nb_tpr, rf_fpr, rf_tpr = search.tweet_search(city, 1, 3, split_size)
    
    st.title("RESULTS")
    st.header("Traffic - Related tweets in {0}".format(city.capitalize()))
    st.dataframe(df)

    col_metric1, col_metric2, col_metric3 = st.columns([1,1,1])
    col_metric1.metric(label="Trafic tweets fetched", value=df_size)
    col_metric2.metric(label="Tweets classified as 'Traffic'", value=new_df_size)
    csv = convert_df(df)
    col_metric3.download_button(label="Download the Tweets as a CSV", data=csv, file_name="search.csv", mime='text/csv')

    exp1 = st.expander("SHOW ACCURACY SCORES FOR VARIOUS MODELS")
    exp1.table(scores)
    exp2 = st.expander("SHOW ROC CURVES FOR THE MODELS")
    plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
    plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % rf_auc)
    plt.plot(nb_fpr, nb_tpr, marker='.', label='Naive Bayes (AUROC = %0.3f)' % nb_auc)
    plt.plot(logreg_fpr, logreg_tpr, marker='.', label='Logistic Regression (AUROC = %0.3f)' % logreg_auc)
    # Title
    plt.title('ROC Plot')
    # Axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # Show legend
    plt.legend() # 
    # Show plot
    plt.show()
    exp2.pyplot()

    st.snow()
    

    


elif city_but and (vector_type == 'TFID VECTORIZER'):
    input_check(city)
    if model_type == 'LOGISTIC REGRESSION':
        df, scores, df_size, new_df_size, r_auc, logreg_auc, nb_auc, rf_auc, r_fpr, r_tpr, logreg_fpr, logreg_tpr, nb_fpr, nb_tpr, rf_fpr, rf_tpr = search.tweet_search(city, 2, 1, split_size)
    elif model_type == 'NAIVE - BAYES MODEL':
        df, scores, df_size, new_df_size, r_auc, logreg_auc, nb_auc, rf_auc, r_fpr, r_tpr, logreg_fpr, logreg_tpr, nb_fpr, nb_tpr, rf_fpr, rf_tpr = search.tweet_search(city, 2, 2, split_size)
    else:
        df, scores, df_size, new_df_size, r_auc, logreg_auc, nb_auc, rf_auc, r_fpr, r_tpr, logreg_fpr, logreg_tpr, nb_fpr, nb_tpr, rf_fpr, rf_tpr = search.tweet_search(city, 2, 3, split_size)
    
    st.title("RESULTS")
    st.header("Traffic - Related tweets in {0}".format(city.capitalize()))
    st.dataframe(df)

    col_metric1, col_metric2, col_metric3 = st.columns([1,1,1])
    col_metric1.metric(label="Trafic tweets fetched", value=df_size)
    col_metric2.metric(label="Tweets classified as 'Traffic'", value=new_df_size)
    csv = convert_df(df)
    col_metric3.download_button(label="Download the Tweets as a CSV", data=csv, file_name="search.csv", mime='text/csv')

    exp1 = st.expander("SHOW ACCURACY SCORES FOR VARIOUS MODELS")
    exp1.table(scores)
    exp2 = st.expander("SHOW ROC CURVES FOR THE MODELS")
    plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
    plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % rf_auc)
    plt.plot(nb_fpr, nb_tpr, marker='.', label='Naive Bayes (AUROC = %0.3f)' % nb_auc)
    plt.plot(logreg_fpr, logreg_tpr, marker='.', label='Logistic Regression (AUROC = %0.3f)' % logreg_auc)
    # Title
    plt.title('ROC Plot')
    # Axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # Show legend
    plt.legend() # 
    # Show plot
    plt.show()
    exp2.pyplot()
    
    st.snow()


st.set_option('deprecation.showPyplotGlobalUse', False)
    
