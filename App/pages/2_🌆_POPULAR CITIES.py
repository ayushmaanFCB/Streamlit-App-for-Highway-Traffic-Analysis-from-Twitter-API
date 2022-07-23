import streamlit as st
import search
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Commonly searched cities",
    page_icon="🌆",
    layout="wide"
)

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

st.title("COMMONLY SEARCHED CITIES")
st.markdown("<hr/>",True)

# USA ------------------------------------------

usa = st.container()
usa.markdown("# <u><i>United States</i></u>", True)
us1, us2, us3 = usa.columns([1,1,1])
us1.image(".\\pages\\la.jpg")
us2.image(".\\pages\\nyc.jpg")
us3.image(".\\pages\\seattle.jpg")

if us1.button("LOS ANGELES"):
    df, scores, df_size, new_df_size, r_auc, logreg_auc, nb_auc, rf_auc, r_fpr, r_tpr, logreg_fpr, logreg_tpr, nb_fpr, nb_tpr, rf_fpr, rf_tpr = search.tweet_search("los angeles", 1, 1, 0.3)
    usa.title("RESULTS")
    usa.header("Traffic - Related tweets for LOS ANGELES")
    usa.dataframe(df)
    col_metric1, col_metric2, col_metric3 = st.columns([1,1,1])
    col_metric1.metric(label="Trafic tweets fetched", value=df_size)
    col_metric2.metric(label="Tweets classified as 'Traffic'", value=new_df_size)
    csv = convert_df(df)
    col_metric3.download_button(label="Download the Tweets as a CSV", data=csv, file_name="search.csv", mime='text/csv')
    exp1 = usa.expander("SHOW ACCURACY SCORES FOR VARIOUS MODELS")
    exp1.dataframe(scores)
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
    usa.snow()
if us2.button("NEW YORK"):
    df, scores, df_size, new_df_size, r_auc, logreg_auc, nb_auc, rf_auc, r_fpr, r_tpr, logreg_fpr, logreg_tpr, nb_fpr, nb_tpr, rf_fpr, rf_tpr = search.tweet_search("nyc", 1, 1, 0.3)
    usa.title("RESULTS")
    usa.header("Traffic - Related tweets for NEW YORK")
    usa.dataframe(df)
    col_metric1, col_metric2, col_metric3 = st.columns([1,1,1])
    col_metric1.metric(label="Trafic tweets fetched", value=df_size)
    col_metric2.metric(label="Tweets classified as 'Traffic'", value=new_df_size)
    csv = convert_df(df)
    col_metric3.download_button(label="Download the Tweets as a CSV", data=csv, file_name="search.csv", mime='text/csv')
    exp1 = usa.expander("SHOW ACCURACY SCORES FOR VARIOUS MODELS")
    exp1.dataframe(scores)
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
    usa.snow()
if us3.button("SEATTLE"):
    df, scores, df_size, new_df_size, r_auc, logreg_auc, nb_auc, rf_auc, r_fpr, r_tpr, logreg_fpr, logreg_tpr, nb_fpr, nb_tpr, rf_fpr, rf_tpr = search.tweet_search("seattle", 1, 1, 0.3)
    usa.title("RESULTS")
    usa.header("Traffic - Related tweets for SEATTLE")
    usa.dataframe(df)
    col_metric1, col_metric2, col_metric3 = st.columns([1,1,1])
    col_metric1.metric(label="Trafic tweets fetched", value=df_size)
    col_metric2.metric(label="Tweets classified as 'Traffic'", value=new_df_size)
    csv = convert_df(df)
    col_metric3.download_button(label="Download the Tweets as a CSV", data=csv, file_name="search.csv", mime='text/csv')
    exp1 = usa.expander("SHOW ACCURACY SCORES FOR VARIOUS MODELS")
    exp1.dataframe(scores)
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
    usa.snow()

st.markdown("<hr/>",True)



# CANADA ------------------------------------------

canada = st.container()
canada.markdown("# <u><i>Canada</i></u>", True)
canada1, canada2, canada3 = canada.columns([1,1,1])
canada1.image(".\\pages\\toronto.jpg")
canada2.image(".\\pages\\ottawa.jpg")
canada3.image(".\\pages\\montreal.jpg")

if canada1.button("TORONTO"):
    df, scores, df_size, new_df_size, r_auc, logreg_auc, nb_auc, rf_auc, r_fpr, r_tpr, logreg_fpr, logreg_tpr, nb_fpr, nb_tpr, rf_fpr, rf_tpr = search.tweet_search("toronto", 1, 1, 0.3)
    canada.title("RESULTS")
    canada.header("Traffic - Related tweets for TORONTO")
    canada.dataframe(df)
    col_metric1, col_metric2, col_metric3 = st.columns([1,1,1])
    col_metric1.metric(label="Trafic tweets fetched", value=df_size)
    col_metric2.metric(label="Tweets classified as 'Traffic'", value=new_df_size)
    csv = convert_df(df)
    col_metric3.download_button(label="Download the Tweets as a CSV", data=csv, file_name="search.csv", mime='text/csv')
    exp1 = canada.expander("SHOW ACCURACY SCORES FOR VARIOUS MODELS")
    exp1.dataframe(scores)
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
    canada.snow()
if canada2.button("OTTAWA"):
    df, scores, df_size, new_df_size, r_auc, logreg_auc, nb_auc, rf_auc, r_fpr, r_tpr, logreg_fpr, logreg_tpr, nb_fpr, nb_tpr, rf_fpr, rf_tpr = search.tweet_search("ottawa", 1, 1, 0.3)
    canada.title("RESULTS")
    canada.header("Traffic - Related tweets for OTTAWA")
    canada.dataframe(df)
    col_metric1, col_metric2, col_metric3 = st.columns([1,1,1])
    col_metric1.metric(label="Trafic tweets fetched", value=df_size)
    col_metric2.metric(label="Tweets classified as 'Traffic'", value=new_df_size)
    csv = convert_df(df)
    col_metric3.download_button(label="Download the Tweets as a CSV", data=csv, file_name="search.csv", mime='text/csv')
    exp1 = canada.expander("SHOW ACCURACY SCORES FOR VARIOUS MODELS")
    exp1.dataframe(scores)
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
    canada.snow()
if canada3.button("MONTREAL"):
    df, scores, df_size, new_df_size, r_auc, logreg_auc, nb_auc, rf_auc, r_fpr, r_tpr, logreg_fpr, logreg_tpr, nb_fpr, nb_tpr, rf_fpr, rf_tpr = search.tweet_search("montreal", 1, 1, 0.3)
    canada.title("RESULTS")
    canada.header("Traffic - Related tweets for OTTAWA")
    canada.dataframe(df)
    col_metric1, col_metric2, col_metric3 = st.columns([1,1,1])
    col_metric1.metric(label="Trafic tweets fetched", value=df_size)
    col_metric2.metric(label="Tweets classified as 'Traffic'", value=new_df_size)
    csv = convert_df(df)
    col_metric3.download_button(label="Download the Tweets as a CSV", data=csv, file_name="search.csv", mime='text/csv')
    exp1 = canada.expander("SHOW ACCURACY SCORES FOR VARIOUS MODELS")
    exp1.dataframe(scores)
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
    canada.snow()

st.markdown("<hr/>",True)



# UK ------------------------------------------

uk = st.container()
uk.markdown("# <u><i>Great Britain</i></u>", True)
uk1, uk2, uk3 = uk.columns([1,1,1])
uk1.image(".\\pages\\manchester.jpeg")
uk2.image(".\\pages\\london.jpg")
uk3.image(".\\pages\\glasgow.jpg")

if uk1.button("MANCHESTER"):
    df, scores, df_size, new_df_size, r_auc, logreg_auc, nb_auc, rf_auc, r_fpr, r_tpr, logreg_fpr, logreg_tpr, nb_fpr, nb_tpr, rf_fpr, rf_tpr = search.tweet_search("manchester", 1, 1, 0.3)
    uk.title("RESULTS")
    uk.header("Traffic - Related tweets for MANCHESTER")
    uk.dataframe(df)
    col_metric1, col_metric2, col_metric3 = st.columns([1,1,1])
    col_metric1.metric(label="Trafic tweets fetched", value=df_size)
    col_metric2.metric(label="Tweets classified as 'Traffic'", value=new_df_size)
    csv = convert_df(df)
    col_metric3.download_button(label="Download the Tweets as a CSV", data=csv, file_name="search.csv", mime='text/csv')
    exp1 = uk.expander("SHOW ACCURACY SCORES FOR VARIOUS MODELS")
    exp1.dataframe(scores)
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
    uk.snow()
if uk2.button("LONDON"):
    df, scores, df_size, new_df_size, r_auc, logreg_auc, nb_auc, rf_auc, r_fpr, r_tpr, logreg_fpr, logreg_tpr, nb_fpr, nb_tpr, rf_fpr, rf_tpr = search.tweet_search("london", 1, 1, 0.3)
    uk.title("RESULTS")
    uk.header("Traffic - Related tweets for LONDON")
    uk.dataframe(df)
    col_metric1, col_metric2, col_metric3 = st.columns([1,1,1])
    col_metric1.metric(label="Trafic tweets fetched", value=df_size)
    col_metric2.metric(label="Tweets classified as 'Traffic'", value=new_df_size)
    csv = convert_df(df)
    col_metric3.download_button(label="Download the Tweets as a CSV", data=csv, file_name="search.csv", mime='text/csv')
    exp1 = uk.expander("SHOW ACCURACY SCORES FOR VARIOUS MODELS")
    exp1.dataframe(scores)
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
    uk.snow()
if uk3.button("GLASGOW"):
    df, scores, df_size, new_df_size, r_auc, logreg_auc, nb_auc, rf_auc, r_fpr, r_tpr, logreg_fpr, logreg_tpr, nb_fpr, nb_tpr, rf_fpr, rf_tpr = search.tweet_search("glasgow", 1, 1, 0.3)
    uk.title("RESULTS")
    uk.header("Traffic - Related tweets for GLASGOW")
    uk.dataframe(df)
    col_metric1, col_metric2, col_metric3 = st.columns([1,1,1])
    col_metric1.metric(label="Trafic tweets fetched", value=df_size)
    col_metric2.metric(label="Tweets classified as 'Traffic'", value=new_df_size)
    csv = convert_df(df)
    col_metric3.download_button(label="Download the Tweets as a CSV", data=csv, file_name="search.csv", mime='text/csv')
    exp1 = uk.expander("SHOW ACCURACY SCORES FOR VARIOUS MODELS")
    exp1.dataframe(scores)
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
    uk.snow()

st.markdown("<hr/>",True)


st.set_option('deprecation.showPyplotGlobalUse', False)