import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import plotly.express as px
import plotly.graph_objects as go
import re
import nltk
import string
import numpy as np
import pandas as pd
import streamlit as st
import pandasql as pl
import seaborn as sea
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

st.sidebar.title("Select the model")
sheet = st.sidebar.selectbox("Select an option", ["Twitter Sentiment Analysis", "Product Inventory Prediction"])
show_source_data = st.sidebar.checkbox("Show Source Data", value=True)
if sheet == "Product Inventory Prediction":
    data = pd.read_csv('grouped.csv')
    data = data.rename(columns = {'vendor_id' : 'vendor_count',
                                                  'promo_id' : 'promo_count'})

    #X = data[['Month','Year','week_number',
                    #  'product_id','vendor_count','total_price']]
    #y = data['quantity']
    #X_train, X_test, y_train, y_test = train_test_split(X, y,
                     #                                   test_size = 0.3,
                      #                                  random_state = 1)
    #Reg = RandomForestRegressor()
    #Reg.fit(X_train, y_train)
    #pred = Reg.predict(X_test)
    #joblib.dump(Reg, 'model_Reg.pkl', protocol=2)


    #prediction = pd.DataFrame({'Actual_quantity' : y_test,
                               #'Predicted_quantity' : pred})
    #prediction.Predicted_quantity = prediction.Predicted_quantity.astype(int)
    #prediction.head(2)


    def pred(a, b, c, d, e, f):
        model = joblib.load("model_Reg.pkl")
        array = pd.DataFrame({'Month' : [a], 'Year' : [b], 'week_number' : [c],
                              'product_id' : [d], 'vendor_count' : [e],
                              'total_price' : [f]})
        a = model.predict(array).astype(int)   
        st.write("The minimum number of products you should keep in your inventry are",a[0])

    st.title("Product Inventory Prediction")

    def product_details(x):
        raw_data = pd.DataFrame(data.loc[data.product_id == x, ['Month', 'Year','week_number', 'product_id',
                                                                'quantity','total_price',
                                                                'vendor_count']])
        return raw_data

    Product_id = st.number_input("Enter the Product id to visualize")
    raw_da = product_details(Product_id)

    raw_data_p = pd.DataFrame(raw_da.loc[raw_da.total_price != 0, ['Month', 'Year','week_number',
                                                                       'product_id',
                                                                'quantity','total_price',
                                                               'vendor_count']])
    maximum = 91
    minimum = 1
    if Product_id:
        a = st.slider("Slide for week", min_value = minimum, max_value = maximum)
        #b = st.slider("Slide for Month", min_value = 1, max_value = 12)
        data = raw_data_p[raw_data_p["week_number"] == a]
        st.write(data)
        fig = px.bar(raw_data_p, x="Month", y="quantity", title='Product quantity in different months')
        st.plotly_chart(fig)
        
       

        #fig = go.Figure()

        #fig.add_trace(
        #    go.Scatter(
         #       x=raw_data_p['week_number']
          #  ))

        #fig.add_trace(KAdbcd
         #   go.Bar(
          #      x=raw_data_p["Month"],
           #     y=raw_data_p["quantity"]
            #))

        #st.plotly_chart(fig)

    st.subheader("Enter the month in which you want to predict product.")
    input1 = st.number_input("Month")
    st.subheader("Enter the year in which you want to predict product.")
    input2 = st.number_input("Year")
    st.subheader("Enter the week number in which you want to predict product.")
    input3 = st.number_input("Week Number")
    st.subheader("Enter the Product ID.")
    input4 = st.number_input("Product")
    st.subheader("Enter the number of Vendors.")
    input5 = st.number_input("Vendors")
    st.subheader("Enter the Price that you'll be setting in future.")
    input6 = st.number_input("Price")

    if input1 and input2 and input3 and input4 and input5 and input6:
        if st.button("Predict Products"):
            pred(input1, input2, input3, input4, input5, input6)
        

if sheet == "Twitter Sentiment Analysis":
    train = pd.read_csv("train_E6oV3lV.csv")

    #tweet_token = merged.tweets.apply(lambda l : l.split())

    from nltk.stem.porter import *
    stemmed = PorterStemmer()
    #tweet_token = tweet_token.apply(lambda l : [stemmed.stem(i) for i in l])

    from sklearn.feature_extraction.text import TfidfVectorizer
    #tf_vectors = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1200, stop_words='english') 
    #tf_vector = tf_vectors.fit_transform(merged.tweet)



    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        return input_txt
    #remove_pattern(merged['tweet'][0], "@[\w]*")

    train['tweet'] = np.vectorize(remove_pattern)(train['tweet'], "([@])\w+")

    pun_list = string.punctuation
    punc_list = []
    for i in pun_list:
        mark = '#'
        if i == mark:
            continue
        else:
            punc_list.append(i)

    def find_pattern(text):
        pattern = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
        common  = re.findall(pattern, text)
        for com in common:
            text = text.replace(com[0], ', ')    
        return text

    def remove_mention(text):
        mention = ['@']
        for punc in  punc_list:
            if punc not in mention :
                text = text.replace(punc,' ')
        words = []
        for word in text.split():
            word = word.strip()
            if word:
                if word[0] not in mention:
                    words.append(word)
        return ' '.join(words)


    r_mention = []
    for i in train['tweet']:
        a = remove_mention(find_pattern(i))
        r_mention.append(a)

    train.tweet = r_mention

    Full_cleaned_tweet = []
    for i in train['tweet']:
        b = i.replace('user', '')
        Full_cleaned_tweet.append(b)


    train.tweet = Full_cleaned_tweet

    tweets = []
    for i in train['tweet']:
        a = re.sub('\d+', "", i)
        tweets.append(a)

    train.tweet = tweets

    train['tweet'] = train['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

    import nltk.corpus
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    train['tweet_without_stopwords'] = train['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


    st.title("Twitter Sentiment Analysis")
    st.subheader("Select the options")

    @st.cache(suppress_st_warning = True)
    def graph(data):
        if (data == "negative_tweets"):
            negative_tweets = pl.sqldf("SELECT tweet FROM train where label = 1",globals())
            words = ' '.join([t for t in negative_tweets.tweet])
            wc = WordCloud(width = 1000, height = 800, random_state = 18, max_font_size = 130,background_color = None, mode = 'RGBA').generate(words)
            plt.imshow(wc, interpolation = 'bilinear')
            plt.axis('off')
            st.pyplot()
        elif (data == "positive_tweets"):
            positive_tweets = pl.sqldf("SELECT tweet FROM train where label = 0",globals())
            words = ' '.join([t for t in positive_tweets.tweet])
            wc1 = WordCloud(width = 1000, height = 800, random_state = 18, max_font_size = 130,background_color = None, mode = 'RGBA').generate(words)
            plt.imshow(wc1, interpolation = 'bilinear')
            plt.axis('off')
            st.pyplot()

    #user_input = st.text_input("Enter which type of tweets you wanna see.")
    user_input = st.selectbox("",options = ["negative_tweets", "positive_tweets"], key=None)
    if user_input:
        graph(user_input)

    def checking(text, patterrn):
        tweets =[]
        for t in text:
            com_text = [p for p in pattern if pattern in t]
            if(len(com_text) == len(pattern)):
                tweets.append(t)
        return tweets

    text = train.tweet_without_stopwords
    st.subheader("Enter the hashtag")
    pattern = st.text_input('')

    @st.cache(suppress_st_warning = True)
    def graph3():
        if pattern:
            related_words = checking(text, pattern)
            related_word = ' '.join([t for t in related_words])
            wc3 = WordCloud(width = 800, height = 600, random_state = 21, max_font_size = 100,background_color = None, mode = 'RGBA').generate(related_word)
            plt.figure(figsize = (15,10))
            plt.imshow(wc3, interpolation = 'bilinear')
            plt.axis('off')
            st.pyplot()
    graph3()

    def extract_ht(a):
        hashtag = []
        for i in a:
            ht = re.findall("#(\w+)", i)
            hashtag.append(ht)
        return hashtag

    negative_tweets = pl.sqldf("SELECT tweet FROM train where label = 1",globals())
    positive_tweets = pl.sqldf("SELECT tweet FROM train where label = 0",globals())

    st.subheader("Select the option for hashtag")

    @st.cache(suppress_st_warning = True)
    def graph2(data, value):
        if data == "Positive Hashtag":
            positive_ht = extract_ht(positive_tweets.tweet)
            positive_ht = sum(positive_ht,[])
            freq_values = nltk.FreqDist(positive_ht) 
            data = pd.DataFrame({'Hashtags': list(freq_values.keys()),'No_of_count': list(freq_values.values())})    
            data = data.nlargest(columns="No_of_count", n = value) 
            plt.figure(figsize=(16,5))
            ax = sea.barplot(data=data, x= "Hashtags", y = "No_of_count")
            ax.set(ylabel = 'Number of Count')
            st.pyplot()

        elif data == "Negative Hashtag":
            negative_ht = extract_ht(negative_tweets.tweet)
            negative_ht = sum(negative_ht,[])
            freq_values = nltk.FreqDist(negative_ht) 
            data = pd.DataFrame({'Hashtags': list(freq_values.keys()),'No_of_count': list(freq_values.values())})    
            data = data.nlargest(columns="No_of_count", n = value) 
            plt.figure(figsize=(16,5))
            ax = sea.barplot(data=data, x= "Hashtags", y = "No_of_count")
            ax.set(ylabel = 'Number of Count')
            st.pyplot()

    user_input2 = st.selectbox("",options = ["Positive Hashtag", "Negative Hashtag"], key=None)
    st.subheader("Enter a number")
    user_input3 = st.number_input("Number of Hashtags you want to see")
    if user_input2 and user_input3:
        graph2(user_input2, user_input3)


    from sklearn.feature_extraction.text import HashingVectorizer
    token = HashingVectorizer(ngram_range=(1, 3), stop_words=stop)
    #all_token = token.fit_transform(train.tweet)
    #x_train, x_test, Y_train, Y_test = train_test_split(all_token, train.label, random_state = 777, test_size=.25)
    #SVC = LinearSVC(verbose = 1)
    #SVC.fit(all_token, train.label)
    #joblib.dump(SVC, 'model_SVM.pkl', protocol=2)




    @st.cache(suppress_st_warning = True)
    def pred_new(text):
        new_model = joblib.load('model_SVM.pkl')
        stop = stopwords.words('english')
        text = ' '.join([word for word in text.split() if word not in (stop)])
        print(text)
        text = token.transform([text])
        predicted = new_model.predict(text)
        pred = predicted[0]
        pred = 'racist' if predicted[0]==1 else 'non-racist'
        return pred

    
    st.subheader("Enter the tweet for sentiment analysis")

    tweet1 = st.text_input('Tweet in the box :')
    #tweet1 = st.write(title)

    if st.button("Predict Sentiment"):
        predict = pred_new(tweet1)
        if predict == 'racist':
            image = Image.open('racist.jpeg')
            st.image(image, caption='Racist',use_column_width=True)
        else:
            image = Image.open('no_racist.jpeg')
            st.image(image, caption='Non-Racist',use_column_width=True)
