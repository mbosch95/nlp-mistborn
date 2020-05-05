from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


def word_cloud(data, width=400, height=400, background_color='white', min_font_size=10):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=width, height=height, 
                    background_color=background_color, 
                    stopwords=stopwords, 
                    min_font_size=min_font_size).generate_from_frequencies(data) 

    return wordcloud

def print_image(figure, figsize=(4, 4), pad=0):
    plt.figure(figsize=figsize, facecolor=None) 
    plt.imshow(figure) 
    plt.axis("off") 
    plt.tight_layout(pad=pad) 

    plt.show() 

def print_bar(data):
    plt.bar(range(len(data)), list(data.values()), align='center')
    plt.xticks(range(len(data)), list(data.keys()))

    plt.show()
