from flask import Flask, render_template, request
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser
from textblob import TextBlob  # Import TextBlob for sentiment analysis
import json
# Initialize Flask app
app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def determine_sentiment(polarity):
    if polarity > 0.75:
        return 'Extremely Positive (Cực kỳ tích cực)'
    elif polarity > 0.5:
        return 'Very Positive (Rất tích cực)'
    elif polarity > 0.25:
        return 'Moderately Positive (Khá tích cực)'
    elif polarity > 0.05:
        return 'Slightly Positive (Hơi tích cực)'
    elif polarity > -0.05:
        return 'Neutral (Trung lập)'
    elif polarity > -0.25:
        return 'Slightly Negative (Hơi tiêu cực)'
    elif polarity > -0.5:
        return 'Moderately Negative (Khá tiêu cực)'
    elif polarity > -0.75:
        return 'Very Negative (Rất tiêu cực)'
    else:
        return 'Extremely Negative (Cực kỳ tiêu cực)'



@app.route('/', methods=['GET'])
def index():
    # Display the main form
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text'] # your input
    action = request.form['action'] #show what action you chose

    with open('description.json', 'r') as file:
        pos_tags_description = json.load(file)

    # Initialize variables to hold results
    noun_phrases = None
    word_classification = None
    sentiment = None
    sentiment_description = "No sentiment analysis performed."  # Default message
    result_type = None

    tokens = word_tokenize(text) #use to separate sentences become each word
    tagged_tokens = pos_tag(tokens) #word classification

    if action == 'Extract Noun Phrases':
        grammar = r"Noun Phrases: {<DT>?<JJ>*<NN>}"
        chunk_parser = RegexpParser(grammar)
        parsed_tree = chunk_parser.parse(tagged_tokens)

        noun_phrases = [" ".join(word for word, tag in subtree.leaves())
                        for subtree in parsed_tree.subtrees()
                        if subtree.label() == 'Noun Phrases']
        result_type = 'noun_phrases'
    elif action == 'Classify Words':
        word_classification = [(word, pos_tags_description.get(tag, "Not Found")) for word, tag in tagged_tokens]
        result_type = 'word_classification'
    elif action == 'Analyze Sentiment':
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        sentiment_description = determine_sentiment(sentiment)
        result_type = 'sentiment_analysis'

    # Pass all variables to the template
    return render_template('result.html', noun_phrases=noun_phrases,
                           word_classification=word_classification,
                           sentiment=sentiment, result_type=result_type,
                           sentiment_description=sentiment_description)
if __name__ == '__main__':
    app.run(debug=True)
