import gradio as gr
from transformers import pipeline

# task = sentiment-analysis
# model = if model not spcified by defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b 

# Function to perform sentiment analysis
def sentiment_classifier(text):
    try:
        sentiment_classifier = pipeline("sentiment-analysis")
        sentiment_response = sentiment_classifier(text)
        label = sentiment_response[0]['label']
        score = sentiment_response[0]['score']
        return label, score
    except Exception as e:
        return str(e)

# Create Gradio interface
input_text = gr.Textbox(lines=10, label="Input Text", placeholder="Enter text for sentiment analysis...")
output_label = gr.Textbox(label="Sentiment Label", placeholder="Sentiment label will appear here...")
output_score = gr.Textbox(label="Sentiment Score", placeholder="Sentiment score will appear here...")

# Author information
author = "Ajeetkumar Ukande"

# Create Gradio interface
interface = gr.Interface(sentiment_classifier, inputs=input_text, outputs=[output_label, output_score], 
             title="<div style='color: #336699; font-size: 24px; font-weight: bold; border: 2px solid #336699; padding: 10px; border-radius: 10px;'>Sentiment Analysis</div>", 
             description=f"""<div style='color: #666666; font-family: Arial, sans-serif;'>
                             <p style='margin-top: 10px;'>Enter some text for sentiment analysis.</p>
                             <p>Developed by <span style='color: #336699; font-weight: bold;'>{author}</span>.</p>
                             </div>""", 
             theme="default" # Change theme to default
             )

# Deploy the interface to Hugging Face Spaces
interface.launch(debug=True)
