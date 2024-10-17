import re
import os
from langchain_together import ChatTogether
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain import LLMChain
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
load_dotenv()
api_key=os.getenv("API_KEY")
app = Flask(__name__)

#initialize llm
llm = ChatTogether(api_key=api_key,temperature=0.0, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

def is_youtube_url(url):
    #regex pattern to match youtube urls
    youtube_regex = re.compile(
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

    #match the url against the regex pattern
    match = youtube_regex.match(url)

    return bool(match)

def summarise(video_url):
    #loader to get youtube video transcript
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
    data = loader.load()

    #prompt template in which we are passing the video transcript
    #why prompt template?
    #prompt template is used to format the prompt,
    # i.e it is used to tell the llm what to do with the data
    #in this case, the prompt template is used to tell the llm to summarise the video transcript
    #the video transcript from loader is passed to the prompt template
    #the prompt template is then passed to the llm
    #the llm will use the prompt template to summarise the video transcript
    
    product_description_template = PromptTemplate(
        input_variables=["video_transcript"],
        template="""
        Read through the entire transcript carefully.
    Provide a concise summary of the video's main topic and purpose.
    Extract and list the five most interesting or important points from the transcript. For each point: State the key idea clearly and concisely

    Ensure your summary and key points capture the essence of the video without including unnecessary details.
    Use clear, engaging language that is accessible to a general audience.
    If the transcript includes any statistical data, expert opinions, or unique insights, prioritize including these in your summary or key points.

        video transcript: {video_transcript}    """
    )

    #llmchain,here we are creating a chain using llm and prompt template,
    # i.e the prompt template defined earlier is passed to llm
    #why llmchain?
    #llmchain is used to create a chain using llm and prompt template
    #the chain will use the prompt template to summarise the video transcript
    #the video transcript is passed to the chain
    #the chain will generate the summary and return it
    #the summary is stored in the text variable
    #the summary is then returned`6`
    chain = LLMChain(llm=llm, prompt=product_description_template)

    #summary from llmchain,here we are passing the video transcript to the chain,
    #the chain will generate the summary and return it
    #data[0].page_content contains the video transcript
    #the summary is stored in the text variable
    #the summary is then returned
    summary = chain.invoke({
        "video_transcript": data[0].page_content
    })
    #order of execution
    #YoutubeLoader is initialized with the YouTube video URL
    #Loader loads the YouTube video transcript and metadata
    #PromptTemplate is created with instructions for summarization
    #LLMChain is created, combining the language model (llm) and the PromptTemplate
    #The video transcript is passed to the LLMChain via the invoke method
    #The LLMChain processes the transcript using the LLM and generates the summary
    #The generated summary is stored in the summary variable
    #The summary is then returned from the function
    return (summary['text'])

# Route to test if server is running
@app.route("/ping", methods=['GET'])
def pinger():
    return "<p>Hello world!</p>"

# Route to summarize YouTube video from a given URL
@app.route('/summary', methods=['POST'])
def summary():
    url = request.form.get('Body')  # Get the JSON data from the request body
    print(url)
    if is_youtube_url(url):
        response = summarise(url)
    else:
        response = "Please check if this is a correct youtube video url"
    print(response)
    resp = MessagingResponse()
    msg = resp.message()
    msg.body(response)
    return str(resp)

# Run the Flask app
if __name__ == '__main__':
    app.run(port=4040)



