from flask import Blueprint, render_template, abort, request, jsonify
from langchain.document_loaders import SeleniumURLLoader
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import textwrap

import langchain
langchain_test_page = Blueprint('langchain_test', __name__, template_folder='templates')
@langchain_test_page.route('/summary_url', methods=['GET', 'POST'])
def summary_url():
    send_message = ""
    if request.method == 'GET':
        send_message = request.values.get("send_message").strip()
    if request.method == 'POST':
        send_message = request.json["content"]
    print(send_message)

    decoded_text = decode_website(send_message)
    print("DEBUG decoded_text", decoded_text)
    summary = summarize_webpage(decoded_text)

    return jsonify({'msg': summary.strip()})


def decode_website(url):
    print("url", url)
    loader = SeleniumURLLoader([url])
    data = loader.load()
    web_text = ""

    for page in data:
        web_text += page.page_content + " "

    return web_text


def summarize_webpage(text):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(text)
    print(len(texts))
    docs = [Document(page_content=t) for t in texts[:4]]
    chain = load_summarize_chain(llm, chain_type="map_reduce")

    output_summary = chain.run(docs)
    wrapped_text = textwrap.fill(output_summary, width=100)
    print(wrapped_text)

    return wrapped_text


@langchain_test_page.route('/summary_doc', methods=['GET', 'POST'])
def summary_doc():
    print("DEBUG summary_doc")
    return jsonify({'msg': "DEBUG summary_doc"})