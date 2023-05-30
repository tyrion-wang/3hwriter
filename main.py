import flask
from flask import Flask, render_template, request, jsonify, Response, session
import openai
import os
import json
import gpt_lib
# import logging

# 配置openai的API Key
gpt_lib.set_openai_key()
# 初始化Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
# logging.basicConfig(level=logging.DEBUG)
# logging.disable()

# 定义首页
@app.route('/test', methods=['GET', 'POST'])
def index_test():
    return render_template('index2.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

# 定义转写函数
@app.route('/transcribe', methods=['POST'])
def transcribe():
    # 获取用户输入的文字
    text = request.form['text']
    # 获取用户选择的相似度
    similarity = request.form['similarity']
    temperature = 1.0 - float(similarity) / 10.0
    # transcription = gpt_lib.chat(text, "围绕这个命题，生成一个800字的作文：", temperature)
    transcription = gpt_lib.chat(text, "总结这段文本，10个字以内：", temperature)
    # gpt_lib.chat_stream(text, "围绕这个命题，生成一个800字的作文：", temperature, socketio)
    # gpt_lib.chat_stream(text, "总结这段文本", temperature, socketio)
    # transcription = "123"
    # 返回json格式的结果
    return jsonify({'transcription': transcription.strip()})


def gen_prompt(docs, query) -> str:
    return f"""To answer the question please only use the Context given, nothing else. Do not make up answer, simply say 'I don't know' if you are not sure.
Question: {query}
Context: {[doc.page_content for doc in docs]}
Answer:
"""


def prompt(query):
    # print(query)
    # docs = docsearch.similarity_search(query, k=4)
    # print(docs)
    # prompt = gen_prompt(docs, query)
    # prompt = query
    prompt = "500字总结2000年的全球大事件。"
    return prompt


def stream(input_text):
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
        {"role": "system", "content": "You're an assistant."},
        {"role": "user", "content": f"{prompt(input_text)}"},
    ], stream=True, max_tokens=4000, temperature=0)
    for line in completion:
        if 'content' in line['choices'][0]['delta']:
            yield line['choices'][0]['delta']['content']

@app.route('/completion', methods=['GET'])
def completion():
    def stream():
        completion = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[{"role": "user", "content": "500字总结2000年的全球大事件。"}],
            stream=True)
        for line in completion:
            if line['choices'][0]['finish_reason'] is not None:
                chunk = '[DONE]'
            else:
                chunk = line['choices'][0].get('delta', {}).get('content', '')
            if chunk:
                yield 'data: %s\n\n' % chunk
    return flask.Response(stream(), mimetype='text/event-stream')

#####################################################

STREAM_FLAG = True  # 是否开启流式推送
CHAT_CONTEXT_NUMBER_MAX = 12
API_KEY = os.environ.get('OPENAI_API_KEY')

def get_message_context(message_history, have_chat_context, chat_with_history):
    """
    获取上下文
    :param message_history:
    :param have_chat_context:
    :param chat_with_history:
    :return:
    """
    message_context = []
    total = 0
    if chat_with_history:
        num = min([len(message_history), CHAT_CONTEXT_NUMBER_MAX, have_chat_context])
        # 获取所有有效聊天记录
        valid_start = 0
        valid_num = 0
        for i in range(len(message_history) - 1, -1, -1):
            message = message_history[i]
            if message['role'] in {'assistant', 'user'}:
                valid_start = i
                valid_num += 1
            if valid_num >= num:
                break

        for i in range(valid_start, len(message_history)):
            message = message_history[i]
            if message['role'] in {'assistant', 'user'}:
                message_context.append(message)
                total += len(message['content'])
    else:
        message_context.append(message_history[-1])
        total += len(message_history[-1]['content'])

    # print(f"len(message_context): {len(message_context)} total: {total}", )
    return message_context

def get_response_stream_generate_from_ChatGPT_API_V2(message_context):
    """
    从ChatGPT API获取回复
    :param apikey:
    :param message_context: 上下文
    :return: 回复
    """
    def stream():
        completion = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=message_context,
            stream=True)
        for line in completion:
            if line['choices'][0]['finish_reason'] is not None:
                chunk = '[DONE]'
            else:
                chunk = line['choices'][0].get('delta', {}).get('content', '')
            if chunk:
                # yield 'data: %s\n\n' % chunk
                yield 'event: delta\ndata: %s\n\n' % chunk
                # yield "{'event: delta\n\n, 'data: %s\n\n' % chunk}";
    return stream

def get_response_from_ChatGPT_API_V2(message_context):
    """
    从ChatGPT API获取回复
    :param message_context: 上下文
    :return: 回复
    """
    def stream():
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # temperature=temperature,
            stream=False,
            messages=message_context
        )
        # 判断是否含 choices[0].message.content
        if "choices" in response \
                and len(response["choices"]) > 0 \
                and "message" in response["choices"][0] \
                and "content" in response["choices"][0]["message"]:
            yield 'data: %s\n\n' % response["choices"][0]["message"]["content"]
            yield 'data: [DONE]\n\n'
        else:
            yield 'data: %s\n\n' % str(response)
    return stream

def handle_messages_get_response(message, message_history, have_chat_context, chat_with_history):
    """
    处理用户发送的消息，获取回复
    :param message: 用户发送的消息
    :param apikey:
    :param message_history: 消息历史
    :param have_chat_context: 已发送消息数量上下文(从重置为连续对话开始)
    :param chat_with_history: 是否连续对话
    """
    message_history.append({"role": "user", "content": message})
    message_context = get_message_context(message_history, have_chat_context, chat_with_history)
    response = get_response_from_ChatGPT_API_V2(message_context)
    print(response)
    return response


def handle_messages_get_response_stream(message, message_history, have_chat_context, chat_with_history):
    message_history.append({"role": "user", "content": message})
    message_context = get_message_context(message_history, have_chat_context, chat_with_history)
    generate = get_response_stream_generate_from_ChatGPT_API_V2(message_context)
    return generate


@app.route('/returnMessage', methods=['GET', 'POST'])
def return_message():
    """
    获取用户发送的消息，调用get_chat_response()获取回复，返回回复，用于更新聊天框
    :return:
    """
    send_message = ""
    if request.method == 'GET':
        send_message = request.values.get("send_message").strip()
    if request.method == 'POST':
        send_message = request.json["content"]

    messages_history = [{"role": "assistant", "content": "1.有2个苹果"},
                        {"role": "assistant", "content": "2.有3个梨子"},
                        {"role": "assistant", "content": "3.有2只鸡"}]
    chat_with_history = True

    if STREAM_FLAG:
        generate = handle_messages_get_response_stream(send_message, messages_history, CHAT_CONTEXT_NUMBER_MAX, chat_with_history)
        return flask.Response(generate(), mimetype='text/event-stream')
    else:
        generate = handle_messages_get_response(send_message, messages_history, CHAT_CONTEXT_NUMBER_MAX, chat_with_history)
        return flask.Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=os.getenv("PORT", default=5000), debug=False)
    # app.run(debug=False, port=os.getenv("PORT", default=5000))
