from flask import Flask, render_template, request, jsonify, Response, session
import openai
import os
import json
import gpt_lib
import logging
from flask_socketio import SocketIO, emit
# import eventlet
# eventlet.monkey_patch()
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI

# import unstructured
from dotenv import load_dotenv

# 配置openai的API Key
gpt_lib.set_openai_key()
print(os.environ.get('OPENAI_API_KEY'))
# 初始化Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

logging.basicConfig(level=logging.DEBUG)
# logging.disable()
load_dotenv()

# embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
# embeddings = OpenAIEmbeddings(openai_api_key=str(os.environ.get('OPENAI_API_KEY')))
# persist_directory = '/Users/bytedance/Downloads/database/'
# docsearch = Chroma(embedding_function=embeddings, persist_directory=persist_directory)


# 定义首页
@app.route('/')
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
    #transcription = gpt_lib.chat(text, "围绕这个命题，生成一个800字的作文：", temperature)
    transcription = gpt_lib.chat(text, "总结这段文本，10个字以内：", temperature)
    #gpt_lib.chat_stream(text, "围绕这个命题，生成一个800字的作文：", temperature, socketio)
    # gpt_lib.chat_stream(text, "总结这段文本", temperature, socketio)
    # transcription = "123"
    # 返回json格式的结果
    return jsonify({'transcription': transcription.strip()})

# @app.route('/write', methods=['GET', 'POST'])
# def write():
#     print("write")

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
    prompt = "写一篇2000字的关于气候变暖的论文。"
    return prompt


def stream(input_text):
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
            {"role": "system", "content": "You're an assistant."},
            {"role": "user", "content": f"{prompt(input_text)}"},
        ], stream=True, max_tokens=4000, temperature=0)
        for line in completion:
            if 'content' in line['choices'][0]['delta']:
                yield line['choices'][0]['delta']['content']

@app.route('/completion', methods=['GET', 'POST'])
def completion_api():
    # print("111")
    # return "test return"
    if request.method == "POST":
        data = request.form
        input_text = data['input_text']
        return Response(stream(input_text), mimetype='text/event-stream')
    else:
        return Response(None, mimetype='text/event-stream')


########################################################
import uuid
import requests
STREAM_FLAG = True  # 是否开启流式推送
CHAT_CONTEXT_NUMBER_MAX = 12
# API_KEY = os.environ.get('OPENAI_API_KEY')
API_KEY = os.environ.get('OPENAI_API_KEY')
def check_session(current_session):
    """
    检查session，如果不存在则创建新的session
    :param current_session: 当前session
    :return: 当前session
    """
    if current_session.get('session_id') is not None:
        print("existing session, session_id:\t", current_session.get('session_id'))
    else:
        current_session['session_id'] = uuid.uuid1()
        print("new session, session_id:\t", current_session.get('session_id'))
    return current_session['session_id']

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

    print(f"len(message_context): {len(message_context)} total: {total}",)
    return message_context

def get_response_from_ChatGPT_API(message_context, apikey):
    """
    从ChatGPT API获取回复
    :param apikey:
    :param message_context: 上下文
    :return: 回复
    """
    if apikey is None:
        apikey = API_KEY

    header = {"Content-Type": "application/json",
              "Authorization": "Bearer " + apikey}

    data = {
        "model": "gpt-3.5-turbo",
        "messages": message_context
    }
    url = "https://api.openai.com/v1/chat/completions"

    try:
        response = requests.post(url, headers=header, data=json.dumps(data))
        response = response.json()
        # 判断是否含 choices[0].message.content
        if "choices" in response \
                and len(response["choices"]) > 0 \
                and "message" in response["choices"][0] \
                and "content" in response["choices"][0]["message"]:
            data = response["choices"][0]["message"]["content"]
        else:
            data = str(response)

    except Exception as e:
        print(e)
        return str(e)

    return data

def get_response_stream_generate_from_ChatGPT_API(message_context, apikey, message_history):
    """
    从ChatGPT API获取回复
    :param apikey:
    :param message_context: 上下文
    :return: 回复
    """
    if apikey is None:
        apikey = API_KEY

    header = {"Content-Type": "application/json",
              "Authorization": "Bearer " + apikey}

    data = {
        "model": "gpt-3.5-turbo",
        "messages": message_context,
        "stream": True
    }
    print("开始流式请求")
    url = "https://api.openai.com/v1/chat/completions"
    # 请求接收流式数据 动态print
    try:
        response = requests.request("POST", url, headers=header, json=data, stream=True)

        def generate():
            stream_content = str()
            one_message = {"role": "assistant", "content": stream_content}
            message_history.append(one_message)
            i = 0
            for line in response.iter_lines():
                # print(str(line))
                line_str = str(line, encoding='utf-8')
                if line_str.startswith("data:"):
                    if line_str.startswith("data: [DONE]"):
                        # asyncio.run(save_all_user_dict())
                        break
                    line_json = json.loads(line_str[5:])
                    if 'choices' in line_json:
                        if len(line_json['choices']) > 0:
                            choice = line_json['choices'][0]
                            if 'delta' in choice:
                                delta = choice['delta']
                                if 'role' in delta:
                                    role = delta['role']
                                elif 'content' in delta:
                                    delta_content = delta['content']
                                    i += 1
                                    if i < 40:
                                        print(delta_content, end="")
                                    elif i == 40:
                                        print("......")
                                    one_message['content'] = one_message['content'] + delta_content
                                    yield delta_content

                elif len(line_str.strip()) > 0:
                    print(line_str)
                    yield line_str

    except Exception as e:
        ee = e

        def generate():
            yield "request error:\n" + str(ee)

    return generate

def handle_messages_get_response(message, apikey, message_history, have_chat_context, chat_with_history):
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
    response = get_response_from_ChatGPT_API(message_context, apikey)
    message_history.append({"role": "assistant", "content": response})
    # 换行打印messages_history
    # print("message_history:")
    # for i, message in enumerate(message_history):
    #     if message['role'] == 'user':
    #         print(f"\t{i}:\t{message['role']}:\t\t{message['content']}")
    #     else:
    #         print(f"\t{i}:\t{message['role']}:\t{message['content']}")

    return response

def handle_messages_get_response_stream(message, apikey, message_history, have_chat_context, chat_with_history):
    message_history.append({"role": "user", "content": message})
    message_context = get_message_context(message_history, have_chat_context, chat_with_history)
    generate = get_response_stream_generate_from_ChatGPT_API(message_context, apikey, message_history)
    return generate

@app.route('/returnMessage', methods=['GET', 'POST'])
def return_message():
    """
    获取用户发送的消息，调用get_chat_response()获取回复，返回回复，用于更新聊天框
    :return:
    """
    check_session(session)
    send_message = request.values.get("send_message").strip()
    send_time = request.values.get("send_time").strip()
    url_redirect = "url_redirect:/"
    if send_message == "帮助":
        return "### 帮助\n" \
               "1. 输入`new:xxx`创建新的用户id\n " \
               "2. 输入`id:your_id`切换到已有用户id，新会话时无需加`id:`进入已有用户\n" \
               "3. 输入`set_apikey:`[your_apikey](https://platform.openai.com/account/api-keys)设置用户专属apikey，`set_apikey:none`可删除专属key\n" \
               "4. 输入`rename_id:xxx`可将当前用户id更改\n" \
               "5. 输入`查余额`可获得余额信息及最近几天使用量\n" \
               "6. 输入`帮助`查看帮助信息"

    user_id = session.get('user_id')
    print(f"用户({user_id})发送消息:{send_message}")
    # user_info = get_user_info(user_id)
    chat_id = '001'
    messages_history = [{"role": "assistant", "content": "123"},
                        {"role": "assistant", "content": "#### 当前浏览器会话为首次请求\n"
                                                         "#### 请输入已有用户`id`或创建新的用户`id`。\n"
                                                         "- 已有用户`id`请在输入框中直接输入\n"
                                                         "- 创建新的用户`id`请在输入框中输入`new:xxx`,其中`xxx`为你的自定义id，请牢记\n"
                                                         "- 输入`帮助`以获取帮助提示"}]
    chat_with_history = False
    apikey = API_KEY
    # if chat_with_history:
    #     # user_info['chats'][chat_id]['have_chat_context'] += 1
    if send_time != "":
        messages_history.append({'role': 'system', "content": send_time})
    if not STREAM_FLAG:
        content = handle_messages_get_response(send_message, apikey, messages_history,
                                               '123',
                                               chat_with_history)

        print(f"用户({session.get('user_id')})得到的回复消息:{content[:40]}...")
        # if chat_with_history:
        #     # user_info['chats'][chat_id]['have_chat_context'] += 1
        # 异步存储all_user_dict
        # asyncio.run(save_all_user_dict())
        return content
    else:
        generate = handle_messages_get_response_stream(send_message, apikey, messages_history,
                                                       '123',
                                                       chat_with_history)

        # if chat_with_history:
        #     user_info['chats'][chat_id]['have_chat_context'] += 1

        return app.response_class(generate(), mimetype='application/json')




if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
