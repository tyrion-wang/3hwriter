import openai
import json
import os
import logging
import time

chat_model = "gpt-3.5-turbo"
edit_model = "text-davinci-edit-001"


def set_openai_key():
    """
    设置OpenAI API密钥
    """
    openai.api_key = os.environ.get('OPENAI_API_KEY')

def edit(input, instruction, temperature):
    """
        使用edit方法生成结果，主要用于文本的改写。

        Args:
            input (str): 需要改写的文本。
            instruction (str): 改写建议。
            temperature(num): 随机程度。
        Returns:
            str: 生成的文本响应。
    """
    response = openai.Edit.create(
        model=edit_model,
        input=input,
        temperature=temperature,
        instruction=instruction
    )
    return response.choices[0].text.strip()



def chat(input, instruction, temperature):
    """
    使用GPT-3.5-Turbo模型生成文本。

    Args:
        prompt (str): 模型生成文本的起始语句。
        model (str): 使用的模型名称。
        temperature (float): 控制随机性的温度值。默认值为0.5。

    Returns:
        str: 生成的文本响应。
    """
    start_time = time.time()
    # logging.info(f"开始请求OPEN AI API:{start_time}")
    response = openai.ChatCompletion.create(
        model=chat_model,
        temperature=temperature,
        stream=False,
        messages=[
            {"role": "user", "content": f"{instruction}:{input}"}
        ]
    )

    end_time = time.time()
    logging.info(f"结束请求OPEN AI API耗时：{end_time-start_time}")
    return response.choices[0].message.content.strip()

def chat_stream(input, instruction, temperature, socketio):
    """
    使用GPT-3.5-Turbo模型生成文本。

    Args:
        prompt (str): 模型生成文本的起始语句。
        model (str): 使用的模型名称。
        temperature (float): 控制随机性的温度值。默认值为0.5。

    Returns:
        str: 生成的文本响应。
    """
    for chunk in openai.ChatCompletion.create(
            model=chat_model,
            temperature=temperature,
            stream=True,
            messages=[
                {"role": "user", "content": f"{instruction}:{input}"}
            ],
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            socketio.emit('my response', content)
            print(content, end='')