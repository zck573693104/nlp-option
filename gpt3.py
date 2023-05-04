#-*- coding:utf-8 -*-
import os

import requests
import json

url = "api"


def gpt3_learn(text):
  if text is None or text == '':
    text = '你好'
  payload = json.dumps({
    "prompt": text,
    "userId": "",
    "network": True,
    "apikey": "",
    "system": ""
  })
  headers = {
    'Content-Type': 'application/json'
  }
  response = requests.request("POST", url, headers=headers, data=payload)
  return response.text

if __name__ == "__main__":
  text = '''
你是猪猪侠嘛
  '''
  print(gpt3_learn(text))
