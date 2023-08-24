import yaml
import json
import requests

with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

api_key = config['api_key']

API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": f"Bearer {api_key}"}


def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


data = query("Can you please let us know more details about your ")

print(data)

data = query("Tell me about how about Taiwan's weather?")

print(data)
