import requests
import os
from OpenAI.api_key import key


headers={
    'Authorization':'Bearer '+key
}

result=requests.get('https://api.openai.com/v1/engines',headers=headers)

print(result.json())