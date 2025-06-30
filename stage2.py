
import requests

# Replace with your actual values
API_KEY = 'ezDqBEZN.5iVQlBt4QC0Ka7GBTcf8KIX5KBFyKA1P'
ASSISTANT_ID = 1164  # or the actual assistant ID you're targeting
TEXT_PROMPT = 'ОСЫ СӨЙЛЕМДЕГІ ГРАММАТИКАЛЫҚ ҚАТЕЛЕРДІ ТАУЫП, ДҰРЫСТАП, ТҮСІНДІР: Сәлеметсіз бе! Мен атым Аңсар.'

# API URL
url = f'https://oylan.nu.edu.kz/api/v1/assistant/{ASSISTANT_ID}/interactions/'

# Request headers
headers = {
    'accept': 'application/json',
    'Authorization': f'Api-Key {API_KEY}',
}

# Form data payload
data = {
    'content': TEXT_PROMPT,
    'assistant': str(ASSISTANT_ID),
    'stream': 'false',  # or 'true' if you support streaming
}

# Send the POST request
response = requests.post(url, headers=headers, data=data)

# Handle the response
if response.status_code in (200, 201):
    json_response = response.json()
    print("Assistant response:", json_response['response']['content'])
else:
    print(f"Error {response.status_code}:", response.text)
