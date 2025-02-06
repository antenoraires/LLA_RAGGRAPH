
from dotenv import load_dotenv
import os

# Carrega as variáveis do arquivo .env
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

print(OPENAI_API_KEY)