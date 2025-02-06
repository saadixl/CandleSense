import openai
from config import OPENAI_API_KEY

# Create an OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)