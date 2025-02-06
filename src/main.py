from openai_client import openai_client

# Test OpenAI API with ChatGPT
response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)

# Print response
print(response.choices[0])