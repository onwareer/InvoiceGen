import openai
import os
# Load environment variables
os.getenv("OPENAI_API_KEY")
#openai.api_key = "your-api-key"

# JSON-style input data
sample_data = {
    "client_name": "Jane Doe",
    "valuation": 1000000,
    "management_fee": 2000,
    "performance_fee": 1000,
    "quarter": "Q2-2025"
}

# Prompt template
prompt_template = """
Generate a plain-English invoice summary for the following client:

Client Name: {client_name}
Quarter: {quarter}
Valuation: ${valuation}
Management Fee: ${management_fee}
Performance Fee: ${performance_fee}

Explain what each fee means and provide a professional but friendly tone.
"""


# Function to generate summary using GPT
def generate_summary(data):
    prompt = prompt_template.format(**data)

    response = openai.ChatCompletion.create(
        model="gpt-4",  #or gpt-3.5-turbo
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )

    return response.choices[0].message['content']


# Run and display the result
summary = generate_summary(sample_data)
print("Generated Summary:\n")
print(summary)

