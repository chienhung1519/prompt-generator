from anthropic import Anthropic
from openai import OpenAI
import re
import streamlit as st

from prompt import ANTHROPIC_META_PROMPT, OPENAI_META_PROMPT

# Parameters
ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"
OPENAI_MODEL = "gpt-4o"

# Initilize Anthropic and OpenAI clients
anthropic_client = Anthropic()
openai_client = OpenAI()

# Anthropic helper functions
def extract_between_tags(tag: str, string: str, strip: bool = False) -> list[str]:
    ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
    if strip:
        ext_list = [e.strip() for e in ext_list]
    return ext_list

def remove_empty_tags(text):
    return re.sub(r'\n<(\w+)>\s*</\1>\n', '', text, flags=re.DOTALL)

def strip_last_sentence(text):
    sentences = text.split('. ')
    if sentences[-1].startswith("Let me know"):
        sentences = sentences[:-1]
        result = '. '.join(sentences)
        if result and not result.endswith('.'):
            result += '.'
        return result
    else:
        return text

def extract_prompt(metaprompt_response):
    between_tags = extract_between_tags("Instructions", metaprompt_response)[0]
    return between_tags[:1000] + strip_last_sentence(remove_empty_tags(remove_empty_tags(between_tags[1000:]).strip()).strip())

# Generate prompt
def generate_prompt_by_anthropic(task: str):
    prompt = ANTHROPIC_META_PROMPT.replace("{{TASK}}", task)
    assistant_partial = "<Inputs>\n</Inputs>\n<Instructions Structure>"
    message = anthropic_client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content":  prompt
            },
            {
                "role": "assistant",
                "content": assistant_partial
            }
        ],
        temperature=0
    ).content[0].text
    print(message)
    return extract_prompt(message)

def generate_prompt_by_openai(task: str):
    completion = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": OPENAI_META_PROMPT,
            },
            {
                "role": "user",
                "content": "Task, Goal, or Current Prompt:\n" + task,
            },
        ],
    )
    return completion.choices[0].message.content

# UI
# Title
st.title("Prompt Generator")

# Select model
model = st.selectbox(
    "Select model",
    ["Anthropic", "OpenAI"]
)

# Input task
task = st.text_area("Input task")

# Generate prompt
if st.button("Generate prompt"):
    if model == "Anthropic":
        prompt = generate_prompt_by_anthropic(task)
    else:
        prompt = generate_prompt_by_openai(task)
    st.info("Generated prompt:")
    st.text(prompt)