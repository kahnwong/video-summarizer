import json
import os

from jinja2 import Template
from loguru import logger as log
from openai import OpenAI

from config import summarize


def read_transcript() -> str:
    log.info(f"Reading transcript from {summarize.TRANSCRIPT_FILE}")
    dialogue_lines = []
    with open(summarize.TRANSCRIPT_FILE, "r") as f:
        for line in f:
            t = json.loads(line.strip())
            dialogue_lines.append(f"[{t['start']:.2f}s] {t['speaker']}: {t['word']}")

    dialogue = "\n".join(dialogue_lines)
    log.info(f"Loaded {len(dialogue_lines)} dialogue lines")

    return dialogue


def load_template() -> Template:
    log.info(f"Loading template from {summarize.TEMPLATE_FILE}")
    with open(summarize.TEMPLATE_FILE, "r") as f:
        template_content = f.read()
    return Template(template_content)


def generate_summary(dialogue: str) -> str:
    # Load prompt template
    prompt_template = load_template()
    prompt = prompt_template.render(max_images=summarize.MAX_IMAGES, dialogue=dialogue)

    # Initialize OpenAI client with custom endpoint
    log.info(
        f"Initializing OpenAI client with base URL: {os.getenv('OPENAI_BASE_URL')}"
    )
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
    )

    # Generate with streaming
    log.info("Generating summary...")
    stream = client.chat.completions.create(
        model=summarize.MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        temperature=0.7,
        max_tokens=2048,
    )

    # Stream output to stdout and collect response
    response_text = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            response_text += content

    print()  # New line after streaming

    # Strip markdown code blocks if present
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]  # Remove ```json
    elif response_text.startswith("```"):
        response_text = response_text[3:]  # Remove ```

    if response_text.endswith("```"):
        response_text = response_text[:-3]  # Remove trailing ```

    response_text = response_text.strip()

    return response_text


def save_summary(summary: str):
    log.info(f"Writing summary to {summarize.OUTPUT_FILE}")
    with open(summarize.OUTPUT_FILE, "w") as f:
        f.write(summary + "\n")
    log.info("Summary generation complete")


if __name__ == "__main__":
    dialogue = read_transcript()
    summary = generate_summary(dialogue)
    save_summary(summary)
