"""
GPT-4o via OpenRouter (vision). Send the artifact image + prompt -> L-F-V JSON.

Key in env OPENROUTER_API_KEY. Model 'openai/gpt-4o'. JSON is forced via
response_format. Low temperature for metric consistency.
"""
import base64
import json
import os

from openai import OpenAI


def get_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def _data_url(path):
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f"data:image/png;base64,{b}"


def explain(client, prompt, image_path, model="openai/gpt-4o", max_tokens=900):
    """Return (parsed_dict, raw_text). parsed_dict is None if JSON parsing fails."""
    r = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
            ],
        }],
        max_tokens=max_tokens,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    raw = r.choices[0].message.content
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        return None, raw
