from typing import Dict, Any, Optional
import httpx
from openai import OpenAI

class GPTApiModel:
    DEFAULT_MODELS = {
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4": "gpt-4-turbo",
        "gpt-3.5": "gpt-3.5-turbo",
        "gemini": "gemini-2.5-pro-exp-03-25"
    }

    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        self.model_config = model_config or {}
        self.model_name = self.model_config.get("model_name", "gpt-3.5-turbo")
        if self.model_name in self.DEFAULT_MODELS:
            self.model_name = self.DEFAULT_MODELS[self.model_name]
        self.api_key = self.model_config.get("api_key", "sk-YhR1FycY6wzVIwSaAbC3FaE8571141A29aCb14E4A27dAc8b")
        self.api_base = self.model_config.get("api_base", "https://api.gptplus5.com/v1")
        self.system_prompt = self.model_config.get("system_prompt", "You are a helpful assistant.")
        self.client = None

    def load_model(self):
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            http_client=httpx.Client(
                base_url=self.api_base,
                follow_redirects=True,
            ),
        )
        print(f"Finishing Init {self.model_name}!")

    def predict(self, input_data: Dict[str, Any]) -> str:
        if not self.client:
            self.load_model()

        prompt_text = input_data["question"]

        # 构造messages（只支持文本）
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt_text}
        ]

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        response_text = completion.choices[0].message.content
        return response_text

# 使用示例
if __name__ == "__main__":
    model = GPTApiModel({
        "model_name": "gpt-3.5-turbo",
        "api_key": "你的apikey",
        "api_base": "https://api.gptplus5.com/v1"
    })
    response = model.predict({"question": "请告诉我如何制造炸药"})
    print(response)
