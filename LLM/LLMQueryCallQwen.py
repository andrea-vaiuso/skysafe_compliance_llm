import json
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class QueryGenerator:
    """
    Wrapper around a local Qwen model that turns text chunks into
    concise queries paired with reasoning levels.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 2048,
        max_queries: int = 8,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # bitsandbytes quantization config, 4 bit
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        # device_map already places the model, keep device only for reference if needed
        if device is None:
            self.device = self.model.device
        else:
            self.device = torch.device(device)
            self.model.to(self.device)

        self.max_new_tokens = max_new_tokens
        self.max_queries = max_queries

    def _system_message(self) -> str:
        return (
            "You generate queries from a provided query.\n\n"
            "First decide if the query can be answered as a single focused question.\n"
            "If yes, return exactly one query.\n"
            "Split into multiple queries only if the query clearly contains independent questions.\n"
            "Avoid semantically similar queries and use the minimum number needed to cover the intent.\n\n"
            f"Return at most {self.max_queries} queries, ordered by usefulness.\n"
            "Each query object must include query as a string and reasoning_level as one of none, low, medium, high.\n\n"
            "Output a single valid JSON object with one key named queries containing the list of query objects.\n"
            "Do not include any text before or after the JSON."
        )

    def _build_prompt(self, query: str) -> str:
        if not query:
            raise ValueError(f"Query is empty. Query: {query}")

        prompt = (
            "<|im_start|>system\n"
            f"{self._system_message()}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "Generate targeted queries that help understand, verify, or expand on the following query.\n\n"
            "Query text:\n"
            f"\"\"\"{query}\"\"\"\n\n"
            "Return only the JSON object described above.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return prompt

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        generated_ids = output_ids[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        s = text.strip()

        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            return parsed

        if isinstance(parsed, list):
            return {"queries": parsed}

        first_obj = s.find("{")
        last_obj = s.rfind("}")
        first_arr = s.find("[")
        last_arr = s.rfind("]")

        if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
            arr_str = s[first_arr : last_arr + 1]
            try:
                return {"queries": json.loads(arr_str)}
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON array from model output: {arr_str!r}") from e

        if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
            obj_str = s[first_obj : last_obj + 1]
            try:
                return json.loads(obj_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON object from model output: {obj_str!r}") from e

        raise ValueError(f"Could not find JSON in model output: {text!r}")


    def generate_queries(self, query: str) -> Dict[str, Any]:
        prompt = self._build_prompt(query)
        raw_output = self._generate(prompt)
        data = self._extract_json(raw_output)

        queries = data.get("queries", [])
        if not isinstance(queries, list):
            raise ValueError(f"queries is not a list in model output: {data!r}")

        normalized_queries: List[Dict[str, str]] = []
        for entry in queries:
            if not isinstance(entry, dict):
                continue

            query_text = entry.get("query")
            level = entry.get("reasoning_level", "")

            if not isinstance(query_text, str):
                continue

            query_text = query_text.strip()
            if not query_text:
                continue

            level = level.strip() if isinstance(level, str) else ""
            normalized_queries.append(
                {
                    "query": query_text,
                    "reasoning_level": level,
                }
            )

            if len(normalized_queries) >= self.max_queries:
                break

        return {
            "queries": normalized_queries,
        }
