from openai import OpenAI

def test_model():
    client = OpenAI(
        base_url="http://localhost:12434/engines/llama.cpp/v1/",
        api_key="docker",
    )

    try:
        response = client.chat.completions.create(
            model="ai/gpt-oss:latest",
            messages=[
                {"role": "user", "content": "Say OK if you can read this message"}
            ],
        )

        print("Model response:")
        print(response.choices[0].message.content)

    except Exception as e:
        print("Error while calling the model:")
        print(e)


if __name__ == "__main__":
    test_model()
