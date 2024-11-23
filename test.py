import requests
import time


def test_rag_api():
    base_url = "http://localhost:8999"

    # Test health check first
    try:
        health_response = requests.get(f"{base_url}/")
        print("Health Check:", health_response.json())
    except Exception as e:
        print("Health check failed:", str(e))
        return

    # Test questions
    test_questions = [
        "what does chatbot in healthcare do?",
        "how do medical chatbots help patients?",
        "what are the benefits of healthcare chatbots?"
    ]

    for question in test_questions:
        try:
            print("\nSending question:", question)
            response = requests.post(
                f"{base_url}/ask",
                json={"question": question},
                timeout=30  # 30 seconds timeout
            )

            if response.status_code == 200:
                print("Answer:", response.json()['answer'])
            else:
                print(f"Error {response.status_code}:", response.json())

        except requests.exceptions.Timeout:
            print("Request timed out")
        except Exception as e:
            print("Error:", str(e))

        time.sleep(2)  # Wait 2 seconds between questions


if __name__ == "__main__":
    test_rag_api()