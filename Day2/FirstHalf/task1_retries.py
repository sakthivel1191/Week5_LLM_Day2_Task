import time
from dotenv import load_dotenv
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception,
    before_sleep_log
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)

class APIError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"API Error: {status_code}")

def should_retry(exception):
    if isinstance(exception, APIError):
        return (
            exception.status_code == 429
            or 500 <= exception.status_code <= 599
        )
    return False

def log_retry(retry_state):
    exception = retry_state.outcome.exception()
    attempt = retry_state.attempt_number
    wait_time = retry_state.next_action.sleep
    print(
        f"Attempt {attempt} failed "
        f"({exception.status_code}) "
        f"-> wait {wait_time:.0f} s"
    )

@retry(
    wait=wait_exponential(min=1, max=20),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(should_retry),
    before_sleep=log_retry,
    reraise=True
)
def call_llm():
    print("Calling LLM API...")
    return mock_api_call()
attempt_count = 0
def mock_api_call():
    global attempt_count
    attempt_count += 1
    if attempt_count <= 2:
        raise APIError(429)
    return "Hello! LLM response successful."
if __name__ == "__main__":
    try:
        result = call_llm()
        print(f"Attempt {attempt_count} succeeded")
        print(result)
    except Exception as error:
        print("Final error:", error)
