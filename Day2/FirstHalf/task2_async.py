import asyncio
import time
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("OpenRouter API key not found!")
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    max_retries=0,
    timeout=60.0
)
MODEL = "openai/gpt-4o-mini"
semaphore = asyncio.Semaphore(3)
async def ask(q):
    start_time = time.perf_counter()
    try:
        if "FAIL_TEST" in q:
            raise ValueError("Simulated API failure")
        async with semaphore:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": q
                    }
                ],
                max_tokens=100
            )
            answer = response.choices[0].message.content
            status = "Success"
    except Exception as error:
        answer = str(error)
        status = "Failed"
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    return {
        "question": q,
        "status": status,
        "time": elapsed,
        "answer": answer
    }
questions = [
    "What is Python?",
    "Explain machine learning.",
    "What is an LLM?",
    "Explain API in simple words.",
    "What is async programming?",
    "FAIL_TEST - simulate a failed request",
    "Explain prompt engineering.",
    "What is a neural network?",
    "Explain generative AI.",
    "What is a REST API?"
]
async def run_sequential():
    print("\nRunning Sequential Calls...\n")
    start = time.perf_counter()
    results = []
    for question in questions:
        result = await ask(question)
        results.append(result)
    total_time = time.perf_counter() - start
    return results, total_time
async def run_parallel():
    print("\nRunning Parallel Calls...\n")
    start = time.perf_counter()
    tasks = [
        ask(question)
        for question in questions
    ]
    results = await asyncio.gather(
        *tasks,
        return_exceptions=True
    )
    total_time = time.perf_counter() - start
    final_results = []
    for question, result in zip(questions, results):
        if isinstance(result, Exception):
            final_results.append({
                "question": question,
                "status": "Failed",
                "time": 0,
                "answer": str(result)
            })
        else:
            final_results.append(result)
    return final_results, total_time
def print_table(results):
    print("\nQUESTION | STATUS | TIME")
    for result in results:
        print(
            f"{result['question'][:30]:30} | "
            f"{result['status']:8} | "
            f"{result['time']:.2f} sec"
        )
async def main():
    try:
        seq_results, seq_time = await run_sequential()
        print_table(seq_results)
        par_results, par_time = await run_parallel()
        print_table(par_results)
        print("\nTIME COMPARISON")
        print(f"Sequential: {seq_time:.2f} seconds")
        print(f"Parallel:   {par_time:.2f} seconds")
        if par_time > 0:
            speedup = seq_time / par_time
            print(f"Speedup:    {speedup:.2f}x faster")
    finally:
        await client.close()
if __name__ == "__main__":
    asyncio.run(main())
