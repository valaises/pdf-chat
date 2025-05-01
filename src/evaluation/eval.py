import asyncio
from typing import List, Dict, Optional, Tuple

from aiohttp import ClientSession
from pydantic import BaseModel

from core.logger import exception, info
from evaluation.eval_utils import parse_model_output_json
from evaluation.questions import EvalQuestionCombined
from evaluation.subchat import call_chat_completions_non_streaming
from openai_wrappers.types import ChatMessage, ChatMessageUser


__all__ = ["evaluate_model_outputs", "EvaluationResult", "Question"]


class Question(BaseModel):
    question: str
    answer: str
    is_question_mentioned: bool
    is_question_answered: bool


class EvaluationResult(BaseModel):
    questions: List[Question]


async def evaluate_answer_worker(
        http_session: ClientSession,
        messages: List[ChatMessage],
        question_id: int,
) -> Optional[Tuple[int, EvaluationResult]]:
    try:
        resp = await call_chat_completions_non_streaming(http_session, messages)
        answer: str = resp["choices"][0]["message"]["content"]
        e_res: EvaluationResult = parse_model_output_json(answer, EvaluationResult)
        return question_id, e_res

    except Exception as e:
        exception(f"Error evaluating answer: {e}")
        return None


async def evaluate_answers_for_doc(
        http_session: ClientSession,
        questions: List[EvalQuestionCombined],
        doc_answers: Dict[int, str],
) -> Dict[int, EvaluationResult]:
    semaphore = asyncio.Semaphore(5)

    async def evaluate_answer_with_semaphore(
            _messages: List[ChatMessage],
            question_id: int,
    ):
        async with semaphore:
            return await evaluate_answer_worker(http_session, _messages, question_id)

    tasks = []
    for question in questions:
        answer = doc_answers[question.id]

        prompt = PROMPT.replace(
            "%questions%", "\n".join(question.questions_split)
        ).replace(
            "%answer%", answer
        )

        messages = [
            ChatMessageUser(role="user", content=prompt)
        ]
        tasks.append(asyncio.create_task(
            evaluate_answer_with_semaphore(messages, question.id)
        ))

    results: List[Optional[Tuple[int, EvaluationResult]]] = await asyncio.gather(*tasks)
    results: List[Tuple[int, EvaluationResult]] = [r for r in results if r is not None]

    results: Dict[int, EvaluationResult] = {r[0]: r[1] for r in results}

    return results


def evaluate_model_outputs(
        loop: asyncio.AbstractEventLoop,
        http_session: ClientSession,
        questions: List[EvalQuestionCombined],
        answers: Dict[str, Dict[int, str]]
):
    results = {}

    for doc_name, doc_answers in answers.items():
        info(f"Evaluating results for {doc_name}")
        assert len(doc_answers)

        max_iters = 5
        iters = 0
        not_answered = doc_answers.copy()
        eval_results_for_doc: Dict[int, EvaluationResult] = {}

        while True:
            info(f"{iters=}; TASKS: {len(not_answered)}")

            if iters >= max_iters:
                raise Exception("Failed to eval: too many tries")

            eval_results_for_doc_iter: Dict[int, EvaluationResult] = loop.run_until_complete(
                evaluate_answers_for_doc(http_session, questions, not_answered)
            )
            eval_results_for_doc.update(eval_results_for_doc_iter)

            not_answered = {
                k: v for k, v in not_answered.items()
                if k not in eval_results_for_doc.keys()
            }
            if not not_answered:
                break

            iters += 1

        results[doc_name] = eval_results_for_doc

    return results

# todo: add case: question cannot be answered as there is no information in text
PROMPT = """
QUESTIONS: 
%questions%


ANSWER: 
%answer%


You are given a list of one or several questions.
You have to evaluate the answer.
Question text could be complex and contain several questions.

Detect Questions (one or several) in the question text.
For each question in Questions:
    Check if the question is mentioned AND
    Check if the question is answered

CRITERIA:
is_question_mentioned -- answer text contains mentions of the question. Not necessarily gives an answer.
is_question_answered -- question has a reasonable answer, supplemented with details.  

Example:
QUESTION: Is Stevens listed as an Approved Manufacturer or is a Substitution Request needed for this?
ANSWER: Yes, "Stevens Industries, Inc." is listed as an approved manufacturer in the document under the acceptable products for casework.

Example above gets is_answered=true, as it matches criteria.

Output format:
```json
{
    "questions": [
        {
            "question": "Is Stevens an Approved Manufacturer?",
            "answer": "Yes",
            "is_question_mentioned": boolean,
            "is_question_answered": boolean
        }
        ...
    ]
}
```

WHERE::
question: very compact question text
answer: very compact summary of the actual answer

Provide output in a valid machine-readable JSON format.
"""
