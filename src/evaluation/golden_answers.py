import asyncio
from typing import Tuple, List, Dict, Optional

from aiohttp import ClientSession

from core.logger import exception, info
from core.repositories.repo_files import FileItem
from evaluation.questions import EvalQuestionCombined
from evaluation.subchat import call_chat_completions_non_streaming
from processing.p_models import ParagraphData
from openai_wrappers.types import ChatMessage, ChatMessageUser, ChatMessageSystem


__all__= ["produce_golden_answers"]


async def golden_answers_worker(
        http_session: ClientSession,
        messages: List[ChatMessage],
        question_id: int,
) -> Optional[Tuple[int, str]]:
    try:
        resp = await call_chat_completions_non_streaming(
            http_session,
            messages
        )
        answer: str = resp["choices"][0]["message"]["content"]
        return question_id, answer

    except Exception as e:
        exception(f"Error while getting golden answers: {e}")
        return None


async def golden_answers_for_doc(
        http_session: ClientSession,
        doc_text: str,
        questions: List[EvalQuestionCombined]
) -> Dict[int, str]:
    semaphore = asyncio.Semaphore(5)

    async def golden_answers_with_semaphore(
            _messages: List[ChatMessage],
            question_id: int,
    ):
        async with semaphore:
            return await golden_answers_worker(http_session, _messages, question_id)

    init_messages = [
        ChatMessageSystem(
            role="system",
            content="You are a helpful assistant that is utmost concise, yet precise, in its responses."
                    "You use additional context provided by a user and construct clear concise answer using that context."
        ),
        ChatMessageUser(role="user", content=doc_text),
    ]

    tasks = []
    for question in questions:
        q_messages = [
            *init_messages,
            ChatMessageUser(role="user", content=question.question_text),
        ]
        tasks.append(asyncio.create_task(
            golden_answers_with_semaphore(q_messages, question.id)
        ))

    results: List[Optional[Tuple[int, str]]] = await asyncio.gather(*tasks)
    results: List[Tuple[int, str]] = [r for r in results if r is not None]

    golden_answers = {
        q_id: answer for q_id, answer in results
    }

    return golden_answers


def produce_golden_answers(
        loop: asyncio.AbstractEventLoop,
        http_session: ClientSession,
        file_paragraphs: List[Tuple[FileItem, List[ParagraphData]]],
        questions: List[EvalQuestionCombined]
) -> Dict[str, Dict[int, str]]:
    results = {}

    for file, paragraphs in file_paragraphs:
        info(f"GOLDEN FOR FILE: {file.file_name_orig}")
        assert len(paragraphs)
        assert sum([(len(p.paragraph_text) or 1) / 4. for p in paragraphs]) <= 60_000, f"file {file.file_name} is too big: >60k tok"

        doc_text = "\n".join([p.paragraph_text for p in paragraphs])

        max_iters = 5
        iters = 0
        not_answered = questions.copy()
        answers_for_doc: Dict[int, str] = {}

        while True:
            info(f"{iters=}; TASKS: {len(not_answered)}")

            if iters == max_iters:
                raise Exception(f"Failed to produce golden answers: too many tries")

            answers_for_doc_iter: Dict[int, str] = loop.run_until_complete(
                golden_answers_for_doc(http_session, doc_text, questions)
            )
            answers_for_doc.update(answers_for_doc_iter)

            not_answered = [q for q in not_answered if q.id not in answers_for_doc.keys()]
            if not not_answered:
                break

            iters += 1

        results[file.file_name_orig] = answers_for_doc

    return results
