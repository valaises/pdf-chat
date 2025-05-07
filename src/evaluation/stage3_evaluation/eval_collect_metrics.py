from dataclasses import dataclass, field
from typing import Dict, List

from pydantic import BaseModel

from evaluation.stage3_evaluation.llm_judge import EvaluationResult, QuestionEval
from evaluation.stage3_evaluation.metrics import BooleanFields, calculate_binary_metrics


__all__ = ["collect_eval_metrics", "EvalResultsMetrics"]


@dataclass
class ResultsBinFields:
    is_question_answered: List[bool] = field(default_factory=list)
    requires_additional_information: List[bool] = field(default_factory=list)
    is_speculative: List[bool] = field(default_factory=list)
    is_confident: List[bool] = field(default_factory=list)
    # is_question_answered=true requires_additional_information=false and is_speculative=false and is_confident=true
    comprehensive_answer: List[bool] = field(default_factory=list)


@dataclass
class ResultsBinFieldsTuple:
    golden: ResultsBinFields = field(default_factory=ResultsBinFields)
    pred: ResultsBinFields = field(default_factory=ResultsBinFields)


@dataclass
class EvalBinResults:
    per_question: Dict[str, ResultsBinFieldsTuple] = field(default_factory=dict)
    per_file: Dict[str, ResultsBinFieldsTuple] = field(default_factory=dict)
    overall: ResultsBinFieldsTuple = field(default_factory=ResultsBinFieldsTuple)



class EvalResultsMetrics(BaseModel):
    per_question: Dict[str, BooleanFields]
    per_file: Dict[str, BooleanFields]
    overall: BooleanFields


def calculate_boolean_fields(bin_res_tpl: ResultsBinFieldsTuple) -> BooleanFields:
    return BooleanFields(
        is_question_answered=calculate_binary_metrics(
            bin_res_tpl.golden.is_question_answered, bin_res_tpl.pred.is_question_answered
        ),
        requires_additional_information=calculate_binary_metrics(
            bin_res_tpl.golden.requires_additional_information, bin_res_tpl.pred.requires_additional_information,
        ),
        is_speculative=calculate_binary_metrics(
            bin_res_tpl.golden.is_speculative, bin_res_tpl.pred.is_speculative
        ),
        is_confident=calculate_binary_metrics(
            bin_res_tpl.golden.is_confident, bin_res_tpl.pred.is_confident
        ),
        comprehensive_answer=calculate_binary_metrics(
            bin_res_tpl.golden.comprehensive_answer, bin_res_tpl.pred.comprehensive_answer
        )
    )


def is_comprehensive_answer(result: QuestionEval) -> bool:
    """
    Determines if an answer is comprehensive based on multiple criteria.

    A comprehensive answer must:
    - Answer the question
    - Not require additional information
    - Not be speculative
    - Be confident
    """
    return (
            result.is_question_answered and
            not result.requires_additional_information and
            not result.is_speculative and
            result.is_confident
    )


def collect_eval_metrics(
        a_golden: Dict[str, Dict[int, EvaluationResult]],
        a_pred: Dict[str, Dict[int, EvaluationResult]],
) -> EvalResultsMetrics:
    assert len(a_golden) == len(a_pred)
    assert set(a_golden.keys()) == set(a_pred.keys())

    eval_bin_fields = [
        "is_question_answered",
        "requires_additional_information",
        "is_speculative",
        "is_confident",
    ]

    results = EvalBinResults()

    # Process all questions across all files
    for filename in a_golden:
        ans_golden_fn = a_golden[filename]
        ans_pred_fn = a_pred[filename]

        assert set(ans_golden_fn.keys()) == set(ans_pred_fn.keys())

        # Initialize file results container if needed
        if filename not in results.per_file:
            results.per_file[filename] = ResultsBinFieldsTuple()

        # Process each question in the file
        for q_id in ans_golden_fn:
            ans_golden_q = ans_golden_fn[q_id]
            ans_pred_q = ans_pred_fn[q_id]

            # Ensure question IDs match between golden and pred
            golden_q_ids = {q.id for q in ans_golden_q.questions}
            pred_q_ids = {q.id for q in ans_pred_q.questions}
            assert golden_q_ids == pred_q_ids

            # Process each sub-question
            for q_split_id in golden_q_ids:
                key = f"{q_id}_{q_split_id}"

                # Initialize question results container if needed
                if key not in results.per_question:
                    results.per_question[key] = ResultsBinFieldsTuple()

                # Get the specific question evaluations
                q_split_golden_val = next(q for q in ans_golden_q.questions if q.id == q_split_id)
                q_split_pred_val = next(q for q in ans_pred_q.questions if q.id == q_split_id)

                # Calculate comprehensive answer flags
                golden_comprehensive = is_comprehensive_answer(q_split_golden_val)
                pred_comprehensive = is_comprehensive_answer(q_split_pred_val)

                # Update all metrics for this question
                for f in eval_bin_fields:
                    f_golden_val = getattr(q_split_golden_val, f)
                    f_pred_val = getattr(q_split_pred_val, f)

                    # Update per-question metrics
                    getattr(results.per_question[key].golden, f).append(f_golden_val)
                    getattr(results.per_question[key].pred, f).append(f_pred_val)

                    # Update per-file metrics
                    getattr(results.per_file[filename].golden, f).append(f_golden_val)
                    getattr(results.per_file[filename].pred, f).append(f_pred_val)

                # Update comprehensive answer metrics
                results.per_question[key].golden.comprehensive_answer.append(golden_comprehensive)
                results.per_question[key].pred.comprehensive_answer.append(pred_comprehensive)
                results.per_file[filename].golden.comprehensive_answer.append(golden_comprehensive)
                results.per_file[filename].pred.comprehensive_answer.append(pred_comprehensive)

    # Aggregate overall results
    for file_results in results.per_file.values():
        for f in eval_bin_fields + ["comprehensive_answer"]:
            getattr(results.overall.golden, f).extend(getattr(file_results.golden, f))
            getattr(results.overall.pred, f).extend(getattr(file_results.pred, f))

    # Calculate metrics
    metrics = EvalResultsMetrics(
        per_question={k: calculate_boolean_fields(v) for k, v in results.per_question.items()},
        per_file={k: calculate_boolean_fields(v) for k, v in results.per_file.items()},
        overall=calculate_boolean_fields(results.overall),
    )

    return metrics
