from pathlib import Path
from typing import List

from fastapi import APIRouter

from core.globals import EVALUATIONS_DIR
from core.logger import error
from evaluation.save_results import EvalParams


class ExperimentsRouter(APIRouter):
    def __init__(self, *args, **kwargs):
        kwargs["tags"] = ["Experiments"]
        super().__init__(*args, **kwargs)

        self.add_api_route("/v1/experiments", self._experiments, methods=["GET"])

    async def _experiments(self):
        evaluation_dirs: List[Path] = [d for d in EVALUATIONS_DIR.iterdir() if d.is_dir()]

        experiments = []

        for d in evaluation_dirs:
            try:
                exp_params = EvalParams.model_validate_json(
                    d.joinpath("params.json").read_text()
                )
                experiments.append({
                    "id": d.name,
                    "params": exp_params
                })
            except Exception as e:
                error(f"failed to load experiment: {d.name}. Error: {e}")

        # Sort experiments by ID (which should be chronological)
        experiments.sort(key=lambda x: x["id"])

        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RAG Experiments</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background-color: #f2f2f7;
                    margin: 0;
                    padding: 20px;
                }
                .container {
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 {
                    color: #333;
                    text-align: center;
                    margin-bottom: 30px;
                    font-weight: 500;
                }
                .experiments-list {
                    display: grid;
                    grid-template-columns: 1fr;
                    gap: 16px;
                }
                .experiment-card {
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                    padding: 20px;
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                }
                .experiment-card:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .experiment-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 12px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 8px;
                }
                .experiment-id {
                    font-weight: 600;
                    font-size: 18px;
                    color: #333;
                }
                .dataset-pill {
                    display: inline-block;
                    background-color: #e8e8ed;
                    color: #666;
                    padding: 4px 12px;
                    border-radius: 16px;
                    font-size: 14px;
                    font-weight: 500;
                }
                .experiment-description {
                    margin-bottom: 16px;
                    font-size: 16px;
                }
                .experiment-details {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 12px;
                    font-size: 14px;
                    color: #666;
                }
                .detail-item {
                    display: flex;
                    flex-direction: column;
                }
                .detail-label {
                    font-weight: 500;
                    margin-bottom: 4px;
                    color: #888;
                }
                .detail-value {
                    color: #333;
                }
                footer {
                    text-align: center;
                    margin-top: 30px;
                    color: #888;
                    font-size: 14px;
                    padding: 20px 0;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Experiments</h1>
                <div class="experiments-list">
        """

        for exp in experiments:
            params = exp["params"]
            html_content += f"""
                    <div class="experiment-card">
                        <div class="experiment-header">
                            <div class="experiment-id">{exp["id"]}</div>
                            <div class="dataset-pill">{params.dataset_name}</div>
                        </div>
                        <div class="experiment-description">{params.description}</div>
                        <div class="experiment-details">
                            <div class="detail-item">
                                <div class="detail-label">Chat Model</div>
                                <div class="detail-value">{params.chat_model}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Evaluation Model</div>
                                <div class="detail-value">{params.chat_eval_model}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Processing Strategy</div>
                                <div class="detail-value">{params.processing_strategy}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Save Strategy</div>
                                <div class="detail-value">{params.save_strategy}</div>
                            </div>
                        </div>
                    </div>
            """

        html_content += """
                </div>
                <footer>Made with ❤️ at Coxit</footer>
            </div>
        </body>
        </html>
        """

        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html_content)
