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
                    line-height: 1.4;
                    color: #333;
                    background-color: #f2f2f7;
                    margin: 0;
                    padding: 12px;
                }
                .container {
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 10px;
                }
                h1 {
                    color: #333;
                    text-align: center;
                    margin-bottom: 20px;
                    font-weight: 500;
                    font-size: 24px;
                }
                .experiments-list {
                    display: grid;
                    grid-template-columns: 1fr;
                    gap: 10px;
                }
                .experiment-card {
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
                    padding: 12px;
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                }
                .experiment-card:hover {
                    transform: translateY(-1px);
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }
                .experiment-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 6px;
                }
                .experiment-id {
                    font-weight: 600;
                    font-size: 15px;
                    color: #333;
                }
                .dataset-pill {
                    display: inline-block;
                    background-color: #e8e8ed;
                    color: #666;
                    padding: 2px 8px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: 500;
                }
                .experiment-description {
                    margin-bottom: 8px;
                    font-size: 14px;
                    color: #555;
                }
                .experiment-details {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px;
                    font-size: 12px;
                    color: #666;
                }
                .detail-item {
                    background-color: #f5f5f7;
                    padding: 3px 8px;
                    border-radius: 4px;
                }
                .detail-label {
                    font-weight: 500;
                    color: #888;
                    margin-right: 4px;
                }
                footer {
                    text-align: center;
                    margin-top: 20px;
                    color: #888;
                    font-size: 12px;
                    padding: 10px 0;
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
                                <span class="detail-label">Chat:</span>
                                <span>{params.chat_model}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Eval:</span>
                                <span>{params.chat_eval_model}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Processing:</span>
                                <span>{params.processing_strategy}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Save:</span>
                                <span>{params.save_strategy}</span>
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