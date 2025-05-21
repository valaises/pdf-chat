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

        self.add_api_route("v1/experiments", self._experiments, methods=["GET"])

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

        experiments.sort(key=lambda x: -x["id"])

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
                    color: #353740;
                    background-color: #f7f7f8;
                    margin: 0;
                    padding: 20px;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                    padding: 20px;
                }
                h1 {
                    color: #10a37f;
                    text-align: center;
                    margin-bottom: 30px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                th {
                    background-color: #f0f0f0;
                    padding: 12px;
                    text-align: left;
                    font-weight: 600;
                    border-bottom: 2px solid #ddd;
                }
                td {
                    padding: 12px;
                    border-bottom: 1px solid #eee;
                    vertical-align: top;
                }
                tr:hover {
                    background-color: #f9f9f9;
                }
                .dataset-pill {
                    display: inline-block;
                    background-color: #e9f7f2;
                    color: #10a37f;
                    padding: 4px 10px;
                    border-radius: 16px;
                    font-size: 14px;
                    font-weight: 500;
                }
                .experiment-id {
                    font-weight: 600;
                    color: #444;
                }
                footer {
                    text-align: center;
                    margin-top: 30px;
                    color: #888;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Experiments</h1>
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Dataset</th>
                            <th>Description</th>
                            <th>Models</th>
                            <th>Strategy</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for exp in experiments:
            params = exp["params"]
            html_content += f"""
                        <tr>
                            <td class="experiment-id">{exp["id"]}</td>
                            <td><span class="dataset-pill">{params.dataset_name}</span></td>
                            <td>{params.description}</td>
                            <td>
                                <div>Chat: {params.chat_model}</div>
                                <div>Eval: {params.chat_eval_model}</div>
                            </td>
                            <td>
                                <div>Processing: {params.processing_strategy}</div>
                                <div>Save: {params.save_strategy}</div>
                            </td>
                        </tr>
            """

        html_content += """
                    </tbody>
                </table>
                <footer>Made with ❤️ at COXIT</footer>
            </div>
        </body>
        </html>
        """

        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html_content)
