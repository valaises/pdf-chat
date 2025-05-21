import os
import datetime

import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from core.globals import EVALUATIONS_DIR
from core.logger import error
from evaluation.save_results import EvalParams


class ExperimentsRouter(APIRouter):
    def __init__(self, *args, **kwargs):
        kwargs["tags"] = ["Experiments"]
        super().__init__(*args, **kwargs)

        self.add_api_route("/v1/experiments", self._experiments, methods=["GET"])
        self.add_api_route("/v1/experiments/{experiment_id}", self._experiment_detail, methods=["GET"])

    async def _experiments(self):
        evaluation_dirs: List[Path] = [d for d in EVALUATIONS_DIR.iterdir() if d.is_dir()]

        experiments = []

        for d in evaluation_dirs:
            try:
                exp_params = EvalParams.model_validate_json(
                    d.joinpath("params.json").read_text()
                )
                completed = d.joinpath("analysis_overall.md").exists()
                metrics = {}
                if completed:
                    metrics_file = d / "stage3_evaluation" / "metrics" / "comprehensive_answer.json"
                    metrics_data = json.loads(metrics_file.read_text())["overall"]
                    metrics = {
                        "accuracy": metrics_data["accuracy"]["value"]
                    }

                # Get directory creation time
                creation_time = os.path.getctime(d)
                creation_date = datetime.datetime.fromtimestamp(creation_time)
                formatted_date = creation_date.strftime("%Y-%m-%d %H:%M")

                experiments.append({
                    "id": d.name,
                    "params": exp_params,
                    "completed": completed,
                    "metrics": metrics,
                    "created_at": formatted_date
                })
            except Exception as e:
                error(f"failed to load experiment: {d.name}. Error: {e}")

        # Sort experiments by ID (which should be chronological)
        experiments.sort(key=lambda x: x["id"], reverse=True)

        # Get unique datasets and assign colors
        unique_datasets = set(exp["params"].dataset_name for exp in experiments)

        # Function to generate a color based on dataset name
        def get_dataset_color(dataset_name):
            # Predefined colors - pastel and visually distinct
            colors = [
                "#FFD6A5", "#CAFFBF", "#9BF6FF", "#BDB2FF", "#FFC6FF",  # pastel
                "#FDFFB6", "#A0C4FF", "#FFB5A7", "#D0F4DE", "#E4C1F9",  # more pastel
                "#F8EDEB", "#F9C74F", "#90BE6D", "#43AA8B", "#577590",  # muted
                "#F94144", "#F3722C", "#F8961E", "#F9C74F", "#90BE6D",  # warm to cool
            ]

            # Simple hash function to pick a color
            hash_value = sum(ord(c) for c in dataset_name)
            return colors[hash_value % len(colors)]

        # Create a mapping of dataset names to colors
        dataset_colors = {dataset: get_dataset_color(dataset) for dataset in unique_datasets}

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
                    display: flex;
                    flex-direction: column;
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
                    display: flex;
                    align-items: center;
                }
                .status-indicator {
                    margin-left: 8px;
                    font-size: 16px;
                }
                .dataset-pill {
                    display: inline-block;
                    padding: 2px 8px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: 500;
                    color: rgba(0, 0, 0, 0.7);
                }
                .experiment-description {
                    margin-bottom: 8px;
                    font-size: 14px;
                    color: #555;
                }
                .experiment-date {
                    font-size: 12px;
                    color: #888;
                    margin-bottom: 8px;
                }
                .experiment-footer {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-end;
                    margin-top: auto;
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
                .metrics-container {
                    display: flex;
                    flex-direction: column;
                    align-items: flex-end;
                    gap: 4px;
                }
                .metric-item {
                    background-color: #f0f7ff;
                    padding: 3px 8px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: 500;
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
            dataset_name = params.dataset_name
            dataset_color = dataset_colors[dataset_name]

            # Status indicator
            status_emoji = "✅" if exp["completed"] else "❌"
            status_title = "Completed" if exp["completed"] else "In Progress"

            # Metrics display
            metrics_html = ""
            if exp["completed"] and "accuracy" in exp["metrics"]:
                accuracy = exp["metrics"]["accuracy"]
                accuracy_formatted = f"{accuracy:.1%}" if isinstance(accuracy, float) else accuracy
                metrics_html = f"""
                <div class="metrics-container">
                    <div class="metric-item">Accuracy: {accuracy_formatted}</div>
                </div>
                """

            html_content += f"""
                    <a href="/v1/experiments/{exp["id"]}" style="text-decoration: none; color: inherit;">
                        <div class="experiment-card">
                            <div class="experiment-header">
                                <div class="experiment-id">
                                    {exp["id"]}
                                    <span class="status-indicator" title="{status_title}">{status_emoji}</span>
                                </div>
                                <div class="dataset-pill" style="background-color: {dataset_color};">{dataset_name}</div>
                            </div>
                            <div class="experiment-date">Created: {exp["created_at"]}</div>
                            <div class="experiment-description">{params.description}</div>
                            <div class="experiment-footer">
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
                                {metrics_html}
                            </div>
                        </div>
                    </a>
            """

        html_content += """
                </div>
                <footer>Made with ❤️ at Coxit</footer>
            </div>
        </body>
        </html>
        """

        return HTMLResponse(content=html_content)

    async def _experiment_detail(self, experiment_id: str):
        experiment_dir = EVALUATIONS_DIR / experiment_id

        if not experiment_dir.exists() or not experiment_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

        try:
            exp_params = EvalParams.model_validate_json(
                experiment_dir.joinpath("params.json").read_text()
            )
        except Exception as e:
            error(f"Failed to load experiment: {experiment_id}. Error: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading experiment data: {str(e)}")

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Experiment {experiment_id}</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    line-height: 1.4;
                    color: #333;
                    background-color: #f2f2f7;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                    padding: 20px;
                }}
                h1 {{
                    color: #333;
                    margin-bottom: 20px;
                }}
                .back-button {{
                    display: inline-block;
                    margin-top: 20px;
                    padding: 8px 16px;
                    background-color: #f2f2f7;
                    color: #333;
                    text-decoration: none;
                    border-radius: 6px;
                    font-weight: 500;
                    border: 1px solid #ddd;
                }}
                .back-button:hover {{
                    background-color: #e5e5ea;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Experiment {experiment_id}</h1>
                <p>Details for this experiment will be available soon.</p>
                <a href="/v1/experiments" class="back-button">← Back to Experiments</a>
            </div>
        </body>
        </html>
        """

        return HTMLResponse(content=html_content)
