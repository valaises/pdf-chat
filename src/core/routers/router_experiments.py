import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from core.globals import EVALUATIONS_DIR, ASSETS_CSS
from core.logger import error
from evaluation.metering import Metering
from evaluation.save_results import EvalParams
import markdown


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
                completion_time = None

                import os
                import datetime

                # Get directory creation time
                dir_creation_time = os.path.getctime(d)
                creation_date = datetime.datetime.fromtimestamp(dir_creation_time)
                formatted_date = creation_date.strftime("%Y-%m-%d %H:%M")

                # Calculate completion time if experiment is completed
                if completed:
                    metrics_file = d / "stage3_evaluation" / "metrics" / "comprehensive_answer.json"
                    metrics_data = json.loads(metrics_file.read_text())["overall"]
                    metrics = {
                        "accuracy": metrics_data["accuracy"]["value"]
                    }

                    analysis_file = d.joinpath("analysis_overall.md")
                    analysis_creation_time = os.path.getctime(analysis_file)
                    completion_minutes = (analysis_creation_time - dir_creation_time) / 60

                    if completion_minutes < 60:
                        completion_time = f"{int(completion_minutes)} min"
                    else:
                        completion_hours = completion_minutes / 60
                        if completion_hours < 24:
                            completion_time = f"{completion_hours:.1f} hours"
                        else:
                            completion_days = completion_hours / 24
                            completion_time = f"{completion_days:.1f} days"

                experiments.append({
                    "id": d.name,
                    "params": exp_params,
                    "completed": completed,
                    "metrics": metrics,
                    "created_at": formatted_date,
                    "completion_time": completion_time
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
                %EXPERIMENTS.CSS%
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

            # Completion time display
            completion_html = ""
            if exp["completed"] and exp["completion_time"]:
                completion_html = f"""
                <div class="completion-time">
                    <span class="completion-emoji">⏱️</span>
                    <span>Completed in {exp["completion_time"]}</span>
                </div>
                """

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
                            <div class="experiment-meta">
                                <div class="experiment-date">Created: {exp["created_at"]}</div>
                                {completion_html}
                            </div>
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

        html_content = html_content.replace("%EXPERIMENTS.CSS%", ASSETS_CSS.joinpath("experiments.css").read_text())

        return HTMLResponse(content=html_content)

    async def _experiment_detail(self, experiment_id: str):
        experiment_dir = EVALUATIONS_DIR / experiment_id

        if not experiment_dir.exists() or not experiment_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

        try:
            # Load experiment data
            exp_params = EvalParams.model_validate_json(
                experiment_dir.joinpath("params.json").read_text()
            )

            # Check if analysis file exists
            analysis_file = experiment_dir.joinpath("analysis_overall.md")
            if analysis_file.exists():
                anal_overall = analysis_file.read_text()
                # Convert markdown to HTML
                anal_overall_html = markdown.markdown(anal_overall, extensions=['tables', 'fenced_code'])
            else:
                anal_overall_html = "<p>Analysis not available yet.</p>"

            # Load metering data if available
            metering_file = experiment_dir.joinpath("metering.json")
            if metering_file.exists():
                metering = Metering.model_validate_json(metering_file.read_text())
            else:
                metering = None

            # Generate HTML for experiment parameters table
            params_html = """
            <div class="section">
                <div class="section-title">Experiment Parameters</div>
                <table class="info-table">
                    <tr>
                        <th>Dataset</th>
                        <td>{dataset_name}</td>
                    </tr>
                    <tr>
                        <th>Description</th>
                        <td>{description}</td>
                    </tr>
                    <tr>
                        <th>Chat Model</th>
                        <td>{chat_model}</td>
                    </tr>
                    <tr>
                        <th>Evaluation Model</th>
                        <td>{chat_eval_model}</td>
                    </tr>
                    <tr>
                        <th>Processing Strategy</th>
                        <td>{processing_strategy}</td>
                    </tr>
                    <tr>
                        <th>Save Strategy</th>
                        <td>{save_strategy}</td>
                    </tr>
                </table>
            </div>
            """.format(
                dataset_name=exp_params.dataset_name,
                description=exp_params.description,
                chat_model=exp_params.chat_model,
                chat_eval_model=exp_params.chat_eval_model,
                processing_strategy=exp_params.processing_strategy,
                save_strategy=exp_params.save_strategy
            )

            # Generate HTML for documents list
            documents_html = """
            <div class="section">
                <div class="section-title">Evaluated Documents</div>
                <ul>
            """
            for doc in exp_params.eval_documents:
                documents_html += f"<li>{doc}</li>"
            documents_html += """
                </ul>
            </div>
            """

            # Generate HTML for metering data if available
            metering_html = ""
            if metering:
                metering_html = """
                <div class="section">
                    <div class="section-title">Metering Information</div>
                    <div class="metering-section">
                """

                # Function to generate a metering card for each stage
                def generate_metering_card(stage_name, stage_data):
                    card = f"""
                    <div class="metering-card">
                        <div class="metering-title">{stage_name}</div>
                        <table class="metering-table">
                            <tr>
                                <th>Model</th>
                                <th>Requests</th>
                                <th>Messages</th>
                                <th>Tokens In</th>
                                <th>Tokens Out</th>
                            </tr>
                    """

                    for model, data in stage_data.items():
                        card += f"""
                            <tr>
                                <td>{model}</td>
                                <td>{data.requests_cnt}</td>
                                <td>{data.messages_sent_cnt}</td>
                                <td>{data.tokens_in}</td>
                                <td>{data.tokens_out}</td>
                            </tr>
                        """

                    card += """
                        </table>
                    </div>
                    """
                    return card

                # Add cards for each metering stage
                if metering.dataset_compose:
                    metering_html += generate_metering_card("Dataset Compose", metering.dataset_compose)
                if metering.stage1:
                    metering_html += generate_metering_card("Stage 1", metering.stage1)
                if metering.stage2:
                    metering_html += generate_metering_card("Stage 2", metering.stage2)
                if metering.stage3:
                    metering_html += generate_metering_card("Stage 3", metering.stage3)
                if metering.stage4:
                    metering_html += generate_metering_card("Stage 4", metering.stage4)

                metering_html += """
                    </div>
                </div>
                """

            # Generate the full HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Experiment {experiment_id}</title>
                <style>
                    %EXPERIMENT-DETAIL.CSS%
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Experiment {experiment_id}</h1>

                    {params_html}

                    {documents_html}

                    {metering_html}

                    <div class="section">
                        <div class="section-title">Analysis Results</div>
                        <div class="markdown-content">
                            {anal_overall_html}
                        </div>
                    </div>

                    <a href="/v1/experiments" class="back-button">← Back to Experiments</a>
                </div>
            </body>
            </html>
            """

            html_content = html_content.replace("%EXPERIMENT-DETAIL.CSS%",
                                                ASSETS_CSS.joinpath("experiment-detail.css").read_text())

            return HTMLResponse(content=html_content)

        except Exception as e:
            error(f"Failed to load experiment: {experiment_id}. Error: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading experiment data: {str(e)}")
