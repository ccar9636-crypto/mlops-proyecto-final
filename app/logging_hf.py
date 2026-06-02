from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from huggingface_hub import hf_hub_download, upload_file


def write_prediction_log(app_env: str, request_payload: dict, prediction: dict) -> None:
    repo_id = os.getenv("HF_LOG_REPO_ID")
    token = os.getenv("HF_TOKEN")

    if not repo_id or not token:
        return

    log_path = f"logs/predicciones_{app_env}.txt"
    line = json.dumps(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "environment": app_env,
            "request": request_payload,
            "prediction": prediction,
        },
        ensure_ascii=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = Path(tmpdir) / f"predicciones_{app_env}.txt"

        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=log_path,
                token=token,
            )
            local_file.write_text(Path(downloaded).read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            local_file.write_text("", encoding="utf-8")

        with local_file.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

        upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=log_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Add {app_env} prediction log",
        )

