import argparse
import os
from pathlib import Path

import requests


def download_file(url: str, output: str) -> None:
    if not url:
        raise ValueError("MODEL_URL esta vacio. Configuralo antes de construir o probar.")
    if not url.startswith(("http://", "https://")):
        raise ValueError(
            "MODEL_URL no parece una URL valida. Debe empezar por https:// y no debe incluir "
            "el texto MODEL_URL= dentro del valor."
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    print(f"Modelo descargado en {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("MODEL_URL", ""))
    parser.add_argument("--output", default=os.getenv("MODEL_PATH", "artifacts/model.onnx"))
    args = parser.parse_args()
    download_file(args.url, args.output)


if __name__ == "__main__":
    main()
