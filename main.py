"""Entry point for the LLM Customer Segmentation Ads API."""

import uvicorn
from src.api import create_app

app = create_app()


def main():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
