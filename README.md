# Getting started

## Use Docker

- Install and run Docker Engine
- For GPU use (Nvidia on Windows) follow [GPU support in Docker Desktop for Windows
](https://docs.docker.com/desktop/features/gpu/)
- Run Docker services with compose : `docker compose up -d --build`

## AI Agents development

- Create python virtual env with `python -m venv venv && ./venv/Scripts/activate`
- Install dependencies in venv `pip install -r ./requirements.txt`
- You can now work in `./src` folder to dev AI agents (with python's *smolagents*)