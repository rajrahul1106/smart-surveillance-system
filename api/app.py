from __future__ import annotations

import threading
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import init_routes, router as routes_router
from api.ws import init_ws, router as ws_router


def create_app(
    event_bus: Any,
    state_machine: Any,
    storage_service: Any,
    camera_manager: Any,
    config: Any,
    *,
    gesture_model: Any = None,
    face_model: Any = None,
    fire_model: Any = None,
    injury_model: Any = None,
    activity_model: Any = None,
    shared_frame: Any = None,
    enrollment_service: Any = None,
) -> FastAPI:
    app = FastAPI(title="Smart Surveillance API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    init_routes(
        event_bus=event_bus,
        state_machine=state_machine,
        storage_service=storage_service,
        camera_manager=camera_manager,
        config=config,
        gesture_model=gesture_model,
        face_model=face_model,
        fire_model=fire_model,
        injury_model=injury_model,
        activity_model=activity_model,
        shared_frame=shared_frame,
        enrollment_service=enrollment_service,
    )
    init_ws(event_bus=event_bus)

    app.include_router(routes_router)
    app.include_router(ws_router)

    try:
        app.mount("/dashboard", StaticFiles(directory="dashboard/web", html=True), name="dashboard")
    except Exception:
        pass

    return app


def start_server(app: FastAPI, host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    def _run() -> None:
        uvicorn.run(app, host=host, port=port, log_level="info")

    t = threading.Thread(target=_run, daemon=True, name="uvicorn-server")
    t.start()
