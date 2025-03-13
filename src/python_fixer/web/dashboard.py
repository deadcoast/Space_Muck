"""Web dashboard for Python Import Fixer."""

# Standard library imports
import logging

# Third-party library imports

# Local application imports
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from python_fixer.base import (
from python_fixer.core.project_analysis import ProjectAnalyzer
from typing import Dict, List, Optional
import asyncio
import uvicorn

    DEFAULT_DASHBOARD_HOST,
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_DASHBOARD_RELOAD,
    LogRecord,
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Python Import Fixer Dashboard")
    yield
    # Shutdown
    logger.info("Shutting down Python Import Fixer Dashboard")

app = FastAPI(
    title="Python Import Fixer Dashboard",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Mount static files
try:
    templates_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"

    if not templates_dir.exists():
        raise FileNotFoundError(f"Templates directory not found: {templates_dir}")
    if not static_dir.exists():
        static_dir.mkdir(parents=True)
        logger.info(f"Created static directory: {static_dir}")

    templates = Jinja2Templates(directory=str(templates_dir))
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
except Exception as e:
    logger.error(f"Failed to initialize dashboard resources: {e}")
    raise

# Global state
project_analyzer: Optional[ProjectAnalyzer] = None
analysis_results: Dict = {}
log_buffer: List[Dict] = []

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the dashboard homepage."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "project_path": project_analyzer.project_path if project_analyzer else None,
            "analysis_results": analysis_results,
            "recent_logs": log_buffer[-50:],  # Show last 50 logs
        },
    )

@app.get("/analyze")
async def analyze_project():
    """Trigger project analysis."""
    if not project_analyzer:
        raise HTTPException(status_code=400, detail="No project initialized")

    try:
        logger.info(f"Starting analysis of {project_analyzer.project_path}")
        results = project_analyzer.analyze_project()
        analysis_results.update(results)

        log_buffer.append(
            LogRecord(
                level=logging.INFO,
                message=f"Analysis complete: {len(results.get('structure', {}).get('modules', []))} modules analyzed",
                timestamp=results.get("timestamp", ""),
                module=__name__,
                extra={"metrics": results},
            ).to_dict()
        )

        return {"status": "success", "results": results}

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        log_buffer.append(
            LogRecord(
                level=logging.ERROR,
                message=f"Analysis failed: {str(e)}",
                timestamp=results.get("timestamp", ""),
                module=__name__,
            ).to_dict()
        )
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/fix")
async def fix_project(mode: str = "interactive"):
    """Apply fixes to the project."""
    if not project_analyzer:
        raise HTTPException(status_code=400, detail="No project initialized")

    if mode not in ["interactive", "automatic"]:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")

    try:
        logger.info(
            f"Starting fixes in {mode} mode for {project_analyzer.project_path}"
        )
        fixes = project_analyzer.fix_project(mode=mode)

        log_buffer.append(
            LogRecord(
                level=logging.INFO,
                message=f"Fixes complete: {fixes.get('imports_fixed', 0)} imports fixed",
                timestamp=fixes.get("timestamp", ""),
                module=__name__,
                extra={"metrics": fixes},
            ).to_dict()
        )

        return {"status": "success", "fixes": fixes}

    except Exception as e:
        logger.error(f"Fix failed: {e}", exc_info=True)
        log_buffer.append(
            LogRecord(
                level=logging.ERROR,
                message=f"Fix failed: {str(e)}",
                timestamp=fixes.get("timestamp", ""),
                module=__name__,
            ).to_dict()
        )
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/logs")
async def get_logs(limit: int = 50):
    """Get recent logs."""
    return {"logs": log_buffer[-limit:]}

@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    try:
        while True:
            if log_buffer:
                await websocket.send_json({"logs": log_buffer[-1:]})
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

def run_dashboard(
    project_path: Path,
    host: str = DEFAULT_DASHBOARD_HOST,
    port: int = DEFAULT_DASHBOARD_PORT,
    reload: bool = DEFAULT_DASHBOARD_RELOAD,
    log_level: str = "info",
) -> None:
    """Run the web dashboard.

    Args:
        project_path: Path to the project to analyze
        host: Host to bind to
        port: Port to listen on
        reload: Whether to enable auto-reload
        log_level: Logging level for the dashboard

    Raises:
        ValueError: If project_path does not exist
        RuntimeError: If dashboard initialization fails
    """
    global project_analyzer

    # Validate project path
    if not project_path.exists():
        raise ValueError(f"Project path does not exist: {project_path}")

    try:
        # Initialize project analyzer
        project_analyzer = ProjectAnalyzer(str(project_path))
        logger.info(f"Initialized project analyzer for {project_path}")

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Run the dashboard
        logger.info(f"Starting dashboard on {host}:{port}")
        uvicorn.run(
            "python_fixer.web.dashboard:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=True,
        )

    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}", exc_info=True)
        raise RuntimeError(f"Failed to start dashboard: {e}") from e
