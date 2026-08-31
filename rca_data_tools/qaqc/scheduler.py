"""CONDUCTOR FLOW: triggers QAQC_dashboard gh_workflows instead up relying on github cron 
Every target already exposes `workflow_dispatch:`, so we just POST to the dispatch API on a Prefect schedule.
Schedules live in prefect.yaml; the echoharvest -> echogram dependency is the
one case currently that needs a real completion check rather than a time gap.
"""
import os
from datetime import datetime, timezone

import requests
from prefect import flow, task, get_run_logger

OWNER, REPO, REF = "OOI-CabledArray", "QAQC_dashboard", "main"
API = f"https://api.github.com/repos/{OWNER}/{REPO}"


def _headers():
    return {
        "Authorization": f"Bearer {os.environ['GH_PAT']}",
        "Accept": "application/vnd.github+json",
    }


@task
def dispatch(workflow_file: str) -> datetime:
    # workflow_dispatch returns no run id, so stamp the time to find the run later
    t0 = datetime.now(timezone.utc)
    r = requests.post(
        f"{API}/actions/workflows/{workflow_file}/dispatches",
        headers=_headers(),
        json={"ref": REF},
    )
    r.raise_for_status()
    get_run_logger().info(f"dispatched {workflow_file} @ {t0.isoformat()}")
    return t0


@task(retries=60, retry_delay_seconds=30)  # poll up to ~30 min for completion
def wait_for_run(workflow_file: str, since: datetime):
    runs = requests.get(
        f"{API}/actions/workflows/{workflow_file}/runs",
        headers=_headers(),
        params={"event": "workflow_dispatch", "per_page": 5},
    ).json()["workflow_runs"]
    run = next(
        r for r in runs
        if datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")) >= since
    )
    assert run["status"] == "completed", f"{workflow_file} still {run['status']}"
    if run["conclusion"] != "success":
        raise RuntimeError(f"{workflow_file} -> {run['conclusion']}")
    get_run_logger().info(f"{workflow_file} completed: {run['html_url']}")


@flow
def trigger_workflow(workflow_file: str):
    """Fire one workflow_dispatch and return."""
    dispatch(workflow_file)


@flow
def trigger_daily_batch():
    """The 11:00 UTC set. Weekday split mirrors the workflow crons: hydrophones
    run every day; the compute-heavy _weekly variants run Monday only; the daily
    variants run Tue-Fri (so Monday saves compute)."""
    dispatch("hydrophone_docker.yaml")
    dispatch("lf_hydrophone.yaml")
    dow = datetime.now(timezone.utc).weekday()  # Mon=0
    if dow == 0:
        dispatch("pipeline_weekly.yaml")
        dispatch("seismometer_weekly.yaml")
    elif dow in (1, 2, 3, 4):
        dispatch("pipeline.yaml")
        dispatch("seismometer.yaml")


@flow
def trigger_echo_chain():
    """Harvest -> wait for the zarr write to finish -> plot echograms."""
    t0 = dispatch("echoharvest_daily.yaml")
    wait_for_run("echoharvest_daily.yaml", t0)
    dispatch("echogram_daily.yaml")
