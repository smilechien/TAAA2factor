import os
import re
import sys
import uuid
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from flask import Flask, request, send_from_directory, jsonify

BASE_DIR = Path(__file__).resolve().parent
REPORTS_ROOT = BASE_DIR / "web_reports"
UPLOADS_ROOT = BASE_DIR / "web_uploads"
STATIC_DIR = BASE_DIR / "static"

REPORTS_ROOT.mkdir(exist_ok=True)
UPLOADS_ROOT.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}

app = Flask(__name__, static_folder=str(STATIC_DIR))

def safe_name(name: str) -> str:
    name = name or "upload.csv"
    name = os.path.basename(name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:180] if len(name) > 180 else name

@app.get("/")
def home():
    # 永遠回到 static/index.html
    return send_from_directory(str(STATIC_DIR), "index.html")

@app.post("/run")
def run_pipeline():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file field named 'file'."}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"ok": False, "error": "Empty filename."}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"ok": False, "error": f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXT)}"}), 400

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    job_dir = REPORTS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    upload_name = safe_name(f.filename)
    upload_path = UPLOADS_ROOT / f"{job_id}_{upload_name}"
    f.save(str(upload_path))

    enable_pred = request.form.get("enable_pred", "1").strip()
    pred_json = request.form.get("pred_json", "").strip()

    # 用 subprocess 跑 main.py（最穩：不怕 Flask import 時就先跑一堆東西）
    cmd = [
        sys.executable,
        str(BASE_DIR / "main.py"),
        "--input", str(upload_path),
        "--outdir", str(job_dir),
        "--enable_pred", "1" if enable_pred.lower() in ("1", "true", "yes") else "0"
    ]
    if pred_json:
        cmd += ["--pred_json", pred_json]

    proc = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True
    )

    log_path = job_dir / "run.log"
    with open(log_path, "w", encoding="utf-8") as w:
        w.write("=== CMD ===\n" + " ".join(cmd) + "\n\n")
        w.write("=== STDOUT ===\n")
        w.write(proc.stdout or "")
        w.write("\n\n=== STDERR ===\n")
        w.write(proc.stderr or "")

    if proc.returncode != 0:
        return jsonify({
            "ok": False,
            "job_id": job_id,
            "error": "Pipeline failed (non-zero exit code).",
            "log_url": f"/reports/{job_id}/run.log",
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
        }), 500

    report_path = job_dir / "report.html"
    if not report_path.exists():
        # 你的管線如果只產生 report_*.html，這裡幫你補一份 report.html
        cand = sorted(job_dir.glob("report_*.html"))
        if cand:
            shutil.copyfile(str(cand[-1]), str(report_path))

    if not report_path.exists():
        return jsonify({
            "ok": False,
            "job_id": job_id,
            "error": "Pipeline finished but report.html was not found in outdir.",
            "log_url": f"/reports/{job_id}/run.log"
        }), 500

    return jsonify({
        "ok": True,
        "job_id": job_id,
        "report_url": f"/reports/{job_id}/report.html",
        "log_url": f"/reports/{job_id}/run.log",
        "stdout_tail": (proc.stdout or "")[-4000:]
    })

@app.get("/reports/<job_id>/<path:filename>")
def serve_report(job_id, filename):
    d = REPORTS_ROOT / job_id
    return send_from_directory(str(d), filename)

if __name__ == "__main__":
    # 本機跑起來自動開首頁（讓你少點一次）
    try:
        import webbrowser
        webbrowser.open("http://127.0.0.1:8000/")
    except Exception:
        pass

    app.run(host="127.0.0.1", port=8000, debug=False)
