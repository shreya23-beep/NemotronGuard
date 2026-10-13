import os
import re
import subprocess
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI(title="NemotronGuard Autonomous SRE Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("Missing NVIDIA_API_KEY in .env file")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

# NVIDIA Nemotron models hosted on NIM
MODEL_FAST = "nvidia/nemotron-3.5-lightning-30b-a3b"
MODEL_REASONING = "nvidia/nemotron-3-ultra-550b-a55b"

class IncidentRequest(BaseModel):
    broken_code: str
    error_log: str

def clean_code(raw_text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_text.strip()

@app.post("/api/heal")
def heal_code(req: IncidentRequest):
    try:
        # Step 1: Rapid Incident Triage with Nemotron Lightning (30B)
        triage_prompt = (
            "You are a Site Reliability Engineer. In 2 concise sentences, "
            "diagnose the root cause of this failure:\n"
            f"Code:\n{req.broken_code}\n\nLog:\n{req.error_log}"
        )
        triage_res = client.chat.completions.create(
            model=MODEL_FAST,
            messages=[{"role": "user", "content": triage_prompt}],
            temperature=0.2,
            max_tokens=256,
        )
        triage_summary = triage_res.choices[0].message.content

        # Step 2: Code Patch Synthesis with Nemotron Ultra (550B)
        patch_prompt = (
            "You are an automated software repair agent. Fix the following Python code "
            "so that all logic errors are resolved and tests/assertions pass.\n"
            "Respond ONLY with the complete, fixed Python code wrapped inside a ```python ``` code block. "
            "Do not include any conversational preamble or sign-off.\n\n"
            f"Broken Code:\n{req.broken_code}\n\n"
            f"Root Cause Analysis:\n{triage_summary}"
        )
        patch_res = client.chat.completions.create(
            model=MODEL_REASONING,
            messages=[{"role": "user", "content": patch_prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        fixed_code = clean_code(patch_res.choices[0].message.content)

        # Step 3: Autonomous Sandbox Execution
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
            temp_file.write(fixed_code)
            temp_path = temp_file.name

        try:
            exec_res = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=8,
            )
            passed = exec_res.returncode == 0
            sandbox_output = exec_res.stdout if passed else exec_res.stderr
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return {
            "status": "success",
            "triage_summary": triage_summary,
            "fixed_code": fixed_code,
            "sandbox_status": "PASSED" if passed else "FAILED",
            "sandbox_output": sandbox_output.strip() or "Execution completed with exit code 0.",
            "models_used": {
                "triage": MODEL_FAST,
                "reasoning": MODEL_REASONING,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)