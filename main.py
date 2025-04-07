from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # ✅ Required to serve /uploads

import pandas as pd
import uuid
import os

from model import CustomModel
from query_engine import process_query, explain_with_openai, validate_answer_with_ground_truth

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Mount /uploads folder for static chart/image serving
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_store = {}
custom_model = CustomModel()

@app.get("/", tags=["Health Check"])
async def root():
    return {
        "status": "ok",
        "message": "🚀 QueryLens backend is live!",
        "endpoints": ["/upload", "/query"]
    }

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.csv")
    with open(file_path, "wb") as f:
        f.write(await file.read())
    df = pd.read_csv(file_path)
    model_store[file_id] = df
    return {"message": "CSV uploaded and model trained", "file_id": file_id}

@app.post("/query")
async def query(file_id: str = Form(...), user_query: str = Form(...)):
    try:
        if file_id not in model_store:
            return JSONResponse(status_code=404, content={"error": "File not found"})

        df = model_store[file_id]

        # 🧠 Get AI response + chart
        summary, chart_path, filtered_df = process_query(df, user_query)

        # 🔍 Sample for context + explanation
        sample_data = filtered_df.head(5).to_dict(orient="records")
        explanation = explain_with_openai(user_query, summary, list(df.columns), sample_data)

        # 🧪 Validation via dynamic logic (GPT-generated pandas)
        validation_score, validation_reason, validation_chart = validate_answer_with_ground_truth(
            user_query, df, summary
        )

        # ✅ Normalize chart paths (important for frontend image rendering)
        if chart_path and chart_path.startswith("./uploads/"):
            chart_path = chart_path.replace("./", "/")

        if validation_chart and validation_chart.startswith("./uploads/"):
            validation_chart = validation_chart.replace("./", "/")

        return JSONResponse({
            "answer": summary,
            "chart": chart_path,
            "explanation": explanation,
            "validation_score": validation_score,
            "validation_reason": validation_reason,
            "validation_chart": validation_chart
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Something went wrong: {str(e)}"})
