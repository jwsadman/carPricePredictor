from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from app.model_utils import load_model, preprocess_input
import uvicorn

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

ml_model = load_model()
print("Model loaded:", type(ml_model))

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
def predict_form(
    request: Request,
    make: str = Form(...),
    model: str = Form(...),
    body: str = Form(...),
    condition: str = Form(...),
    odometer: float = Form(...),
    saleyear: int = Form(...)
):
    try:
        input_data = {
            "make": make,
            "model": model,
            "body": body,
            "condition": condition,
            "odometer": odometer,
            "saleyear": saleyear
        }

        processed = preprocess_input(input_data)
        prediction = ml_model.predict(processed)
        predicted_price = max(0, round(float(prediction[0]), 2))

        return templates.TemplateResponse("form.html", {
            "request": request,
            "result": f"Predicted Price: ${predicted_price}"
        })

    except Exception as e:
        return templates.TemplateResponse("form.html", {
            "request": request,
            "result": f"Error: {str(e)}"
        })

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080)


