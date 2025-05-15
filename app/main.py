import requests  # used to call other APIs (authenticate, authorize, preprocess, predict)
from fastapi import FastAPI, UploadFile
import uvicorn


# instantiate API
app = FastAPI(
    title="User Interface",
    version="0.0.1",
    description="provide the AI service for end users"
    # openapi_tags=[]
)

# endpoint: API's status
@app.get("/", name="")
def get_status():
    return {"status": "Alive",
            "message": "please call /api for more extensive information"}

# endpoint: API's info
@app.get("/api", name="info")
def get_info():
    return {"status": "Alive",
            "version": app.version,
            "API_documentation": "/docs",
            "manifest": "/openapi.json",
            "authentication": "/api/login",
            "predict_image": "/api/image",
            "predict_text": "/api/text"}

# endpoint: user's authentication
# @app.post()
# call Authenticate API
# call Authorize API

# endpoint: predict image
# option1: receive image from POST request, send it to preprocess, receive preprocessed image, send it to predict
# option2: receive image from POST request, save it to shared volume, sent the path to preprocess, receive the path of preprocessed image, send it to predict
# option3: receive image from POST request, save it to DB, sent key to preprocess, receive key of preprocessed image, send it to predict
@app.post("/api/image")
def upload_img(img: UploadFile):
    """
    img = requests.post(
            "http://127.0.0.1:8001/preprocess",
            files=img
            )
    """
    img_class = requests.post(
        "http://127.0.0.1:8002/predict",
        files={"img": img.file}
    ).json()
    return {"image": img.filename, "class": img_class["class"]}

# call preprocess_image API
# call predict_image API

# endpoint: predict text
# @app.post("api
# call preprocess_text API
# call predict_text API


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
