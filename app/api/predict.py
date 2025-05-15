from fastapi import FastAPI, UploadFile
import uvicorn

#instantiate API
app = FastAPI(
	title="internal - image class prediction",
    version="0.0.1",
    description="internal API"
    #openapi_tags=[]
    )

@app.post("/predict")
def receive_img(img: UploadFile):
	img_class = 33
	return {"image": img.filename, "class": img_class}

if __name__ == "__main__":
	uvicorn.run(app, host="127.0.0.1", port=8002)
