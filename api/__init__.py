from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile
import tensorflow as tf
from PIL import Image
import numpy as np
from json import load

ml_tools = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
  ml_tools["model"] = tf.saved_model.load("./api/model")
  ml_tools["names"] = load(open("./api/class_names.json"))
  yield
  ml_tools.clear()


app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(picture: UploadFile):
  img = Image.open(picture.file).resize((320,320)).convert('RGB')
  img = np.array(img).reshape((1,320,320,3))
  pred = ml_tools["model"].serve(img)[0]
  idx = int(tf.math.argmax(pred))
  perc = pred[idx]
  print({ml_tools["names"][i]:f"{x:.2%}" for i, x in enumerate(pred)})
  return {
    "pred": ml_tools["names"][idx],
    "certainty": f"{perc:.2%}"
  }