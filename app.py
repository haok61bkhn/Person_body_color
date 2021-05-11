from typing import Optional
from fastapi import FastAPI,APIRouter, Request, Depends, File, UploadFile
from pydantic import BaseModel
from io import BytesIO
import numpy as np
from person_color import Person_body_color
from PIL import Image
person_color =Person_body_color()
class Item(BaseModel):
    data: str


app = FastAPI()
def read_imagefile(data) -> Image.Image:
    image = Image.open(BytesIO(data))
    return image

@app.post("/person_body_color/")
async def create_upload_file(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    image = np.array(image) 
    image = image[:, :, ::-1]
    image = np.array(image) 
    res=person_color.predict(image)

    return res