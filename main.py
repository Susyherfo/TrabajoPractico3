from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# ----- MODELOS -----

class User(BaseModel):
    id: int
    nombre: str
    apellido: str
    email: str

class Product(BaseModel):
    id: int
    nombre: str
    descripcion: str
    precio: float

# ----- BASES DE DATOS TEMPORALES -----

users = []
products = []

# ----- ENDPOINTS -----

@app.post("/users/")
def create_user(user: User):
    users.append(user)
    return {"message": "Usuario creado correctamente", "user": user}

@app.get("/users/{user_id}")
def get_user(user_id: int):
    for user in users:
        if user.id == user_id:
            return user
    return {"error": "Usuario no encontrado"}

@app.post("/products/")
def create_product(product: Product):
    products.append(product)
    return {"message": "Producto creado correctamente", "product": product}

@app.get("/products/")
def list_products():
    return products