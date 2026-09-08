"""FACTLINE FastAPI Application Entrypoint."""

import os
from dotenv import load_dotenv

# Automatically load environment variables from .env if present
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router

app = FastAPI(
    title="FACTLINE API",
    description="Evidence-First Cross-Document Fact Intelligence API",
    version="0.1.0",
)

# Configure CORS for local development and eventual React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "name": "FACTLINE API",
        "version": "0.1.0",
        "docs": "/docs",
        "status": "online",
    }
