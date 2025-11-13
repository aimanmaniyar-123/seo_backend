#!/bin/bash
uvicorn updated_main:app --host 0.0.0.0 --port $PORT
