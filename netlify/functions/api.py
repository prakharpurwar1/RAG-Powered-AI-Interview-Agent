import os
from mangum import Mangum
from interviewer.app import app

# Set default DB path for serverless writable directory
os.environ.setdefault("INTERVIEW_DB", "/tmp/interviews.db")

handler = Mangum(app, lifespan="off")
