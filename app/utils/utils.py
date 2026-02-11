import shutil
import os
from app.config.config import Config

def cleanup_session(session_id: str):
    """
    Deletes uploaded files and vector DB for a session.
    """
    uploads_path = os.path.join(Config.UPLOAD_DIR, session_id)
    if os.path.exists(uploads_path):
        shutil.rmtree(uploads_path)

    vector_db_path=os.path.join(
        Config.VECTOR_DB_DIR,
        f"{session_id}_vector_db"
    )
    if os.path.exists(vector_db_path):
        shutil.rmtree(vector_db_path)
