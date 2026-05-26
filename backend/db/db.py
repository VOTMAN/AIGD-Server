import os

from .models import PredResults
from sqlmodel import Session, SQLModel, create_engine, select

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "data", "results.db"))

SQLLITE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(SQLLITE_URL, connect_args={"check_same_thread": False})

def initDB():
    SQLModel.metadata.create_all(engine)

def saveResult(result: PredResults):
    try:
        with Session(engine) as session:
            existing = session.get(PredResults, result.id)
            if existing:
                for key, value in result.model_dump().items():
                    setattr(existing, key, value)
                session.add(existing)
            else:
                session.add(result)
            session.commit()
    except Exception as e:
        print(e)
        return { "success": False, "error": e }    
    return { "success": True }

def getResult(id: str):
    try:
        with Session(engine) as session:
            # statement = select(PredResults).where(PredResults.id == id)
            # res = session.exec(statement).first()
            # print(res)
            # return res
            return session.get(PredResults, id)
    except Exception as e:
        print(e)
        return { "success": False, "error": e }    

def getAllResults() -> list[PredResults]:
    try:
        with Session(engine) as session:
            statement = select(PredResults).order_by(PredResults.created_datetime.desc())
            results = session.exec(statement)
            res = results.all()
            return res
    except Exception as e:
        print(e)
        return { "success": False, "error": e }    
