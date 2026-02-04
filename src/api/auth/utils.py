from fastapi.exceptions import HTTPException
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
import uuid
import logging
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from typing import Any

from src.core.config import Config

passwd_context = CryptContext(schemes=["argon2", "bcrypt"])
url_serializer = URLSafeTimedSerializer(secret_key=Config.JWT_SECRET, salt="email-configuration")

def generate_password_hash(password: str) -> str:
    hash = passwd_context.hash(password)
    return hash

def verify_password(password: str, hash: str) -> bool:
    return passwd_context.verify(password, hash)

def create_access_token(
    user_data: dict,
    expiry: timedelta = timedelta(seconds=Config.ACCESS_TOKEN_EXPIRY_IN_SECONDS),
    refresh: bool = False,
) -> str:
    payload = dict()

    payload["user"] = user_data
    payload["exp"] = datetime.now() + expiry
    payload["jti"] = str(uuid.uuid4()) # unique id for jwt
    payload["refresh"] = refresh

    token = jwt.encode(
        payload=payload, key=Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM
    )

    return token

def decode_token(token: str) -> dict | None:
    try:
        token_data = jwt.decode(
            jwt=token,
            key=Config.JWT_SECRET,
            algorithms=[Config.JWT_ALGORITHM]
        )

        return token_data
    except jwt.PyJWTError as jwte:
        logging.exception(jwte)
        return None
    except Exception as e:
        logging.exception(e)
        return None
    
def create_url_safe_token(data: dict) -> str:
    """Serialize a dict into a URLSafe token"""

    token = url_serializer.dumps(data)

    return token

def decode_url_safe_token(token: str) -> Any:
    try:
        token_data = url_serializer.loads(token, max_age=Config.ACCESS_TOKEN_EXPIRY_IN_SECONDS)

        return token_data
    except SignatureExpired:
        raise HTTPException(status_code=400, detail="Token has expired")
    except BadSignature:
        raise HTTPException(status_code=400, detail="Invalid token")
    except Exception as e:
        logging.error(str(e))
        raise e
