from abc import ABC, abstractmethod

from datetime import datetime

from fastapi import HTTPException, Request, status, Depends
from fastapi.security import HTTPBearer
from .service import UserService
from .utils import decode_token
from src.core.errors import AccessTokenRequired, InsufficientPermission, InvalidToken, RefreshTokenRequired,RevokedToken
from src.db.main import get_session
from src.db.models import User, Role
from src.db.redis import is_token_in_blocklist
from sqlmodel.ext.asyncio.session import AsyncSession
from typing import List


user_service = UserService()

class TokenBearer(HTTPBearer, ABC):
    def __init__(
        self, *, bearerFormat=None, scheme_name=None, description=None, auto_error=True
    ):
        super().__init__(
            bearerFormat=bearerFormat,
            scheme_name=scheme_name,
            description=description,
            auto_error=auto_error,
        )

    async def __call__(self, request: Request) -> dict:
        creds = await super().__call__(request)
        token = creds.credentials
        decoded_token = decode_token(token)

        if decoded_token is None:
            raise InvalidToken()

        # When token has been revoked means the user has logged out
        if await is_token_in_blocklist(decoded_token["jti"]):
            raise RevokedToken()

        if self.is_expired(decoded_token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "This token is expired",
                    "resolution": "Please get new token",
                },
            )

        self.verify_token_data(decoded_token)

        return decoded_token

    def is_expired(self, token_data:dict) -> bool:
        expiry_timestamp = token_data["exp"]

        return datetime.fromtimestamp(expiry_timestamp) <= datetime.now()

    @abstractmethod
    def verify_token_data(self, token_data) -> None:
        raise NotImplementedError("Please Override this method in child classes")

class AccessTokenBearer(TokenBearer):
    def verify_token_data(self, token_data:dict) -> None:
        if token_data and token_data["refresh"]:
            raise AccessTokenRequired()

class RefreshTokenBearer(TokenBearer):
    def verify_token_data(self, token_data:dict) -> None:
        if token_data and not token_data["refresh"]:
            raise RefreshTokenRequired()

async def get_current_user(
    token_details: dict = Depends(AccessTokenBearer()),
    session: AsyncSession = Depends(get_session),
) -> User:
    user_email = token_details["user"]["email"]

    user = await user_service.get_user_by_email(user_email, session)

    return user

class RoleChecker:
    def __init__(self, allowed_roles: List[Role]) -> None:
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: User = Depends(get_current_user)) -> bool:
        if current_user.role in self.allowed_roles:
            return True

        raise InsufficientPermission()