from datetime import timedelta
from sqlite3 import IntegrityError
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlmodel.ext.asyncio.session import AsyncSession


from .dependency import AccessTokenBearer, RefreshTokenBearer, RoleChecker, get_current_user
from .schema import UserModel, UserCreateModel
from .service import UserService
from .utils import create_access_token, create_url_safe_token, verify_password
from src.core.config import Config
from src.core.errors import InvalidCredentials, UserAlreadyExists, UserNotFound
from src.db.main import get_session
from src.db.models import User, Role
from src.db.redis import add_jti_to_blocklist


role_checker = Depends(RoleChecker([Role.ADMIN, Role.USER]))
router = APIRouter()
user_service = UserService()

RefreshToken = Annotated[dict, Depends(RefreshTokenBearer())]
AccessToken = Annotated[dict, Depends(AccessTokenBearer())]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]

@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def create_user_account(
    user_data: UserCreateModel,
    session: SessionDep,
): 
    """
    Create user account using email, username, first_name, last_name
    params:
        user_data: UserCreateModel
    """
    is_user_exists = await user_service.user_exists(user_data.email, session)

    if is_user_exists:
        raise UserAlreadyExists
    
    try:
        new_user = await user_service.create_user(user_data, session)

    except IntegrityError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User with this username already exists")
        
    return {
        "message": "Account Created!",
        "data": new_user.model_dump()
    }

@router.post("/signin")
async def login_user(
    login_data: UserCreateModel, 
    session: SessionDep,
):
    user = await user_service.get_user_by_email(login_data.email, session)

    if not user:
        raise UserNotFound()

    password_valid = verify_password(login_data.password, user.password_hash)
    if password_valid:
        access_token = create_access_token(user_data={"email": user.email, "user_id": str(user.id)})

        refresh_token = create_access_token(
            user_data={"email": user.email, "user_id": str(user.id)}, 
            expiry=timedelta(seconds=Config.REFRESH_TOKEN_EXPIRY_IN_SECONDS),
            refresh=True
        )

        return JSONResponse(
            content={
                "message": "Login successful",
                "data": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "user": {"email": user.email, "id": str(user.id)},
                }
            }
        )
    
    raise InvalidCredentials()

@router.get("/signout")
async def logout(token_details: AccessToken):
    jti = token_details["jti"]
    await add_jti_to_blocklist(jti)

    return JSONResponse(
        content={"message": "Logged Out Successfully"}, status_code=status.HTTP_200_OK
    )

@router.get("/refresh_token")
async def get_new_access_token(token_details: RefreshToken):
    new_access_token = create_access_token(user_data=token_details["user"])

    return JSONResponse(content={"access_token": new_access_token})

@router.get("/me", response_model=UserModel)
async def my_profile(current_user: CurrentUser):
    return current_user