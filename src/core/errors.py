from fastapi import FastAPI, status
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from typing import Any, Callable

class WebException(Exception):
    """This is the base class for all web errors"""
    pass

class InvalidToken(WebException):
    """User has provided an invalid or expired token"""
    pass

class RevokedToken(WebException):
    """User has provided a token that has been revoked"""
    pass

class AccessTokenRequired(WebException):
    """User has provided a refresh token when an access token is needed"""
    pass

class RefreshTokenRequired(WebException):
    """User has provided an access token when a refresh token is needed"""
    pass

class UserAlreadyExists(WebException):
    """User has provided an email for a user who exists during sign up."""
    pass

class InvalidCredentials(WebException):
    """User has provided wrong email or password during log in."""
    pass

class InsufficientPermission(WebException):
    """User does not have the neccessary permissions to perform an action."""
    pass

class FileNotFound(WebException):
    """File Not found"""
    pass

class UserNotFound(WebException):
    """User Not found"""
    pass

class InvalidAudioFormat(WebException):
    """Invalid audio file format."""
    pass

class FileNotTranscribed(WebException):
    """User tries to translate a file that has not been transcribed yet."""
    pass

class FileNotTranslated(WebException):
    """User tries to download a file that has not been translated yet."""
    pass

class TranslationInProgress(WebException):
    """File is already being translated."""
    pass

class PipelineIsNotActive(WebException):
    """The configuration of pipeline for certain task is disabled"""
    pass

def create_exception_handler(
    status_code: int, 
    initial_detail: Any,
) -> Callable[[Request, Exception], JSONResponse]:
    async def exception_handler(request: Request, exc: WebException):

        return JSONResponse(content=initial_detail, status_code=status_code)

    return exception_handler


def register_all_errors(app: FastAPI):
    app.add_exception_handler(
        UserAlreadyExists,
        create_exception_handler(
            status_code=status.HTTP_409_CONFLICT,
            initial_detail={
                "message": "User with this email already exists",
                "error_code": "user_exists",
            },
        ),
    )

    app.add_exception_handler(
        UserNotFound,
        create_exception_handler(
            status_code=status.HTTP_404_NOT_FOUND,
            initial_detail={
                "message": "User not found",
                "error_code": "user_not_found",
            },
        ),
    )
    app.add_exception_handler(
        FileNotFound,
        create_exception_handler(
            status_code=status.HTTP_404_NOT_FOUND,
            initial_detail={
                "message": "File not found",
                "error_code": "file_not_found",
            },
        ),
    )

    app.add_exception_handler(
        InvalidAudioFormat,
        create_exception_handler(
            status_code=status.HTTP_400_BAD_REQUEST,
            initial_detail={
                "message": "Invalid audio format. Allowed: mp3, wav, m4a, ogg",
                "error_code": "invalid_file_format",
            },
        ),
    )
    
    app.add_exception_handler(
        InvalidCredentials,
        create_exception_handler(
            status_code=status.HTTP_400_BAD_REQUEST,
            initial_detail={
                "message": "Invalid Email Or Password",
                "error_code": "invalid_email_or_password",
            },
        ),
    )
    
    app.add_exception_handler(
        InvalidToken,
        create_exception_handler(
            status_code=status.HTTP_401_UNAUTHORIZED,
            initial_detail={
                "message": "Token is invalid Or expired",
                "resolution": "Please sign in to get new token",
                "error_code": "invalid_token",
            },
        ),
    )
    app.add_exception_handler(
        RevokedToken,
        create_exception_handler(
            status_code=status.HTTP_401_UNAUTHORIZED,
            initial_detail={
                "message": "Token is invalid or has been revoked",
                "resolution": "Please sign in to get new token",
                "error_code": "token_revoked",
            },
        ),
    )
    app.add_exception_handler(
        AccessTokenRequired,
        create_exception_handler(
            status_code=status.HTTP_401_UNAUTHORIZED,
            initial_detail={
                "message": "Please provide a valid access token",
                "resolution": "Please sign in to get an access token",
                "error_code": "access_token_required",
            },
        ),
    )
    app.add_exception_handler(
        RefreshTokenRequired,
        create_exception_handler(
            status_code=status.HTTP_403_FORBIDDEN,
            initial_detail={
                "message": "Please provide a valid refresh token",
                "resolution": "Please sign in to get an refresh token",
                "error_code": "refresh_token_required",
            },
        ),
    )
    app.add_exception_handler(
        InsufficientPermission,
        create_exception_handler(
            status_code=status.HTTP_403_FORBIDDEN,
            initial_detail={
                "message": "You do not have enough permissions to perform this action",
                "error_code": "insufficient_permissions",
            },
        ),
    )
    
    app.add_exception_handler(
        FileNotTranscribed,
        create_exception_handler(
            status_code=status.HTTP_400_BAD_REQUEST,
            initial_detail={
                "message": "File must be transcribed first",
                "error_code": "file_not_transcribed",
            },
        ),
    )
    
    app.add_exception_handler(
        FileNotTranslated,
        create_exception_handler(
            status_code=status.HTTP_400_BAD_REQUEST,
            initial_detail={
                "message": "File must be translated first",
                "error_code": "file_not_translated",
            },
        ),
    )
    
    app.add_exception_handler(
        TranslationInProgress,
        create_exception_handler(
            status_code=status.HTTP_409_CONFLICT,
            initial_detail={
                "message": "Translation process is already running",
                "error_code": "translation_in_progress",
            },
        ),
    )

    app.add_exception_handler(
        PipelineIsNotActive,
        create_exception_handler(
            status_code=status.HTTP_409_CONFLICT,
            initial_detail={
                "message": "The pipeline for this task is not active",
                "error_code": "pipeline_is_disabled",
            },
        ),
    )

    @app.exception_handler(500)
    async def internal_server_error(request, exc):

        return JSONResponse(
            content={
                "message": "Oops! Something went wrong",
                "error_code": "server_error",
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
