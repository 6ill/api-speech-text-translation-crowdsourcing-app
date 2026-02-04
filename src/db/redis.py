import redis.asyncio as redis
from src.core.config import Config

token_blocklist =  redis.Redis(
    host=Config.REDIS_HOST,
    port=Config.REDIS_PORT,
    db=0,
    decode_responses=True,
)


async def add_jti_to_blocklist(jti: str) -> None:
    await token_blocklist.set(
        name=jti, value="", ex=Config.ACCESS_TOKEN_EXPIRY_IN_SECONDS
    )

async def is_token_in_blocklist(jti: str) -> bool:
    jti = await token_blocklist.get(jti)

    return jti is not None
