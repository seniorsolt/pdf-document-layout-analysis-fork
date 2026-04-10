from functools import wraps
from fastapi import HTTPException

from configuration import service_logger


def catch_exceptions(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        filename = kwargs["file"].filename if kwargs.get("file") else None
        try:
            service_logger.info(f"Calling endpoint: {func.__name__}")
            if filename:
                service_logger.info(f"Processing file: {filename}")
            if kwargs.get("xml_file_name"):
                service_logger.info(f"Asking for file: {kwargs['xml_file_name']}")
            return await func(*args, **kwargs)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="No xml file")
        except Exception:
            service_logger.error(
                f"Error processing file: {filename or '<unknown>'}", exc_info=1
            )
            raise HTTPException(status_code=422, detail="Error see traceback")

    return wrapper
