import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(funcName)s() @ line %(lineno)d | %(message)s"
)
log = logging.getLogger("acai")