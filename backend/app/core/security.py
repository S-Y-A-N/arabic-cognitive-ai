# hash password, rate limiting
from typing import Dict
from collections import defaultdict
import time
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"])

def hash_password(password: str):
  return pwd_context.hash(password)

class RateLimiter:
    def __init__(self, rpm: int = 40):
        self.rpm = rpm
        self._log: Dict[str, list] = defaultdict(list)

    def check(self, key: str) -> bool:
        now = time.time()
        self._log[key] = [t for t in self._log[key] if now - t < 60]
        if len(self._log[key]) >= self.rpm: return False
        self._log[key].append(now)
        return True


limiter = RateLimiter()