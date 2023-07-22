from log import logger
from model import Polenum
from repository import PolenumRepository

class PolenumService:
    def __init__(self, polenum_repo: PolenumRepository) -> None:
        self.polenum_repo = polenum_repo

    def create_polenum(self, polenum: Polenum) -> Polenum:
        if polenum == None:
            return None
        if polenum.uid <= 0:
            logger.error("invalid user id")
            return None
        return self.polenum_repo.create_polenum(polenum)