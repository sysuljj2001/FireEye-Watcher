from model import Polenum
from pymysql import Connection

class PolenumRepository:
    def __init__(self, conn: Connection) -> None:
        self.conn = conn
        self.cursor = self.conn.cursor()

    def create_polenum(self, polenum: Polenum) -> Polenum:
        self.ping()
        sql = "INSERT INTO `records` (`uid`, `path`, `request_time`, `finished_time`, \
            `left_top_x`, `left_top_y`, `right_bottom_x`, `right_bottom_y`, \
                `name`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        self.cursor.execute(sql, (polenum.uid, polenum.path, polenum.request_time, polenum.finished_time, 
                                  polenum.left_top_x, polenum.left_top_y, 
                                  polenum.right_bottom_x, polenum.right_bottom_y,
                                  polenum.name))
        self.conn.commit()
        return polenum

    def ping(self):
        self.conn.ping(reconnect=True)