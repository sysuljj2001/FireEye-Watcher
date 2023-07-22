class Polenum:
    def __init__(
            self,
            id=0,
            uid=0,
            path="",
            request_time=0,
            finished_time=0,
            left_top_x=0,
            left_top_y=0,
            right_bottom_x=0,
            right_bottom_y=0,
            name="") -> None:
        self.id = id
        self.uid = uid
        self.path = path
        self.request_time = request_time
        self.finished_time = finished_time
        self.left_top_x = left_top_x
        self.left_top_y = left_top_y
        self.right_bottom_x = right_bottom_x
        self.right_bottom_y = right_bottom_y
        self.name = name

    def to_json(self):
        return {
            "id": self.id,
            "uid": self.uid,
            "path": self.path,
            "request_time": self.request_time,
            "finished_time": self.finished_time,
            "left_top_x": self.left_top_x,
            "left_top_y": self.left_top_y,
            "right_bottom_x": self.right_bottom_x,
            "right_bottom_y": self.right_bottom_y,
            "name": self.name
        }