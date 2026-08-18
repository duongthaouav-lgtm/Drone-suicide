import cv2

class LogicTracker:

    def __init__(self):

        self.bbox = None
        self.template = None

    def init(self, frame, bbox):

        x, y, w, h = map(int, bbox)

        self.bbox = (x, y, w, h)

        roi = frame[y:y+h, x:x+w]

        self.template = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    def update(self, frame):

        if self.bbox is None:
            return False, None

        x, y, w, h = self.bbox

        h_frame, w_frame = frame.shape[:2]

        # 🔍 search area quanh object cũ
        padding = 90

        sx1 = max(x - padding, 0)
        sy1 = max(y - padding, 0)

        sx2 = min(x + w + padding, w_frame)
        sy2 = min(y + h + padding, h_frame)

        search_roi = frame[sy1:sy2, sx1:sx2]

        if search_roi.shape[0] <= 0 or search_roi.shape[1] <= 0:
            return False, self.bbox

        gray_search = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)

        # template matching
        result = cv2.matchTemplate(
            gray_search,
            self.template,
            cv2.TM_CCOEFF_NORMED
        )

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # confidence thấp
        if max_val < 0.45:
            return False, self.bbox

        nx = sx1 + max_loc[0]
        ny = sy1 + max_loc[1]


        alpha = 0.7
        nx = int(alpha * x + (1-alpha) * nx)
        ny = int(alpha * y + (1-alpha) * ny)

        self.bbox = (nx, ny, w, h)

        return True, self.bbox