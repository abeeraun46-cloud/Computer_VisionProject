import math
import time

class BehaviorAnalyzer:
    def __init__(self, max_disappeared=30, max_distance=80):
        self.next_id = 0

        # الأشخاص الحاليين
        self.objects = {}      # id -> (cx, cy)
        self.last_seen = {}    # id -> آخر وقت شوهد فيه
        self.frames_seen = {}  # id -> عدد الفريمات
        self.counted_ids = set()

        # إعدادات التحكم
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        # العد النهائي
        self.total_unique_people = 0

    def update(self, detections):
        current_time = time.time()
        updated_objects = []

        # إذا ما في أي كشف
        if len(detections) == 0:
            self._cleanup_disappeared(current_time)
            return []

        # حساب مراكز الصناديق الجديدة
        centers = []
        for (x1, y1, x2, y2) in detections:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centers.append((cx, cy, (x1, y1, x2, y2)))

        matched_ids = set()

        # مطابقة الكشوف مع الأشخاص الموجودين
        for (cx, cy, bbox) in centers:
            best_match = None
            min_dist = float("inf")

            for pid, (px, py) in self.objects.items():
                dist = math.hypot(px - cx, py - cy)

                if dist < self.max_distance and dist < min_dist:
                    min_dist = dist
                    best_match = pid

            if best_match is not None:
                # تحديث الشخص الحالي
                self.objects[best_match] = (cx, cy)
                self.last_seen[best_match] = current_time
                self.frames_seen[best_match] += 1
                matched_ids.add(best_match)

                status = self._analyze_behavior(best_match)

                updated_objects.append({
                    "id": best_match,
                    "bbox": bbox,
                    "status": status
                })

            else:
                # إنشاء ID جديد
                pid = self.next_id
                self.next_id += 1

                self.objects[pid] = (cx, cy)
                self.last_seen[pid] = current_time
                self.frames_seen[pid] = 1

                self.total_unique_people += 1
                self.counted_ids.add(pid)

                updated_objects.append({
                    "id": pid,
                    "bbox": bbox,
                    "status": "New"
                })

        # تنظيف الأشخاص المختفين
        self._cleanup_disappeared(current_time)

        return updated_objects

    def _cleanup_disappeared(self, current_time):
        to_remove = []

        for pid, last in self.last_seen.items():
            if current_time - last > self.max_disappeared:
                to_remove.append(pid)

        for pid in to_remove:
            del self.objects[pid]
            del self.last_seen[pid]
            del self.frames_seen[pid]

    def _analyze_behavior(self, pid):
        frames = self.frames_seen.get(pid, 0)

        if frames > 150:
            return "Long Stay"
        elif frames > 30:
            return "Normal"
        else:
            return "Moving"