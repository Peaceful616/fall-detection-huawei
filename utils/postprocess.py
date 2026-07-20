"""规则先验 + 学习融合后处理

学习分支：跌倒概率滑动窗口均值 + 方差 + 持续帧数
规则分支：人体框 y 速度 + 横纵比突变
融合：加权 OR + 冷却 + 时序 NMS + 连续确认
"""
from collections import deque
import numpy as np


class Box:
    """YOLO 检测的人体框"""

    def __init__(self, x1, y1, x2, y2, score=1.0):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.score = score
        self.cx = (x1 + x2) / 2
        self.cy = (y1 + y2) / 2
        self.w = x2 - x1
        self.h = y2 - y1

    def aspect_ratio(self):
        return self.w / max(self.h, 1e-6)


class FallAlarmPostprocess:
    """跌倒报警后处理

    使用流程：
        for frame in video:
            p_fall = student_model(frame)  # 学习分支输出
            boxes = yolo_detect(frame)     # 规则分支输入
            alarm = postprocess.update(p_fall, boxes)
            if alarm:
                trigger_alarm()
    """

    def __init__(self,
                 window_size: int = 16,         # 滑动窗口
                 learned_thresh: float = 0.6,   # 学习分支阈值
                 learned_var_thresh: float = 0.1,  # 方差上限（稳定性）
                 learned_min_frames: int = 3,   # 持续帧数
                 rule_v_thresh: float = 15.0,   # y 速度阈值（像素/帧）
                 rule_aspect_change: float = 0.5,  # 横纵比变化阈值
                 fusion_thresh: float = 0.6,    # 融合阈值
                 fusion_w_learned: float = 0.7,
                 fusion_w_rule: float = 0.3,
                 cooldown_frames: int = 30 * 30,  # 冷却（30s @ 30fps）
                 confirm_frames: int = 3):      # 连续确认帧数
        self.window_size = window_size
        self.learned_thresh = learned_thresh
        self.learned_var_thresh = learned_var_thresh
        self.learned_min_frames = learned_min_frames
        self.rule_v_thresh = rule_v_thresh
        self.rule_aspect_change = rule_aspect_change
        self.fusion_thresh = fusion_thresh
        self.fusion_w_learned = fusion_w_learned
        self.fusion_w_rule = fusion_w_rule
        self.cooldown_frames = cooldown_frames
        self.confirm_frames = confirm_frames

        # 状态
        self.p_fall_history = deque(maxlen=window_size)
        self.box_history = deque(maxlen=window_size)
        self.last_alarm_frame = -cooldown_frames  # 上次报警帧
        self.confirm_count = 0

    def reset(self):
        self.p_fall_history.clear()
        self.box_history.clear()
        self.last_alarm_frame = -self.cooldown_frames
        self.confirm_count = 0

    def update(self, p_fall: float, boxes: list, frame_idx: int) -> bool:
        """处理一帧

        p_fall: 学生模型输出的跌倒概率 [0, 1]
        boxes: list[Box]
        frame_idx: 当前帧序号
        return: 是否触发报警
        """
        self.p_fall_history.append(p_fall)
        self.box_history.append(boxes[0] if boxes else None)

        # 学习分支：滑动窗口均值
        if len(self.p_fall_history) < self.learned_min_frames:
            p_learn = 0.0
        else:
            arr = np.array(self.p_fall_history)
            p_mean = arr.mean()
            p_var = arr.var()
            # 持续帧数
            above = sum(1 for p in list(self.p_fall_history)[-self.learned_min_frames:]
                        if p > self.learned_thresh)
            if above >= self.learned_min_frames and p_var < self.learned_var_thresh:
                p_learn = p_mean
            else:
                p_learn = 0.0

        # 规则分支：y 速度 + 横纵比变化
        p_rule = 0.0
        history = list(self.box_history)
        if len(history) >= 6 and history[-1] is not None and history[-4] is not None:
            cur = history[-1]
            prev = history[-4]
            dy = cur.cy - prev.cy
            cur_ar = cur.aspect_ratio()
            prev_ar = history[-6].aspect_ratio() if history[-6] is not None else cur_ar
            ar_change = abs(cur_ar - prev_ar)
            if dy > self.rule_v_thresh and ar_change > self.rule_aspect_change:
                p_rule = 1.0

        # 融合
        p_final = self.fusion_w_learned * p_learn + self.fusion_w_rule * p_rule

        # 确认 + 冷却
        if p_final > self.fusion_thresh:
            self.confirm_count += 1
            if self.confirm_count >= self.confirm_frames and \
               frame_idx - self.last_alarm_frame > self.cooldown_frames:
                self.last_alarm_frame = frame_idx
                self.confirm_count = 0
                return True
        else:
            self.confirm_count = max(0, self.confirm_count - 1)
        return False

    def get_fall_score_series(self) -> np.ndarray:
        """获取跌倒概率时序曲线（用于可视化）"""
        return np.array(self.p_fall_history)
