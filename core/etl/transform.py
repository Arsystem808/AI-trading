# core/etl/transform.py
class IncrementalState:
    def __init__(self): self.cache = {}

    def roll_update(self, name, series, window):
        from collections import deque
        buf = self.cache.get(name, deque(maxlen=window))
        if isinstance(series, float): 
            buf.append(series)
        else:
            for v in series: buf.append(v)
        self.cache[name] = buf
        return sum(buf)/len(buf) if buf else None
