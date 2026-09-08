"""Backend-independent reference layout. No tensor, model or GPU dependency."""
from dataclasses import dataclass
from collections import deque

@dataclass(frozen=True)
class Layout:
    context_tokens: int = 1280
    visual_tokens: int = 80
    action_tokens: int = 13
    @property
    def block_tokens(self):
        return self.visual_tokens + self.action_tokens

class ReCAPContext:
    """Frozen context + one anchor + at most W generated visual/action blocks.

    A returned prompt must be prefetched with rebased positions whenever eviction
    occurs. Slicing a raw KV tensor without repairing positions is not equivalent.
    """
    def __init__(self, context, anchor, window=6, layout=Layout()):
        if type(window) is not int or window < 0:
            raise ValueError('window must be a non-negative integer')
        self.layout = layout
        self.context = tuple(context)
        self.anchor = tuple(anchor)
        if len(self.context) != layout.context_tokens or len(self.anchor) != layout.block_tokens:
            raise ValueError('invalid context or anchor length')
        self.recent = deque(maxlen=window)
        self.window = window
        self.evictions = 0
    def append(self, visual, action):
        if len(visual) != self.layout.visual_tokens or len(action) != self.layout.action_tokens:
            raise ValueError('a complete visual/action block is required')
        evicted = len(self.recent) == self.window
        self.recent.append(tuple(visual) + tuple(action))
        self.evictions += int(evicted)
        return evicted
    def prompt(self):
        return list(self.context + self.anchor + tuple(t for block in self.recent for t in block))
