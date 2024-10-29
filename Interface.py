import asyncio
import signal
from contextlib import suppress

import whisper
from whisper import Whisper
import numpy as np
import librosa
from frame_sdk import Frame
from frame_sdk.display import Alignment, PaletteColors, Display
from typing import Optional

DISP_MAX_W = 640  #px
DISP_MAX_H = 400  #px


class Interface:
    def __init__(self, frame: Optional[Frame]):
        self.frame: Frame = frame
        self.display: Display = frame.display if frame is not None else None
        self.sample_rate: int = frame.microphone.sample_rate if frame is not None else None
        self.whisper: Whisper = whisper.load_model("base")

    def set_frame(self, frame: Frame):
        if frame is not None:
            self.frame = frame
            self.display = self.frame.display
            self.sample_rate = self.frame.microphone.sample_rate

    async def get_battery_level(self):
        print(f"Frame battery level: {await self.frame.get_battery_level()}%")

    async def break_script(self):
        await self.frame.bluetooth.send_break_signal()