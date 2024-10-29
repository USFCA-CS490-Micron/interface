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

    async def sleep(self, deep_sleep=False):
        text = f"Entering {"deep" if deep_sleep else "light"} sleep.{"\nTap to wake up." if deep_sleep else ""}"
        await self.display.write_text(text=text, x=1, y=1, align=Alignment.MIDDLE_CENTER, color=PaletteColors.SEABLUE)
        await self.update_display()
        await asyncio.sleep(2)
        await self.frame.sleep(deep_sleep=deep_sleep)

    async def on_tap(self, callback):
        return await self.frame.motion.run_on_tap(callback=callback)

    async def wait_for_tap(self):
        await self.frame.motion.wait_for_tap()

    async def update_display(self):
        await self.display.show()

    async def wipe_display(self):
        await self.display.show_text(" ")

    async def write_splash(self):
        title = "CALDERA"
        subtitle = "Tap to speak, anytime."
        await self.display.write_text(text=title, x=1, y=1, align=Alignment.MIDDLE_CENTER, color=PaletteColors.WHITE)
        rect_width = int(DISP_MAX_W / 2)
        # fix rect placement (currently invisible)
        rect_x = int((DISP_MAX_W / 2) - (rect_width / 2))
        rect_y = int(DISP_MAX_H / 2) + self.display.get_text_height(title)
        await self.display.draw_rect_filled(x=rect_x, y=rect_y, w=rect_width, h=8, border_width=0,
                                            border_color=PaletteColors.NIGHTBLUE, fill_color=PaletteColors.SKYBLUE)
        await self.display.write_text(text=subtitle, x=1, y=128, align=Alignment.MIDDLE_CENTER,
                                      color=PaletteColors.WHITE)
        await self.update_display()
