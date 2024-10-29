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

    async def write_title(self, title: str):
        await self.display.draw_rect_filled(x=1, y=56, w=int(DISP_MAX_W * 0.75), h=8, border_width=0,
                                            border_color=PaletteColors.NIGHTBLUE, fill_color=PaletteColors.NIGHTBLUE)
        await self.display.write_text(text=title, x=1, y=1, align=Alignment.TOP_LEFT, color=PaletteColors.CLOUDBLUE)

    async def write_content(self, content: str):
        await self.display.scroll_text(text=content, lines_per_frame=5, delay=0.12, color=PaletteColors.WHITE)

    async def write_loading(self):
        try:
            dots = 0
            while True:
                print("loading")
                text = "Loading" + "." * dots
                await self.display.show_text(text, align=Alignment.TOP_CENTER, color=PaletteColors.RED)
                dots = dots + 1 if dots < 3 else 0
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            await self.wipe_display()

    async def listen(self):
        text = "Listening..."
        await self.display.write_text(text, x=50, y=50, align=Alignment.TOP_CENTER, color=PaletteColors.RED)
        await self.update_display()

        audio_arr = await self.frame.microphone.record_audio(silence_cutoff_length_in_seconds=3,
                                                             max_length_in_seconds=30)

        await self.wipe_display()
        return audio_arr

    @staticmethod
    def preprocess_audio(audio_arr, sample_rate, target_sr=16000):
        if sample_rate != target_sr:
            audio_arr = librosa.resample(audio_arr.astype(float), orig_sr=sample_rate, target_sr=target_sr)
        audio_arr = audio_arr / np.max(np.abs(audio_arr))
        return audio_arr.astype(np.float32)

    async def analyze(self, audio_arr: np.ndarray):
        loading_task = asyncio.create_task(self.write_loading())

        try:
            audio_arr = self.preprocess_audio(audio_arr, self.sample_rate)
            transcription = self.whisper.transcribe(audio_arr)
            print(f"Transcription: {transcription["text"]}")
        finally:
            loading_task.cancel()
            with suppress(asyncio.CancelledError):
                await loading_task

    async def run(self):
        try:
            await self.write_splash()
            await asyncio.sleep(2.5)
            await self.wipe_display()
            while True:
                await self.wait_for_tap()
                audio_arr = await self.listen()
                await self.analyze(audio_arr)
        finally:
            await self.sleep(deep_sleep=True)


async def main():
    interface = Interface(None)

    event_loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def handle_interrupt(signum, frame):
        event_loop.remove_signal_handler(signal.SIGINT)
        print("Shutting down.")
        stop_event.set()

    event_loop.add_signal_handler(signal.SIGINT, handle_interrupt, signal.SIGINT, None)

    async with Frame() as frame:
        interface.set_frame(frame=frame)

        try:
            run_task = asyncio.create_task(interface.run())
            stop_task = asyncio.create_task(stop_event.wait())

            await asyncio.wait(
                {run_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            exit(0)


if __name__ == '__main__':
    asyncio.run(main())



