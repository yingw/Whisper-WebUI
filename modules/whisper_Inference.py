import whisper
import gradio as gr
import time
import os
from typing import BinaryIO, Union, Tuple, List
import numpy as np
from datetime import datetime
import torch

from .base_interface import BaseInterface
from modules.subtitle_manager import get_srt, get_vtt, get_txt, write_file, safe_filename
from modules.youtube_manager import get_ytdata, get_ytaudio

DEFAULT_MODEL_SIZE = "large-v3"


class WhisperInference(BaseInterface):
    def __init__(self):
        super().__init__()
        self.current_model_size = None
        self.model = None
        self.available_models = whisper.available_models()
        self.available_langs = sorted(list(whisper.tokenizer.LANGUAGES.values()))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available_compute_types = ["float16", "float32"]
        self.current_compute_type = "float16" if self.device == "cuda" else "float32"
        self.default_beam_size = 1

    def transcribe_file(self,
                        fileobjs: list,
                        model_size: str,
                        lang: str,
                        file_format: str,
                        istranslate: bool,
                        add_timestamp: bool,
                        beam_size: int,
                        log_prob_threshold: float,
                        no_speech_threshold: float,
                        compute_type: str,
                        progress=gr.Progress()) -> list:
        """
        Write subtitle file from Files

        Parameters
        ----------
        fileobjs: list
            List of files to transcribe from gr.Files()
        model_size: str
            Whisper model size from gr.Dropdown()
        lang: str
            Source language of the file to transcribe from gr.Dropdown()
        file_format: str
            File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        istranslate: bool
            Boolean value from gr.Checkbox() that determines whether to translate to English.
            It's Whisper's feature to translate speech from another language directly into English end-to-end.
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        beam_size: int
            Int value from gr.Number() that is used for decoding option.
        log_prob_threshold: float
            float value from gr.Number(). If the average log probability over sampled tokens is
            below this value, treat as failed.
        no_speech_threshold: float
            float value from gr.Number(). If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
            consider the segment as silent.
        compute_type: str
            compute type from gr.Dropdown().
        progress: gr.Progress
            Indicator to show progress directly in gradio.
            I use a forked version of whisper for this. To see more info : https://github.com/jhj0517/jhj0517-whisper/tree/add-progress-callback

        Returns
        ----------
        A List of
        String to return to gr.Textbox()
        Files to return to gr.Files()
        """
        try:
            self.update_model_if_needed(model_size=model_size, compute_type=compute_type, progress=progress)

            files_info = {}
            for fileobj in fileobjs:
                progress(0, desc="正在加载音频……")
                audio = whisper.load_audio(fileobj.name)

                result, elapsed_time = self.transcribe(audio=audio,
                                                       lang=lang,
                                                       istranslate=istranslate,
                                                       beam_size=beam_size,
                                                       log_prob_threshold=log_prob_threshold,
                                                       no_speech_threshold=no_speech_threshold,
                                                       compute_type=compute_type,
                                                       progress=progress
                                                       )
                progress(1, desc="完成！")

                file_name, file_ext = os.path.splitext(os.path.basename(fileobj.name))
                file_name = safe_filename(file_name)
                subtitle, file_path = self.generate_and_write_file(
                    file_name=file_name,
                    transcribed_segments=result,
                    add_timestamp=add_timestamp,
                    file_format=file_format
                )
                files_info[file_name] = {"subtitle": subtitle, "elapsed_time": elapsed_time, "path":  file_path}

            total_result = ''
            total_time = 0
            for file_name, info in files_info.items():
                total_result += '------------------------------------\n'
                total_result += f'{file_name}\n\n'
                total_result += f"{info['subtitle']}"
                total_time += info["elapsed_time"]

            gr_str = f"Done in {self.format_time(total_time)}! Subtitle is in the outputs folder.\n\n{total_result}"
            gr_file_path = [info['path'] for info in files_info.values()]

            return [gr_str, gr_file_path]
        except Exception as e:
            print(f"Error transcribing file: {str(e)}")
        finally:
            self.release_cuda_memory()
            self.remove_input_files([fileobj.name for fileobj in fileobjs])

    def transcribe_youtube(self,
                           youtubelink: str,
                           model_size: str,
                           lang: str,
                           file_format: str,
                           istranslate: bool,
                           add_timestamp: bool,
                           beam_size: int,
                           log_prob_threshold: float,
                           no_speech_threshold: float,
                           compute_type: str,
                           progress=gr.Progress()) -> list:
        """
        Write subtitle file from Youtube

        Parameters
        ----------
        youtubelink: str
            Link of Youtube to transcribe from gr.Textbox()
        model_size: str
            Whisper model size from gr.Dropdown()
        lang: str
            Source language of the file to transcribe from gr.Dropdown()
        file_format: str
            File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        istranslate: bool
            Boolean value from gr.Checkbox() that determines whether to translate to English.
            It's Whisper's feature to translate speech from another language directly into English end-to-end.
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        beam_size: int
            Int value from gr.Number() that is used for decoding option.
        log_prob_threshold: float
            float value from gr.Number(). If the average log probability over sampled tokens is
            below this value, treat as failed.
        no_speech_threshold: float
            float value from gr.Number(). If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
            consider the segment as silent.
        compute_type: str
            compute type from gr.Dropdown().
        progress: gr.Progress
            Indicator to show progress directly in gradio.
            I use a forked version of whisper for this. To see more info : https://github.com/jhj0517/jhj0517-whisper/tree/add-progress-callback

        Returns
        ----------
        A List of
        String to return to gr.Textbox()
        Files to return to gr.Files()
        """
        try:
            self.update_model_if_needed(model_size=model_size, compute_type=compute_type, progress=progress)

            progress(0, desc="正在从Youtube加载音频……")
            yt = get_ytdata(youtubelink)
            audio = whisper.load_audio(get_ytaudio(yt))

            result, elapsed_time = self.transcribe(audio=audio,
                                                   lang=lang,
                                                   istranslate=istranslate,
                                                   beam_size=beam_size,
                                                   log_prob_threshold=log_prob_threshold,
                                                   no_speech_threshold=no_speech_threshold,
                                                   compute_type=compute_type,
                                                   progress=progress)
            progress(1, desc="完成！")

            file_name = safe_filename(yt.title)
            subtitle, file_path = self.generate_and_write_file(
                file_name=file_name,
                transcribed_segments=result,
                add_timestamp=add_timestamp,
                file_format=file_format
            )

            gr_str = f"Done in {self.format_time(elapsed_time)}! Subtitle file is in the outputs folder.\n\n{subtitle}"
            return [gr_str, file_path]
        except Exception as e:
            print(f"转录Youtube视频出错: {str(e)}")
        finally:
            try:
                if 'yt' not in locals():
                    yt = get_ytdata(youtubelink)
                    file_path = get_ytaudio(yt)
                else:
                    file_path = get_ytaudio(yt)

                self.release_cuda_memory()
                self.remove_input_files([file_path])
            except Exception as cleanup_error:
                pass

    def transcribe_mic(self,
                       micaudio: str,
                       model_size: str,
                       lang: str,
                       file_format: str,
                       istranslate: bool,
                       beam_size: int,
                       log_prob_threshold: float,
                       no_speech_threshold: float,
                       compute_type: str,
                       progress=gr.Progress()) -> list:
        """
        Write subtitle file from microphone

        Parameters
        ----------
        micaudio: str
            Audio file path from gr.Microphone()
        model_size: str
            Whisper model size from gr.Dropdown()
        lang: str
            Source language of the file to transcribe from gr.Dropdown()
        file_format: str
            Subtitle format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        istranslate: bool
            Boolean value from gr.Checkbox() that determines whether to translate to English.
            It's Whisper's feature to translate speech from another language directly into English end-to-end.
        beam_size: int
            Int value from gr.Number() that is used for decoding option.
        log_prob_threshold: float
            float value from gr.Number(). If the average log probability over sampled tokens is
            below this value, treat as failed.
        no_speech_threshold: float
            float value from gr.Number(). If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
            consider the segment as silent.
        compute_type: str
            compute type from gr.Dropdown().
        progress: gr.Progress
            Indicator to show progress directly in gradio.
            I use a forked version of whisper for this. To see more info : https://github.com/jhj0517/jhj0517-whisper/tree/add-progress-callback

        Returns
        ----------
        A List of
        String to return to gr.Textbox()
        Files to return to gr.Files()
        """
        try:
            self.update_model_if_needed(model_size=model_size, compute_type=compute_type, progress=progress)

            result, elapsed_time = self.transcribe(audio=micaudio,
                                                   lang=lang,
                                                   istranslate=istranslate,
                                                   beam_size=beam_size,
                                                   log_prob_threshold=log_prob_threshold,
                                                   no_speech_threshold=no_speech_threshold,
                                                   compute_type=compute_type,
                                                   progress=progress)
            progress(1, desc="完成！")

            subtitle, file_path = self.generate_and_write_file(
                file_name="Mic",
                transcribed_segments=result,
                add_timestamp=True,
                file_format=file_format
            )

            gr_str = f"Done in {self.format_time(elapsed_time)}! Subtitle file is in the outputs folder.\n\n{subtitle}"
            return [gr_str, file_path]
        except Exception as e:
            print(f"转录麦克风出错: {str(e)}")
        finally:
            self.release_cuda_memory()
            self.remove_input_files([micaudio])

    def transcribe(self,
                   audio: Union[str, np.ndarray, torch.Tensor],
                   lang: str,
                   istranslate: bool,
                   beam_size: int,
                   log_prob_threshold: float,
                   no_speech_threshold: float,
                   compute_type: str,
                   progress: gr.Progress
                   ) -> Tuple[List[dict], float]:
        """
        transcribe method for OpenAI's Whisper implementation.

        Parameters
        ----------
        audio: Union[str, BinaryIO, torch.Tensor]
            Audio path or file binary or Audio numpy array
        lang: str
            Source language of the file to transcribe from gr.Dropdown()
        istranslate: bool
            Boolean value from gr.Checkbox() that determines whether to translate to English.
            It's Whisper's feature to translate speech from another language directly into English end-to-end.
        beam_size: int
            Int value from gr.Number() that is used for decoding option.
        log_prob_threshold: float
            float value from gr.Number(). If the average log probability over sampled tokens is
            below this value, treat as failed.
        no_speech_threshold: float
            float value from gr.Number(). If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
            consider the segment as silent.
        compute_type: str
            compute type from gr.Dropdown().
        progress: gr.Progress
            Indicator to show progress directly in gradio.

        Returns
        ----------
        segments_result: List[dict]
            list of dicts that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for transcription
        """
        start_time = time.time()

        def progress_callback(progress_value):
            progress(progress_value, desc="正在转录……")

        if lang == "Automatic Detection":
            lang = None

        translatable_model = ["large", "large-v1", "large-v2", "large-v3"]
        segments_result = self.model.transcribe(audio=audio,
                                                language=lang,
                                                verbose=False,
                                                beam_size=beam_size,
                                                logprob_threshold=log_prob_threshold,
                                                no_speech_threshold=no_speech_threshold,
                                                task="translate" if istranslate and self.current_model_size in translatable_model else "transcribe",
                                                fp16=True if compute_type == "float16" else False,
                                                progress_callback=progress_callback)["segments"]
        elapsed_time = time.time() - start_time

        return segments_result, elapsed_time

    def update_model_if_needed(self,
                               model_size: str,
                               compute_type: str,
                               progress: gr.Progress,
                               ):
        """
        Initialize model if it doesn't match with current model setting
        """
        if compute_type != self.current_compute_type:
            self.current_compute_type = compute_type
        if model_size != self.current_model_size or self.model is None:
            progress(0, desc="正在初始化模型……")
            self.current_model_size = model_size
            self.model = whisper.load_model(
                name=model_size,
                device=self.device,
                download_root=os.path.join("models", "Whisper")
            )

    @staticmethod
    def generate_and_write_file(file_name: str,
                                transcribed_segments: list,
                                add_timestamp: bool,
                                file_format: str,
                                ) -> str:
        """
        This method writes subtitle file and returns str to gr.Textbox
        """
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        if add_timestamp:
            output_path = os.path.join("outputs", f"{file_name}-{timestamp}")
        else:
            output_path = os.path.join("outputs", f"{file_name}")

        if file_format == "SRT":
            content = get_srt(transcribed_segments)
            output_path += '.srt'
            write_file(content, output_path)

        elif file_format == "WebVTT":
            content = get_vtt(transcribed_segments)
            output_path += '.vtt'
            write_file(content, output_path)

        elif file_format == "txt":
            content = get_txt(transcribed_segments)
            output_path += '.txt'
            write_file(content, output_path)
        return content, output_path

    @staticmethod
    def format_time(elapsed_time: float) -> str:
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        time_str = ""
        if hours:
            time_str += f"{hours} hours "
        if minutes:
            time_str += f"{minutes} minutes "
        seconds = round(seconds)
        time_str += f"{seconds} seconds"

        return time_str.strip()
