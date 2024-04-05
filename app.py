import gradio as gr
import os
import argparse

from modules.whisper_Inference import WhisperInference
from modules.faster_whisper_inference import FasterWhisperInference
from modules.nllb_inference import NLLBInference
from ui.htmls import *
from modules.youtube_manager import get_ytmetas
from modules.deepl_api import DeepLAPI

class App:
    def __init__(self, args):
        self.args = args
        self.app = gr.Blocks(css=CSS, theme=self.args.theme)
        self.whisper_inf = WhisperInference() if self.args.disable_faster_whisper else FasterWhisperInference()
        if isinstance(self.whisper_inf, FasterWhisperInference):
            print("Use Faster Whisper implementation")
        else:
            print("Use Open AI Whisper implementation")
        print(f"Device \"{self.whisper_inf.device}\" is detected")
        self.nllb_inf = NLLBInference()
        self.deepl_api = DeepLAPI()

    @staticmethod
    def open_folder(folder_path: str):
        if os.path.exists(folder_path):
            os.system(f"start {folder_path}")
        else:
            print(f"The folder {folder_path} does not exist.")

    @staticmethod
    def on_change_models(model_size: str):
        translatable_model = ["large", "large-v1", "large-v2", "large-v3"]
        if model_size not in translatable_model:
            return gr.Checkbox(visible=False, value=False, interactive=False)
        else:
            return gr.Checkbox(visible=True, value=False, label="Translate to English?", interactive=True)

    def launch(self):
        with self.app:
            with gr.Row():
                with gr.Column():
                    gr.Markdown(MARKDOWN, elem_id="md_project")
            with gr.Tabs():
                with gr.TabItem("文件"):  # tab1
                    with gr.Row():
                        input_file = gr.Files(type="filepath", label="在这里上传文件")
                    with gr.Row():
                        dd_model = gr.Dropdown(choices=self.whisper_inf.available_models, value="large-v3",
                                               label="模型")
                        dd_lang = gr.Dropdown(choices=["自动检测"] + self.whisper_inf.available_langs,
                                              value="自动检测", label="语言")
                        dd_file_format = gr.Dropdown(["SRT", "WebVTT", "txt"], value="SRT", label="文件格式")
                    with gr.Row():
                        cb_translate = gr.Checkbox(value=False, label="翻译成英语？", interactive=True)
                    with gr.Row():
                        cb_timestamp = gr.Checkbox(value=True, label="在文件名末尾添加时间戳", interactive=True)
                    with gr.Accordion("高级参数", open=False):
                        nb_beam_size = gr.Number(label="Beam大小", value=1, precision=0, interactive=True)
                        nb_log_prob_threshold = gr.Number(label="对数概率阈值", value=-1.0, interactive=True)
                        nb_no_speech_threshold = gr.Number(label="无语音阈值", value=0.6, interactive=True)
                        dd_compute_type = gr.Dropdown(label="计算类型", choices=self.whisper_inf.available_compute_types, value=self.whisper_inf.current_compute_type, interactive=True)
                    with gr.Row():
                        btn_run = gr.Button("生成字幕文件", variant="primary")
                    with gr.Row():
                        tb_indicator = gr.Textbox(label="输出", scale=4)
                        files_subtitles = gr.Files(label="下载输出文件", scale=4, interactive=False)
                        btn_openfolder = gr.Button('📂', scale=1)

                    params = [input_file, dd_model, dd_lang, dd_file_format, cb_translate, cb_timestamp]
                    advanced_params = [nb_beam_size, nb_log_prob_threshold, nb_no_speech_threshold, dd_compute_type]
                    btn_run.click(fn=self.whisper_inf.transcribe_file,
                                  inputs=params + advanced_params,
                                  outputs=[tb_indicator, files_subtitles])
                    btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)
                    dd_model.change(fn=self.on_change_models, inputs=[dd_model], outputs=[cb_translate])

                with gr.TabItem("Youtube"):  # tab2
                    with gr.Row():
                        tb_youtubelink = gr.Textbox(label="Youtube链接")
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            img_thumbnail = gr.Image(label="Youtube缩略图")
                        with gr.Column():
                            tb_title = gr.Label(label="Youtube标题")
                            tb_description = gr.Textbox(label="Youtube描述", max_lines=15)
                    with gr.Row():
                        dd_model = gr.Dropdown(choices=self.whisper_inf.available_models, value="large-v3",
                                               label="模型")
                        dd_lang = gr.Dropdown(choices=["自动检测"] + self.whisper_inf.available_langs,
                                              value="自动检测", label="语言")
                        dd_file_format = gr.Dropdown(choices=["SRT", "WebVTT", "txt"], value="SRT", label="文件格式")
                    with gr.Row():
                        cb_translate = gr.Checkbox(value=False, label="翻译成英语？", interactive=True)
                    with gr.Row():
                        cb_timestamp = gr.Checkbox(value=True, label="在文件名末尾添加时间戳",
                                                   interactive=True)
                    with gr.Accordion("高级参数", open=False):
                        nb_beam_size = gr.Number(label="Beam大小", value=1, precision=0, interactive=True)
                        nb_log_prob_threshold = gr.Number(label="对数概率阈值", value=-1.0, interactive=True)
                        nb_no_speech_threshold = gr.Number(label="无语音阈值", value=0.6, interactive=True)
                        dd_compute_type = gr.Dropdown(label="计算类型", choices=self.whisper_inf.available_compute_types, value=self.whisper_inf.current_compute_type, interactive=True)
                    with gr.Row():
                        btn_run = gr.Button("生成字幕文件", variant="primary")
                    with gr.Row():
                        tb_indicator = gr.Textbox(label="输出", scale=4)
                        files_subtitles = gr.Files(label="下载输出文件", scale=4)
                        btn_openfolder = gr.Button('📂', scale=1)

                    params = [tb_youtubelink, dd_model, dd_lang, dd_file_format, cb_translate, cb_timestamp]
                    advanced_params = [nb_beam_size, nb_log_prob_threshold, nb_no_speech_threshold, dd_compute_type]
                    btn_run.click(fn=self.whisper_inf.transcribe_youtube,
                                  inputs=params + advanced_params,
                                  outputs=[tb_indicator, files_subtitles])
                    tb_youtubelink.change(get_ytmetas, inputs=[tb_youtubelink],
                                          outputs=[img_thumbnail, tb_title, tb_description])
                    btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)
                    dd_model.change(fn=self.on_change_models, inputs=[dd_model], outputs=[cb_translate])

                with gr.TabItem("麦克风"):  # tab3
                    with gr.Row():
                        mic_input = gr.Microphone(label="用麦克风录音", type="filepath", interactive=True)
                    with gr.Row():
                        dd_model = gr.Dropdown(choices=self.whisper_inf.available_models, value="large-v3",
                                               label="模型")
                        dd_lang = gr.Dropdown(choices=["自动检测"] + self.whisper_inf.available_langs,
                                              value="自动检测", label="语言")
                        dd_file_format = gr.Dropdown(["SRT", "WebVTT", "txt"], value="SRT", label="文件格式")
                    with gr.Row():
                        cb_translate = gr.Checkbox(value=False, label="翻译成英语？", interactive=True)
                    with gr.Accordion("高级参数", open=False):
                        nb_beam_size = gr.Number(label="Beam大小", value=1, precision=0, interactive=True)
                        nb_log_prob_threshold = gr.Number(label="对数概率阈值", value=-1.0, interactive=True)
                        nb_no_speech_threshold = gr.Number(label="无语音阈值", value=0.6, interactive=True)
                        dd_compute_type = gr.Dropdown(label="计算类型", choices=self.whisper_inf.available_compute_types, value=self.whisper_inf.current_compute_type, interactive=True)
                    with gr.Row():
                        btn_run = gr.Button("生成字幕文件", variant="primary")
                    with gr.Row():
                        tb_indicator = gr.Textbox(label="输出", scale=4)
                        files_subtitles = gr.Files(label="下载输出文件", scale=4)
                        btn_openfolder = gr.Button('📂', scale=1)

                    params = [mic_input, dd_model, dd_lang, dd_file_format, cb_translate]
                    advanced_params = [nb_beam_size, nb_log_prob_threshold, nb_no_speech_threshold, dd_compute_type]
                    btn_run.click(fn=self.whisper_inf.transcribe_mic,
                                  inputs=params + advanced_params,
                                  outputs=[tb_indicator, files_subtitles])
                    btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)
                    dd_model.change(fn=self.on_change_models, inputs=[dd_model], outputs=[cb_translate])

                with gr.TabItem("文本翻译"):  # tab 4
                    with gr.Row():
                        file_subs = gr.Files(type="filepath", label="在这里上传字幕文件以进行翻译",
                                             file_types=['.vtt', '.srt'])

                    with gr.TabItem("DeepL API"):  # sub tab1
                        with gr.Row():
                            tb_authkey = gr.Textbox(label="你的Auth Key (API KEY)",
                                                    value="")
                        with gr.Row():
                            dd_deepl_sourcelang = gr.Dropdown(label="源语言", value="Automatic Detection",
                                                              choices=list(
                                                                  self.deepl_api.available_source_langs.keys()))
                            dd_deepl_targetlang = gr.Dropdown(label="目标语言", value="English",
                                                              choices=list(
                                                                  self.deepl_api.available_target_langs.keys()))
                        with gr.Row():
                            cb_deepl_ispro = gr.Checkbox(label="专业版用户?", value=False)
                        with gr.Row():
                            btn_run = gr.Button("翻译字幕文件", variant="primary")
                        with gr.Row():
                            tb_indicator = gr.Textbox(label="输出", scale=4)
                            files_subtitles = gr.Files(label="下载输出文件", scale=4)
                            btn_openfolder = gr.Button('📂', scale=1)

                    btn_run.click(fn=self.deepl_api.translate_deepl,
                                  inputs=[tb_authkey, file_subs, dd_deepl_sourcelang, dd_deepl_targetlang,
                                          cb_deepl_ispro],
                                  outputs=[tb_indicator, files_subtitles])

                    btn_openfolder.click(fn=lambda: self.open_folder(os.path.join("outputs", "translations")),
                                         inputs=None,
                                         outputs=None)

                    with gr.TabItem("NLLB"):  # sub tab2
                        with gr.Row():
                            dd_nllb_model = gr.Dropdown(label="模型", value=self.nllb_inf.default_model_size,
                                                        choices=self.nllb_inf.available_models)
                            dd_nllb_sourcelang = gr.Dropdown(label="源语言",
                                                             choices=self.nllb_inf.available_source_langs)
                            dd_nllb_targetlang = gr.Dropdown(label="目标语言",
                                                             choices=self.nllb_inf.available_target_langs)
                        with gr.Row():
                            cb_timestamp = gr.Checkbox(value=True, label="在文件名末尾添加时间戳",
                                                       interactive=True)
                        with gr.Row():
                            btn_run = gr.Button("翻译字幕文件", variant="primary")
                        with gr.Row():
                            tb_indicator = gr.Textbox(label="输出", scale=4)
                            files_subtitles = gr.Files(label="下载输出文件", scale=4)
                            btn_openfolder = gr.Button('📂', scale=1)
                        with gr.Column():
                            md_vram_table = gr.HTML(NLLB_VRAM_TABLE, elem_id="md_nllb_vram_table")

                    btn_run.click(fn=self.nllb_inf.translate_file,
                                  inputs=[file_subs, dd_nllb_model, dd_nllb_sourcelang, dd_nllb_targetlang, cb_timestamp],
                                  outputs=[tb_indicator, files_subtitles])

                    btn_openfolder.click(fn=lambda: self.open_folder(os.path.join("outputs", "translations")),
                                         inputs=None,
                                         outputs=None)

        # Launch the app with optional gradio settings
        launch_args = {}
        if self.args.share:
            launch_args['share'] = self.args.share
        if self.args.server_name:
            launch_args['server_name'] = self.args.server_name
        if self.args.server_port:
            launch_args['server_port'] = self.args.server_port
        if self.args.username and self.args.password:
            launch_args['auth'] = (self.args.username, self.args.password)
        self.app.queue(api_open=False).launch(**launch_args)


# Create the parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--disable_faster_whisper', type=bool, default=False, nargs='?', const=True, help='禁用faster_whisper实现。faster_whipser：https://github.com/guillaumekln/faster-whisper')
parser.add_argument('--share', type=bool, default=False, nargs='?', const=True, help='Gradio共享值')
parser.add_argument('--server_name', type=str, default=None, help='Gradio服务器主机')
parser.add_argument('--server_port', type=int, default=None, help='Gradio服务器端口')
parser.add_argument('--username', type=str, default=None, help='Gradio认证用户名')
parser.add_argument('--password', type=str, default=None, help='Gradio认证密码')
parser.add_argument('--theme', type=str, default=None, help='Gradio Blocks主题')
parser.add_argument('--colab', type=bool, default=False, nargs='?', const=True, help='是否为colab用户')
_args = parser.parse_args()

if __name__ == "__main__":
    app = App(args=_args)
    app.launch()
