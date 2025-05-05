#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import os
from typing import List

from deepgram.clients.live import LiveOptions
from dotenv import load_dotenv
from gst import GStreamerPipelinePlayer, PlayPipelineFrame, PlayPipelineFrameEnd
from loguru import logger
from openai.types.chat import ChatCompletionSystemMessageParam

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMTextFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.processors.gstreamer.pipeline_source import GStreamerPipelineSource
from pipecat.processors.producer_processor import ProducerProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

load_dotenv(override=True)

video_path = os.path.abspath(
    "/home/ken-kuro/PycharmProjects/pipecat/examples/p2p-webrtc/video-transform/server/how-are-you-today.mp4"
)  # Ensure path is absolute

# TODO: This is for testing purposes only. Use a real state management system in production.
is_playing = False


class ResetPlayingHandler(FrameProcessor):
    def __init__(self):
        super().__init__()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, PlayPipelineFrameEnd) or isinstance(frame, TTSStoppedFrame):
            global is_playing
            is_playing = False
            logger.debug("ResetPlayingHandler triggered, is_playing set to False")

        # Always pass frames through to maintain original flow
        await self.push_frame(frame, direction)


class SimplePipelineTrigger(FrameProcessor):
    def __init__(self):
        super().__init__()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # When the LLM finishes its response, check for trigger phrase
        if isinstance(frame, LLMTextFrame):
            if "kendrick" in frame.text.lower() or "ken" in frame.text.lower():
                if not os.path.exists(video_path):
                    logger.error(f"Video file not found: {video_path}")
                    await self.push_frame(ErrorFrame(f"Video file not found: {video_path}"))
                    return  # Don't trigger if file missing

                pipeline_desc = f'filesrc location="{video_path}"'

                logger.debug(f"Pushing PlayPipelineFrame with: {pipeline_desc[:200]}...")
                await self.push_frame(PlayPipelineFrame(pipeline_description=pipeline_desc))
                global is_playing
                is_playing = True
                logger.debug("SimplePipelineTrigger triggered, is_playing set to True")

        # Always pass frames through to maintain original flow
        await self.push_frame(frame, direction)


SYSTEM_INSTRUCTION = f"""
"You are Gemini Chatbot, a friendly, helpful robot.

Your goal is to demonstrate your capabilities in a succinct way.

Your output will be converted to audio so don't include special characters in your answers.

Respond to what the user said in a creative and helpful way. Keep your responses brief. One or two sentences at most.
"""


async def run_bot(webrtc_connection):
    transport_params = TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_out_10ms_chunks=2,
        video_in_enabled=True,
        video_out_enabled=True,
        video_out_width=736,
        video_out_height=1310,
        video_out_is_live=True,
        vad_analyzer=SileroVADAnalyzer(),
    )

    pipecat_transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection, params=transport_params
    )

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            model="nova-3",
        ),
    )

    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))

    # tts = OpenAITTSService(base_url="https://dev-vh-voice-center.vuihoc.vn/api/v2", api_key="5b62ef4f-87d9-4928-865a-8d4c70e0bbf9", sample_rate=24000)
    tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    gst = GStreamerPipelinePlayer()

    messages: List[ChatCompletionSystemMessageParam] = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Create a simple pipeline with the normal flow plus a trigger
    trigger = SimplePipelineTrigger()

    reset_handler = ResetPlayingHandler()

    async def is_play_pipeline(frame: Frame):
        logger.debug(
            f"Checking if frame is PlayPipelineFrame: {frame}, current is_playing: {is_playing}"
        )
        return isinstance(frame, PlayPipelineFrame)

    async def is_not_playing(frame: Frame):
        logger.debug(
            f"Checking if the current status not playing, current is_playing: {is_playing}"
        )
        return not is_playing

    pipeline = Pipeline(
        [
            pipecat_transport.input(),
            rtvi,
            stt,
            context_aggregator.user(),
            llm,
            # Just for testing, remove in production
            trigger,
            ParallelPipeline(
                [
                    FunctionFilter(is_play_pipeline),
                    gst,
                ],
                [FunctionFilter(is_not_playing), tts],
            ),
            reset_handler,
            context_aggregator.assistant(),
            pipecat_transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        observers=[RTVIObserver(rtvi)],
        params=PipelineParams(
            allow_interruptions=True,
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ),
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Pipecat client ready.")
        await rtvi.set_bot_ready()

    @pipecat_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Pipecat Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @pipecat_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")

    @pipecat_transport.event_handler("on_client_closed")
    async def on_client_closed(transport, client):
        logger.info("Pipecat Client closed")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)
