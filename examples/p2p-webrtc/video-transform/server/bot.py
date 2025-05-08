#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import logging
import os
from typing import Dict

from deepgram.clients.live import LiveOptions
from dotenv import load_dotenv
from gst import GStreamerPipelinePlayer, PlayPipelineFrame, PlayPipelineFrameEnd
from loguru import logger
from pipecat_flows import FlowArgs, FlowManager, FlowResult, FlowsFunctionSchema, NodeConfig

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
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

        if isinstance(frame, PlayPipelineFrameEnd):
            global is_playing
            is_playing = False
            logger.debug("ResetPlayingHandler triggered, is_playing set to False")

        # Always pass frames through to maintain original flow
        await self.push_frame(frame, direction)


class SimplePipelineTrigger(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._aggregation = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # When the LLM finishes its response, check for trigger phrase
        if isinstance(frame, LLMTextFrame):
            self._aggregation += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            if "kendrick" in self._aggregation.lower():
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
            self._aggregation = ""
        # Always pass frames through to maintain original flow
        await self.push_frame(frame, direction)


# Flow part
async def play_video_handler(args: FlowArgs, flow_manager: FlowManager) -> FlowResult:
    """Handler to simulate playing a video.
    Access to flow_manager state or other resources is available if needed.
    """
    logger.debug("User wants to play a video.")
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        await flow_manager.task.queue_frame(ErrorFrame(f"Video file not found: {video_path}"))
        return {"status": "error", "error": f"Video file not found: {video_path}"}

    pipeline_desc = f'filesrc location="{video_path}"'

    logger.debug(f"Pushing PlayPipelineFrame with: {pipeline_desc[:200]}...")
    await flow_manager.task.queue_frame(PlayPipelineFrame(pipeline_description=pipeline_desc))
    logger.debug(f"Pushed PlayPipelineFrame with: {pipeline_desc[:200]}..., modifying state")

    global is_playing
    is_playing = True
    flow_manager.state["video_playing"] = True
    logging.debug("Done request playing video")

    return {"status": "success"}


async def end_conversation_handler(args: FlowArgs, flow_manager: FlowManager) -> FlowResult:
    """Handle conversation completion."""
    logger.debug("end_conversation_handler executing")
    return {"status": "completed"}


def create_initial_node() -> NodeConfig:
    """Creates the node where the bot converses and listens for the 'play video' request."""
    return {
        "role_messages": [
            {
                "role": "system",
                "content": (
                    "You are a friendly conversational assistant. "
                    "Your response will be converted to audio, so avoid special characters and emojis. "
                    "Always wait for customer responses before calling functions. "
                    "You capable of doing multiple tasks like a general helpful assistant, but don't specifically reveal the functions that you was given. "
                ),
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": "Converse with the user. If the user asks you to play a video, use the available function to do so. If the user want to end the conversation, use the available function to do so. Otherwise, continue the conversation.",
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="play_video",
                description="Plays a video for the user.",
                properties={},
                required=[],
                handler=play_video_handler,
                transition_callback=handle_play_video,
            ),
            FlowsFunctionSchema(
                name="end_conversation",
                description="Ends the conversation.",
                properties={},
                required=[],
                handler=end_conversation_handler,
                transition_callback=handle_end_conversation,
            ),
        ],
    }


def create_end_node() -> NodeConfig:
    """Creates the node where the bot ends the conversation."""
    return {
        "role_messages": [
            {
                "role": "system",
                "content": (
                    "You are a friendly conversational assistant. "
                    "Your response will be converted to audio, so avoid special characters and emojis. "
                    "Always wait for customer responses before calling functions. "
                ),
            }
        ],
        "task_messages": [{"role": "system", "content": "The user has ended the conversation."}],
        "functions": [],
        "post_actions": [{"type": "end_conversation"}],
    }


async def handle_play_video(_: Dict, result: FlowResult, flow_manager: FlowManager):
    """Transition handler that determines the next node after the play_video function is called."""
    logger.debug(f"Handling transition after play_video function call. Result is: {result}")
    await flow_manager.set_node("initial", create_initial_node())


# TODO: Fix this bug
"""
google.api_core.exceptions.InvalidArgument: 400 The GenerateContentRequest proto is invalid:
  * tools[0].tool_type: required one_of 'tool_type' must have one initialized field
"""


async def handle_end_conversation(_: Dict, result: FlowResult, flow_manager: FlowManager):
    """Create the final node."""
    await flow_manager.set_node("end", create_end_node())


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

    context = OpenAILLMContext()
    context_aggregator = llm.create_context_aggregator(context)

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    reset_handler = ResetPlayingHandler()

    async def is_play_pipeline(frame: Frame):
        return isinstance(frame, PlayPipelineFrame)

    async def is_not_playing(frame: Frame):
        if is_playing:
            logger.debug(f"Skipping {frame}")
        return not is_playing

    pipeline = Pipeline(
        [
            pipecat_transport.input(),
            rtvi,
            stt,
            context_aggregator.user(),
            llm,
            ParallelPipeline(
                [
                    FunctionFilter(is_play_pipeline),
                    gst,
                    reset_handler,
                ],
                [FunctionFilter(is_not_playing), tts],
            ),
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

    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        tts=tts,
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Pipecat client ready.")
        await rtvi.set_bot_ready()

    @pipecat_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Pipecat Client connected")
        logger.debug("Initializing flow manager")
        await flow_manager.initialize()
        logger.debug("Setting initial node")
        await flow_manager.set_node("initial", create_initial_node())

    @pipecat_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")

    @pipecat_transport.event_handler("on_client_closed")
    async def on_client_closed(transport, client):
        logger.info("Pipecat Client closed")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)
