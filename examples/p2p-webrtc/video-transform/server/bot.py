#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import os

from dotenv import load_dotenv
from flow import create_initial_node
from gst import GStreamerPipelinePlayer, PlayPipelineFrame
from loguru import logger
from pipecat_flows import FlowManager

from pipecat.audio.filters.noisereduce_filter import NoisereduceFilter
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

load_dotenv(override=True)


async def run_bot(webrtc_connection):
    transport_params = TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_out_10ms_chunks=2,
        audio_in_filter=NoisereduceFilter(),
        video_in_enabled=True,
        video_out_enabled=True,
        video_out_width=736,
        video_out_height=1310,
        video_out_is_live=True,
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(confidence=0.8, start_secs=0.3, stop_secs=0.7, min_volume=0.7)
        ),
    )

    pipecat_transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection, params=transport_params
    )

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
    )

    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))

    # tts = OpenAITTSService(base_url="https://dev-vh-voice-center.vuihoc.vn/api/v2", api_key="5b62ef4f-87d9-4928-865a-8d4c70e0bbf9", sample_rate=24000)
    tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    gst = GStreamerPipelinePlayer()

    context = OpenAILLMContext()
    context_aggregator = llm.create_context_aggregator(context)

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    async def is_play_pipeline(frame: Frame):
        return isinstance(frame, PlayPipelineFrame)

    async def is_not_playing(frame: Frame):
        return not gst.is_playing

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
