import asyncio
import dataclasses
from enum import Enum
from typing import Any, AsyncGenerator, Dict, Optional

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import GLib, Gst, GstApp
from loguru import logger

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


# Reuse frame definitions from original implementation
@dataclasses.dataclass
class PlayPipelineFrame(Frame):
    """A custom frame to request playing a GStreamer source pipeline."""

    pipeline_description: str


@dataclasses.dataclass
class PlayPipelineEndFrame(Frame):
    """A custom frame to indicate the end of a play pipeline request."""

    error: str = dataclasses.field(default=None)


# Signal types for internal queue communication
class SignalType(Enum):
    EOS = "eos"  # End of Stream
    ERROR = "error"  # Error with message
    PLAYING = "playing"  # Pipeline started playing successfully


@dataclasses.dataclass
class GstSignal:
    """Internal signal from GStreamer thread to asyncio event loop"""

    type: SignalType
    message: str = None


class GStreamerPipelinePlayerNew(FrameProcessor):
    """A Pipecat FrameProcessor that dynamically plays a GStreamer pipeline
    when it receives a PlayPipelineFrame, pushing OutputAudioRawFrames
    and OutputImageRawFrames downstream. Uses asyncio-based design.
    """

    def __init__(self, **kwargs):
        """Initialize the GStreamerPipelinePlayerNew processor.

        This version uses asyncio primitives and generator-based processing
        for managing GStreamer pipelines.
        """
        super().__init__(**kwargs)
        logger.debug("Initializing GStreamerPipelinePlayerNew...")
        try:
            Gst.init(None)
            logger.info("GStreamer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize GStreamer: {e}")
            raise

        # Asyncio related attributes
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._output_frames_queue = asyncio.Queue()
        self._gst_control_signal_queue = asyncio.Queue()
        self._pipeline_queue = asyncio.Queue()  # Queue for pipeline requests
        self._gst_pipeline_task: Optional[asyncio.Task] = None
        self._gst_runner_task: Optional[asyncio.Future] = None
        self._pipeline_processor_task: Optional[asyncio.Task] = None

        # Pipeline configuration
        self._audio_sample_rate: int = 16000
        self._audio_channels: int = 1
        self._video_width: int = 640
        self._video_height: int = 480
        self._video_format: str = "RGB"

        # GStreamer state
        self._pipeline: Optional[Gst.Pipeline] = None
        self._glib_loop: Optional[GLib.MainLoop] = None
        self._audio_sink: Optional[GstApp.AppSink] = None
        self._video_sink: Optional[GstApp.AppSink] = None
        self._bus_watch_id: Optional[int] = None
        self._is_playing = False
        self._current_pipeline_description = None

        logger.info("GStreamerPipelinePlayerNew initialized.")

    @property
    def is_playing(self) -> bool:
        """Returns True if a pipeline is currently playing."""
        return self._is_playing

    async def process_generator(self, generator):
        """Process frames from an async generator.

        Args:
            generator: An async generator that yields frames.

        This method consumes frames from the generator and pushes them downstream.
        """
        try:
            async for frame in generator:
                await self.push_frame(frame)
        except asyncio.CancelledError:
            logger.debug(f"{self}: Generator processing cancelled")
            raise
        except Exception as e:
            logger.exception(f"{self}: Error processing generator: {e}")
            await self.push_frame(ErrorFrame(f"Generator processing error: {e}"))

    async def start(self, frame: StartFrame):
        """Start frame processing."""
        await super().start(frame)
        # Start pipeline processor task if not already running
        if not self._pipeline_processor_task or self._pipeline_processor_task.done():
            self._pipeline_processor_task = asyncio.create_task(self._process_pipeline_queue())

    async def stop(self, frame: EndFrame):
        """Stop frame processing."""
        # Cancel and clear pipeline processor task
        if self._pipeline_processor_task and not self._pipeline_processor_task.done():
            self._pipeline_processor_task.cancel()
            try:
                await self._pipeline_processor_task
            except asyncio.CancelledError:
                pass
        self._pipeline_processor_task = None
        await super().stop(frame)

    async def _process_pipeline_queue(self):
        """Process pipeline requests from the queue one at a time."""
        while True:
            # Wait for the next pipeline request
            pipeline_description = await self._pipeline_queue.get()

            try:
                # Process the pipeline
                logger.info(
                    f"{self}: Processing pipeline from queue: {pipeline_description[:50]}..."
                )
                self._current_pipeline_description = pipeline_description
                self._is_playing = True

                # Run the pipeline and process its output
                await self.process_generator(self._run_gst_pipeline_instance(pipeline_description))

            except asyncio.CancelledError:
                logger.info(f"{self}: Pipeline processor task cancelled")
                raise
            except Exception as e:
                logger.exception(f"{self}: Error processing pipeline: {e}")
            finally:
                # Mark task as done
                self._pipeline_queue.task_done()
                self._is_playing = False
                self._current_pipeline_description = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames, handling PlayPipelineFrames specially."""
        await super().process_frame(frame, direction)

        # Capture the running asyncio event loop if not already set
        if not self._loop:
            self._loop = asyncio.get_running_loop()
            logger.debug(f"{self}: Event loop captured.")

        if isinstance(frame, PlayPipelineFrame):
            logger.debug(
                f"{self}: Received PlayPipelineFrame: '{frame.pipeline_description[:100]}...'"
            )

            # Queue the pipeline request instead of starting it immediately
            await self._pipeline_queue.put(frame.pipeline_description)
            logger.debug(f"{self}: Queued pipeline request, is_playing={self._is_playing}")

            # Start processor task if it's not running
            if not self._pipeline_processor_task or self._pipeline_processor_task.done():
                logger.debug(f"{self}: Starting pipeline processor task")
                self._pipeline_processor_task = asyncio.create_task(self._process_pipeline_queue())

        elif isinstance(frame, PlayPipelineEndFrame):
            logger.debug(f"{self}: Received PlayPipelineEndFrame, setting is_playing to False")
            self._is_playing = False
        else:
            # For all other frames, pass through
            await self.push_frame(frame, direction)

    def _handle_pipeline_task_done(self, task):
        """Handle completion of the pipeline task, including exceptions"""
        try:
            # This will re-raise any exception that occurred in the task
            task.result()
            logger.debug(f"{self}: Pipeline task completed successfully")
        except asyncio.CancelledError:
            logger.debug(f"{self}: Pipeline task was cancelled")
        except Exception as e:
            logger.error(f"{self}: Pipeline task failed with error: {e}")
        finally:
            self._is_playing = False

    async def _run_gst_pipeline_instance(
        self, pipeline_description: str
    ) -> AsyncGenerator[Frame, None]:
        """Run a GStreamer pipeline described in pipeline_description.

        This is an async generator that yields frames produced by the pipeline.

        Args:
            pipeline_description: A string describing the GStreamer pipeline to run.

        Yields:
            OutputAudioRawFrames and OutputImageRawFrames from the pipeline,
            as well as PlayPipelineEndFrame at the end.
        """
        logger.info(
            f"{self}: Starting _run_gst_pipeline_instance generator with pipeline: {pipeline_description[:100]}..."
        )

        # Clear any pending items in the queues
        while not self._output_frames_queue.empty():
            try:
                self._output_frames_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        while not self._gst_control_signal_queue.empty():
            try:
                self._gst_control_signal_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Start the GStreamer pipeline in a separate thread via run_in_executor
        # Initialize pipeline_error at the function scope so it's available in all blocks
        pipeline_error = None

        try:
            # Launch the GStreamer pipeline in a separate thread
            self._gst_runner_task = asyncio.create_task(
                asyncio.to_thread(self._execute_gst_pipeline_sync, pipeline_description)
            )

            # Wait for pipeline to report it's in PLAYING state or has an error
            logger.debug(f"{self}: Waiting for pipeline to start playing")
            try:
                signal = await asyncio.wait_for(self._gst_control_signal_queue.get(), timeout=10.0)
                if signal.type == SignalType.ERROR:
                    logger.error(f"{self}: Pipeline failed to start: {signal.message}")
                    yield ErrorFrame(f"Failed to start pipeline: {signal.message}")
                    yield PlayPipelineEndFrame(error=signal.message)
                    return
                elif signal.type == SignalType.PLAYING:
                    logger.info(f"{self}: Pipeline successfully started playing")
                else:
                    logger.warning(
                        f"{self}: Unexpected signal while waiting for pipeline to start: {signal.type}"
                    )
            except asyncio.TimeoutError:
                logger.error(f"{self}: Timeout waiting for pipeline to start")
                yield ErrorFrame("Timeout waiting for pipeline to start")
                yield PlayPipelineEndFrame(error="Timeout waiting for pipeline to start")
                return

            # Main frame yielding loop
            logger.debug(f"{self}: Entering main frame processing loop")

            while True:
                # Process frames and signals from the pipeline
                try:
                    # Use wait_for with a timeout to make the loop cancellation-friendly
                    entry = await asyncio.wait_for(
                        asyncio.create_task(self._get_next_queue_entry()), timeout=0.5
                    )

                    if isinstance(entry, Frame):
                        # It's a frame from the output queue
                        yield entry
                    elif isinstance(entry, GstSignal):
                        # It's a control signal
                        if entry.type == SignalType.EOS:
                            logger.info(f"{self}: Received EOS signal from GStreamer")
                            break
                        elif entry.type == SignalType.ERROR:
                            logger.error(
                                f"{self}: Received error signal from GStreamer: {entry.message}"
                            )
                            pipeline_error = entry.message
                            yield ErrorFrame(f"GStreamer error: {entry.message}")
                            break

                except asyncio.TimeoutError:
                    # Check if the GStreamer thread has finished
                    if self._gst_runner_task.done():
                        try:
                            self._gst_runner_task.result()
                            logger.debug(f"{self}: GStreamer thread exited normally")
                        except Exception as e:
                            logger.error(f"{self}: GStreamer thread exited with error: {e}")
                            pipeline_error = str(e)
                            yield ErrorFrame(f"GStreamer thread error: {e}")
                        break

                    # Continue listening for frames/signals
                    continue

        except asyncio.CancelledError:
            logger.info(f"{self}: Pipeline generator cancelled, stopping pipeline")
            pipeline_error = "Pipeline was cancelled"
            raise

        except asyncio.CancelledError:
            logger.info(f"{self}: Pipeline generator cancelled")
            pipeline_error = "Pipeline was cancelled"
            raise

        except Exception as e:
            logger.exception(f"{self}: Unexpected error in pipeline generator: {e}")
            yield ErrorFrame(f"Unexpected pipeline error: {e}")
            pipeline_error = str(e)

        finally:
            # End the pipeline and yield the end frame
            logger.debug(f"{self}: Cleaning up GStreamer pipeline")

            # Stop and clean up the GLib main loop
            await self._cleanup_gst_resources()

            # Yield the end frame if we haven't already
            yield PlayPipelineEndFrame(error=pipeline_error)

            logger.info(f"{self}: Pipeline generator completed")

    async def _get_next_queue_entry(self):
        """Get the next item from either the frames queue or the control signal queue.

        This method helps us process frames and control signals in the order they arrive.
        """
        # Create tasks for getting from each queue
        frame_task = asyncio.create_task(self._output_frames_queue.get())
        signal_task = asyncio.create_task(self._gst_control_signal_queue.get())

        # Wait for either task to complete
        done, pending = await asyncio.wait(
            [frame_task, signal_task], return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel the pending task
        for task in pending:
            task.cancel()

        # Return the result from whichever task completed
        return done.pop().result()

    def _execute_gst_pipeline_sync(self, pipeline_description: str):
        """Execute a GStreamer pipeline synchronously.

        This method runs in a separate thread (via run_in_executor).
        It sets up and runs the GStreamer pipeline, handling callbacks.

        Args:
            pipeline_description: A string describing the GStreamer pipeline to run.
        """
        logger.info(f"{self} GStreamer Thread: Setting up pipeline: {pipeline_description}")
        self._glib_loop = GLib.MainLoop()

        try:
            registry = Gst.Registry.get()
            element_factory_type = Gst.ElementFactory

            # Lower the rank of nvh264dec (common issue with hardware acceleration)
            nv_feature = registry.find_feature("nvh264dec", element_factory_type)
            if nv_feature:
                nv_feature.set_rank(Gst.Rank.NONE)
                logger.info("Set rank of 'nvh264dec' GStreamer feature to NONE.")

            # Create the pipeline and components
            self._pipeline = Gst.Pipeline.new("player")
            source = Gst.parse_bin_from_description(pipeline_description, True)
            if not source:
                error_msg = "Failed to parse GStreamer source description"
                logger.error(f"{self} GStreamer Thread: {error_msg}")
                self._signal_error(error_msg)
                return

            decodebin = Gst.ElementFactory.make("decodebin", "decodebin")
            if not decodebin:
                error_msg = "Failed to create GStreamer decodebin"
                logger.error(f"{self} GStreamer Thread: {error_msg}")
                self._signal_error(error_msg)
                return

            # Connect the pad-added signal for decodebin
            decodebin.connect("pad-added", self._on_pad_added)

            # Add elements to the pipeline
            self._pipeline.add(source)
            self._pipeline.add(decodebin)

            # Link source bin to decodebin
            if not source.link(decodebin):
                error_msg = "Failed to link GStreamer source to decodebin"
                logger.error(f"{self} GStreamer Thread: {error_msg}")
                self._signal_error(error_msg)
                return

            # Add bus watch
            bus = self._pipeline.get_bus()
            self._bus_watch_id = bus.add_watch(GLib.PRIORITY_DEFAULT, self._on_bus_message)

            if self._bus_watch_id is None:
                error_msg = "Failed to add GStreamer bus watch"
                logger.error(f"{self} GStreamer Thread: {error_msg}")
                self._signal_error(error_msg)
                return

            # Start playing
            logger.info(f"{self} GStreamer Thread: Setting pipeline to PLAYING state")
            ret = self._pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                error_msg = "Failed to set pipeline to PLAYING state"
                logger.error(f"{self} GStreamer Thread: {error_msg}")
                self._signal_error(error_msg)
                return

            # Run the GLib main loop
            logger.debug(f"{self} GStreamer Thread: Starting GLib main loop")
            self._glib_loop.run()
            logger.debug(f"{self} GStreamer Thread: GLib main loop finished")

        except Exception as e:
            logger.exception(f"{self} GStreamer Thread: Exception in pipeline thread: {e}")
            self._signal_error(f"Pipeline error: {e}")
            if self._glib_loop and self._glib_loop.is_running():
                self._glib_loop.quit()

        finally:
            logger.debug(f"{self} GStreamer Thread: Cleaning up pipeline resources")
            self._cleanup_gst_resources_sync()

    def _on_pad_added(self, decodebin: Gst.Element, pad: Gst.Pad):
        """Handle dynamic pads added by decodebin."""
        # This runs in the GStreamer thread
        caps = pad.get_current_caps()
        if not caps:
            logger.warning(
                f"{self} GStreamer Thread: Pad '{pad.get_name()}' added with no caps, ignoring."
            )
            return

        caps_str = caps.to_string()
        structure_name = caps.get_structure(0).get_name()
        pad_name = pad.get_name()
        logger.info(f"{self} GStreamer Thread: Pad '{pad_name}' added with caps: {caps_str}")

        # Process based on the media type
        if structure_name.startswith("audio/"):
            self._handle_audio_pad(pad, pad_name)
        elif structure_name.startswith("video/"):
            self._handle_video_pad(pad, pad_name)
        else:
            logger.warning(
                f"{self} GStreamer Thread: Ignoring pad '{pad_name}' with unhandled caps: {caps_str}"
            )

    def _handle_audio_pad(self, pad: Gst.Pad, pad_name: str = None):
        """Creates and links the necessary elements for an audio stream."""
        logger.debug(f"{self} GStreamer Thread: Handling audio pad '{pad_name}'.")

        if self._audio_sink:
            logger.warning(f"{self} GStreamer Thread: Audio sink already exists, ignoring.")
            return

        # Define element names
        queue_name = f"audio_queue_{pad_name}"
        convert_name = f"audioconvert_{pad_name}"
        resample_name = f"audioresample_{pad_name}"
        capsfilter_name = f"audiocapsfilter_{pad_name}"
        sink_name = f"audio_sink_{pad_name}"

        # Create elements
        queue_audio = Gst.ElementFactory.make("queue", queue_name)
        convert = Gst.ElementFactory.make("audioconvert", convert_name)
        resample = Gst.ElementFactory.make("audioresample", resample_name)
        capsfilter = Gst.ElementFactory.make("capsfilter", capsfilter_name)
        sink = Gst.ElementFactory.make("appsink", sink_name)

        elements = [queue_audio, convert, resample, capsfilter, sink]
        if not all(elements):
            error_msg = "Failed to create audio processing elements"
            logger.error(f"{self} GStreamer Thread: {error_msg}")
            self._signal_error(error_msg)
            return

        # Configure capsfilter
        audio_caps_str = (
            f"audio/x-raw, format=S16LE, layout=interleaved, "
            f"rate={self._audio_sample_rate}, channels={self._audio_channels}"
        )
        target_caps = Gst.caps_from_string(audio_caps_str)
        capsfilter.set_property("caps", target_caps)

        # Configure appsink
        sink.set_property("emit-signals", True)
        sink.set_property("max-buffers", 2)
        sink.set_property("drop", True)
        sink.connect("new-sample", self._on_new_audio_sample)
        self._audio_sink = sink

        # Add elements to the pipeline
        if not all(self._pipeline.add(el) for el in elements):
            error_msg = "Failed to add audio elements to pipeline"
            logger.error(f"{self} GStreamer Thread: {error_msg}")
            self._signal_error(error_msg)
            self._audio_sink = None
            for el in elements:
                if el.get_parent() == self._pipeline:
                    self._pipeline.remove(el)
            return

        # Sync state
        for el in elements:
            el.sync_state_with_parent()

        # Link elements
        if (
            not queue_audio.link(convert)
            or not convert.link(resample)
            or not resample.link(capsfilter)
            or not capsfilter.link(sink)
        ):
            error_msg = "Failed to link audio elements"
            logger.error(f"{self} GStreamer Thread: {error_msg}")
            self._signal_error(error_msg)
            self._audio_sink = None
            for el in elements:
                if el.get_parent() == self._pipeline:
                    self._pipeline.remove(el)
            return

        # Link pad to queue
        queue_pad = queue_audio.get_static_pad("sink")
        if not queue_pad:
            error_msg = "Failed to get audio queue sink pad"
            logger.error(f"{self} GStreamer Thread: {error_msg}")
            self._signal_error(error_msg)
            return

        link_ret = pad.link(queue_pad)
        if link_ret != Gst.PadLinkReturn.OK:
            error_msg = f"Failed to link audio pad {pad_name} to queue"
            logger.error(f"{self} GStreamer Thread: {error_msg}: {link_ret.value_name}")
            self._signal_error(error_msg)
            return

        logger.info(f"{self} GStreamer Thread: Audio processing chain linked for pad '{pad_name}'.")

    def _handle_video_pad(self, pad: Gst.Pad, pad_name: str = None):
        """Creates and links the necessary elements for a video stream."""
        logger.debug(f"{self} GStreamer Thread: Handling video pad '{pad_name}'.")

        if self._video_sink:
            logger.warning(f"{self} GStreamer Thread: Video sink already exists, ignoring.")
            return

        # Define element names
        queue_name = f"video_queue_{pad_name}"
        convert_name = f"videoconvert_{pad_name}"
        scale_name = f"videoscale_{pad_name}"
        capsfilter_name = f"videocapsfilter_{pad_name}"
        sink_name = f"video_sink_{pad_name}"

        # Create elements
        queue_video = Gst.ElementFactory.make("queue", queue_name)
        convert = Gst.ElementFactory.make("videoconvert", convert_name)
        scale = Gst.ElementFactory.make("videoscale", scale_name)
        capsfilter = Gst.ElementFactory.make("capsfilter", capsfilter_name)
        sink = Gst.ElementFactory.make("appsink", sink_name)

        elements = [queue_video, convert, scale, capsfilter, sink]
        if not all(elements):
            error_msg = "Failed to create video processing elements"
            logger.error(f"{self} GStreamer Thread: {error_msg}")
            self._signal_error(error_msg)
            return

        # Configure capsfilter
        video_caps_str = (
            f"video/x-raw, format={self._video_format}, "
            f"width={self._video_width}, height={self._video_height}, "
            f"framerate=25/1"
        )
        target_caps = Gst.caps_from_string(video_caps_str)
        capsfilter.set_property("caps", target_caps)

        # Configure appsink
        sink.set_property("emit-signals", True)
        sink.set_property("max-buffers", 2)
        sink.set_property("drop", True)
        sink.connect("new-sample", self._on_new_video_sample)
        self._video_sink = sink

        # Add elements to the pipeline
        if not all(self._pipeline.add(el) for el in elements):
            error_msg = "Failed to add video elements to pipeline"
            logger.error(f"{self} GStreamer Thread: {error_msg}")
            self._signal_error(error_msg)
            self._video_sink = None
            for el in elements:
                if el.get_parent() == self._pipeline:
                    self._pipeline.remove(el)
            return

        # Sync state
        for el in elements:
            el.sync_state_with_parent()

        # Link elements
        if (
            not queue_video.link(convert)
            or not convert.link(scale)
            or not scale.link(capsfilter)
            or not capsfilter.link(sink)
        ):
            error_msg = "Failed to link video elements"
            logger.error(f"{self} GStreamer Thread: {error_msg}")
            self._signal_error(error_msg)
            self._video_sink = None
            for el in elements:
                if el.get_parent() == self._pipeline:
                    self._pipeline.remove(el)
            return

        # Link pad to queue
        queue_pad = queue_video.get_static_pad("sink")
        if not queue_pad:
            error_msg = "Failed to get video queue sink pad"
            logger.error(f"{self} GStreamer Thread: {error_msg}")
            self._signal_error(error_msg)
            return

        link_ret = pad.link(queue_pad)
        if link_ret != Gst.PadLinkReturn.OK:
            error_msg = f"Failed to link video pad {pad_name} to queue"
            logger.error(f"{self} GStreamer Thread: {error_msg}: {link_ret.value_name}")
            self._signal_error(error_msg)
            return

        logger.info(f"{self} GStreamer Thread: Video processing chain linked for pad '{pad_name}'.")

    def _on_new_audio_sample(self, sink: GstApp.AppSink) -> Gst.FlowReturn:
        """Handle new audio samples from the audio appsink."""
        # This runs in the GStreamer thread
        sample = sink.emit("pull-sample")
        if not sample:
            logger.warning(f"{self} GStreamer Thread: Audio sink emitted null sample.")
            return Gst.FlowReturn.OK

        buffer = sample.get_buffer()
        if not buffer:
            logger.warning(f"{self} GStreamer Thread: Audio sample had no buffer.")
            return Gst.FlowReturn.OK

        caps = sample.get_caps()
        if not caps:
            logger.warning(f"{self} GStreamer Thread: Audio sample had no caps.")
            return Gst.FlowReturn.OK

        structure = caps.get_structure(0)
        rate = structure.get_value("rate")
        channels = structure.get_value("channels")

        # Map buffer to memory
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            logger.error(f"{self} GStreamer Thread: Failed to map audio buffer.")
            return Gst.FlowReturn.OK

        try:
            # Create Pipecat frame
            frame = OutputAudioRawFrame(
                audio=map_info.data,  # bytes object
                sample_rate=rate,
                num_channels=channels,
            )

            # Push to the asyncio queue safely
            self._loop.call_soon_threadsafe(self._output_frames_queue.put_nowait, frame)

        except Exception as e:
            logger.error(f"{self} GStreamer Thread: Error processing audio sample: {e}")
        finally:
            buffer.unmap(map_info)

        return Gst.FlowReturn.OK

    def _on_new_video_sample(self, sink: GstApp.AppSink) -> Gst.FlowReturn:
        """Handle new video samples from the video appsink."""
        # This runs in the GStreamer thread
        sample = sink.emit("pull-sample")
        if not sample:
            logger.warning(f"{self} GStreamer Thread: Video sink emitted null sample.")
            return Gst.FlowReturn.OK

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        if not buffer or not caps:
            logger.warning(f"{self} GStreamer Thread: Video sample missing buffer or caps.")
            return Gst.FlowReturn.OK

        # Map buffer to memory
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            logger.error(f"{self} GStreamer Thread: Failed to map video buffer.")
            return Gst.FlowReturn.OK

        try:
            # Create Pipecat frame
            frame = OutputImageRawFrame(
                image=map_info.data,  # bytes object
                format=self._video_format,
                size=(self._video_width, self._video_height),
            )

            # Push to the asyncio queue safely
            self._loop.call_soon_threadsafe(self._output_frames_queue.put_nowait, frame)

        except Exception as e:
            logger.error(f"{self} GStreamer Thread: Error processing video sample: {e}")
        finally:
            buffer.unmap(map_info)

        return Gst.FlowReturn.OK

    def _on_bus_message(self, bus: Gst.Bus, message: Gst.Message) -> bool:
        """Handle messages from the GStreamer bus."""
        # This runs in the GStreamer thread
        msg_type = message.type
        msg_src_name = message.src.get_name() if message.src else "UnknownSource"

        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            error_msg = f"GStreamer error from {msg_src_name}: {err} ({debug})"
            logger.error(f"{self} GStreamer Thread: {error_msg}")
            self._signal_error(error_msg)

            # Quit the main loop to stop processing
            if self._glib_loop and self._glib_loop.is_running():
                self._glib_loop.quit()

        elif msg_type == Gst.MessageType.EOS:
            logger.info(f"{self} GStreamer Thread: End-Of-Stream from {msg_src_name}.")

            # Check if EOS is from the pipeline itself or just a component
            if message.src == self._pipeline:
                logger.info(f"{self} GStreamer Thread: Pipeline EOS received, quitting GLib loop.")

                # Signal EOS to the asyncio loop
                self._loop.call_soon_threadsafe(
                    self._gst_control_signal_queue.put_nowait, GstSignal(type=SignalType.EOS)
                )

                # Quit the main loop to stop processing
                if self._glib_loop and self._glib_loop.is_running():
                    self._glib_loop.quit()

        elif msg_type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            logger.warning(
                f"{self} GStreamer Thread: Warning from {msg_src_name}: {warn} ({debug})"
            )

        elif msg_type == Gst.MessageType.STATE_CHANGED:
            # Only care about state changes of the pipeline itself
            if message.src == self._pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                logger.debug(
                    f"{self} GStreamer Thread: Pipeline state changed from {old_state.value_nick} "
                    f"to {new_state.value_nick} (pending: {pending_state.value_nick})"
                )
                # Signal when the pipeline reaches PLAYING state
                if new_state == Gst.State.PLAYING:
                    logger.info(f"{self} GStreamer Thread: Pipeline is now PLAYING.")
                    self._loop.call_soon_threadsafe(
                        self._gst_control_signal_queue.put_nowait,
                        GstSignal(type=SignalType.PLAYING),
                    )

        # Return True to keep the signal watch active
        return True

    def _signal_error(self, error_message: str):
        """Signal an error to the asyncio queue from the GStreamer thread."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(
                self._gst_control_signal_queue.put_nowait,
                GstSignal(type=SignalType.ERROR, message=error_message),
            )
        else:
            logger.error(
                f"{self} GStreamer Thread: Cannot signal error, loop not available: {error_message}"
            )

    def _cleanup_gst_resources_sync(self):
        """Clean up GStreamer resources synchronously (called from GStreamer thread)."""
        # Remove bus watch
        if self._bus_watch_id is not None:
            GLib.source_remove(self._bus_watch_id)
            self._bus_watch_id = None

        # Set pipeline to NULL state if it exists
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None

        # Clear sink references
        self._audio_sink = None
        self._video_sink = None

        # Quit GLib loop if it's running
        if self._glib_loop and self._glib_loop.is_running():
            self._glib_loop.quit()

        self._glib_loop = None
        logger.debug(f"{self} GStreamer Thread: Resources cleaned up synchronously")

    async def _cleanup_gst_resources(self):
        """Clean up GStreamer resources asynchronously (called from asyncio task)."""
        logger.debug(f"{self}: Cleaning up GStreamer resources")

        # Cancel and wait for the GStreamer thread to finish
        if self._gst_runner_task and not self._gst_runner_task.done():
            # First try to quit the GLib main loop cleanly
            if self._glib_loop and hasattr(self._glib_loop, "quit"):
                try:
                    # This will signal the thread to exit cleanly
                    GLib.idle_add(self._glib_loop.quit)
                    logger.debug(f"{self}: Sent quit signal to GLib loop")
                except Exception as e:
                    logger.error(f"{self}: Error quitting GLib loop: {e}")

            # Wait for the task to complete with a timeout
            try:
                await asyncio.wait_for(asyncio.shield(self._gst_runner_task), timeout=2.0)
                logger.debug(f"{self}: GStreamer thread exited normally")
            except asyncio.TimeoutError:
                logger.warning(f"{self}: Timeout waiting for GStreamer thread to exit")
                # Cancel the task (this is a forceful termination)
                self._gst_runner_task.cancel()
                try:
                    await self._gst_runner_task
                except asyncio.CancelledError:
                    logger.debug(f"{self}: GStreamer runner task cancelled")
                except Exception as e:
                    logger.error(f"{self}: Error while cancelling GStreamer runner: {e}")

        self._gst_runner_task = None
        self._pipeline = None
        self._audio_sink = None
        self._video_sink = None
        self._glib_loop = None
        self._bus_watch_id = None
        logger.debug(f"{self}: GStreamer resources cleaned up")

    async def cleanup(self):
        """Clean up resources when the processor is destroyed."""
        logger.info(f"{self}: Cleaning up GStreamerPipelinePlayerNew...")

        # Cancel pipeline processor task
        if self._pipeline_processor_task and not self._pipeline_processor_task.done():
            logger.debug(f"{self}: Cancelling pipeline processor task during cleanup")
            self._pipeline_processor_task.cancel()
            try:
                await self._pipeline_processor_task
            except asyncio.CancelledError:
                logger.debug(f"{self}: Pipeline processor task cancelled during cleanup")
            except Exception as e:
                logger.error(f"{self}: Error during pipeline processor task cancellation: {e}")

        # Cancel any running pipeline task
        if self._gst_pipeline_task and not self._gst_pipeline_task.done():
            logger.debug(f"{self}: Cancelling pipeline task during cleanup")
            self._gst_pipeline_task.cancel()
            try:
                await self._gst_pipeline_task
            except asyncio.CancelledError:
                logger.debug(f"{self}: Pipeline task cancelled during cleanup")
            except Exception as e:
                logger.error(f"{self}: Error during pipeline task cancellation: {e}")

        # Clean up GStreamer resources
        await self._cleanup_gst_resources()

        # Clear remaining references
        self._output_frames_queue = None
        self._gst_control_signal_queue = None
        self._pipeline_queue = None
        self._gst_pipeline_task = None
        self._pipeline_processor_task = None

        logger.info(f"{self}: Cleanup complete")
        await super().cleanup()
