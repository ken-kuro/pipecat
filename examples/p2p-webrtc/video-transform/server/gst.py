import asyncio
import dataclasses
import threading
from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import GLib, Gst, GstApp
except ImportError as err:
    logger.error(f"Failed to import GStreamer: {err}")
    logger.error(
        "Please refer to https://gstreamer.freedesktop.org/documentation/tutorials/index.html?gi-language=c for more information on installation."
    )
    raise Exception("GStreamer is not installed. Please install GStreamer to use this module.")


# Step 1: Define the custom frame to trigger playback
@dataclasses.dataclass
class PlayPipelineFrame(Frame):
    """A custom frame to request playing a GStreamer source pipeline."""

    pipeline_description: str


@dataclasses.dataclass
class PlayPipelineFrameEnd(Frame):
    """A custom frame to indicate the end of a play pipeline request."""

    error: str = dataclasses.field(default=None)


# Step 2 & 3: Basic GStreamerPipelinePlayer class structure
class GStreamerPipelinePlayer(FrameProcessor):
    """A Pipecat FrameProcessor that dynamically plays a GStreamer pipeline
    when it receives a PlayPipelineFrame, pushing OutputAudioRawFrames
    and OutputImageRawFrames downstream.
    """

    def __init__(self, **kwargs):
        """Initializes the GStreamerPipelinePlayer.

        Args:
            TODO: Add more config on output in the future
        """
        super().__init__(**kwargs)
        logger.debug("Initializing GStreamerPipelinePlayer...")
        try:
            Gst.init(None)
            logger.info("GStreamer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize GStreamer: {e}")
            raise
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pipeline_lock = threading.Lock()
        self._pipeline_stopped_condition = threading.Condition(self._pipeline_lock)
        self._pipeline: Optional[Gst.Pipeline] = None
        self._pipeline_thread: Optional[threading.Thread] = None
        self._glib_loop: Optional[GLib.MainLoop] = None
        self._bus_watch_id: Optional[int] = None
        self._audio_sink: Optional[GstApp.Element] = None
        self._video_sink: Optional[GstApp.Element] = None

        self._audio_sample_rate: int = 16000
        self._audio_channels: int = 1
        self._video_width: int = 640
        self._video_height: int = 480
        self._video_format: str = "RGB"
        self._audio_sink_caps_str: str = f"audio/x-raw,format=S16LE,layout=interleaved,rate={self._audio_sample_rate},channels={self._audio_channels}"
        self._video_sink_caps_str: str = f"video/x-raw,format={self._video_format}, width={self._video_width},height={self._video_height}"

        logger.info("GStreamerPipelinePlayer initialized.")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes incoming frames, looking for PlayPipelineFrame."""
        await super().process_frame(frame, direction)

        if not self._loop:
            self._loop = asyncio.get_running_loop()
            logger.debug(f"{self}: Event loop captured.")

        if isinstance(frame, PlayPipelineFrame):
            logger.debug(
                f"{self}: Received PlayPipelineFrame request: '{frame.pipeline_description}'"
            )

            if not self._loop:
                logger.error(f"{self}: Event loop not captured, cannot start pipeline thread.")
                # Push an error frame back? This requires the loop, so just log for now.
                await self.push_frame(ErrorFrame(f"{self}: Event loop not available"))
                return  # Don't proceed

            # Use run_in_executor to avoid blocking the event loop with lock acquisition
            # and potential waiting on the condition variable
            await self._loop.run_in_executor(
                None, self._start_pipeline_thread, frame.pipeline_description
            )
        else:
            await self.push_frame(frame, direction)

    def _start_pipeline_thread(self, pipeline_description: str):
        """Stops any existing pipeline and starts a new one in a thread.
        Ensures the previous thread has fully stopped before starting the new one.
        This method is designed to be run in a separate thread (e.g., via run_in_executor)
        to avoid blocking the main asyncio loop.
        """
        with self._pipeline_lock:
            logger.debug(f"{self}: Attempting to start pipeline: {pipeline_description}")
            # Wait for any existing pipeline thread to fully stop and clean up.
            while self._pipeline_thread and self._pipeline_thread.is_alive():
                logger.info(f"{self}: Existing pipeline thread found, stopping and waiting...")
                # Call _stop_pipeline_sync without holding the lock recursively
                # (it acquires the lock itself). This is slightly complex.
                # A simpler model: _stop_pipeline_sync signals, we wait here.
                if self._glib_loop and self._glib_loop.is_running():
                    self._glib_loop.quit()
                if self._pipeline:
                    self._pipeline.set_state(Gst.State.NULL)

                # Wait for the _stop_pipeline_sync (potentially called by cleanup or another thread)
                # or the thread's own finally block to signal completion.
                logger.debug(f"{self}: Waiting for pipeline stopped condition...")
                self._pipeline_stopped_condition.wait(timeout=5.0)  # Add timeout
                if self._pipeline_thread and self._pipeline_thread.is_alive():
                    logger.warning(f"{self}: Pipeline thread still alive after waiting.")
                    # Decide recovery strategy: maybe force kill? For now, log and proceed cautiously.
                    # Break to prevent infinite loop if stop fails repeatedly.
                    break
                else:
                    logger.debug(f"{self}: Pipeline thread confirmed stopped.")
                    self._pipeline_thread = None  # Ensure it's None if wait succeeded

            # If after waiting, the thread object still exists (e.g., timeout occurred), log it.
            if self._pipeline_thread:
                logger.warning(
                    f"{self}: Proceeding to start new pipeline despite previous thread object still existing."
                )
                # Force set to None to allow new thread creation
                self._pipeline_thread = None

            # Reset sink references for new pipeline
            self._audio_sink = None
            self._video_sink = None

            # Start the new pipeline in its own thread
            logger.debug(f"{self}: Creating GStreamer thread...")
            self._pipeline_thread = threading.Thread(
                target=self._run_pipeline,
                # THE EXTRA COMMA IS NECESSARY TO MAKE ARGS A TUPLE
                args=(pipeline_description,),
                daemon=True,  # Daemon threads exit automatically if the main program exits
            )
            self._pipeline_thread.start()
            logger.info(f"{self}: GStreamer thread started for: {pipeline_description}")

    def _run_pipeline(self, pipeline_description: str):
        """The main function for the GStreamer thread."""
        logger.debug(f"{self} GStreamer Thread: Running pipeline: {pipeline_description}")
        self._glib_loop = GLib.MainLoop()
        pipeline = None  # Use local var until successfully setup

        try:
            registry = Gst.Registry.get()

            # Correctly specify the GType for ElementFactory
            element_factory_type = Gst.ElementFactory

            # TODO: For some reason, nvh264dec doesn't work on my work machine
            # https://gstreamer.freedesktop.org/documentation/tutorials/playback/hardware-accelerated-video-decoding.html?gi-language=c

            # Lower the rank of nvh264dec
            nv_feature = registry.find_feature("nvh264dec", element_factory_type)
            if nv_feature:
                nv_feature.set_rank(Gst.Rank.NONE)
                # registry.add_feature(nv_feature) # Re-adding might be needed in some Gst versions, test without first
                logger.info("Set rank of 'nvh264dec' GStreamer feature to NONE.")
            else:
                logger.warning("GStreamer feature 'nvh264dec' not found, cannot set rank.")

            # Create the main pipeline and decodebin
            pipeline = Gst.Pipeline.new("player")

            source = Gst.parse_bin_from_description(pipeline_description, True)
            if not source:
                logger.error(f"{self} GStreamer Thread: Failed to parse source description.")
                self._report_error("Failed to parse GStreamer source description")
                return
            logger.debug(f"{self} GStreamer Thread: Parsed source bin.")

            decodebin = Gst.ElementFactory.make("decodebin", "decodebin")
            if not pipeline or not decodebin:
                logger.error(f"{self} GStreamer Thread: Failed to create pipeline or decodebin.")
                self._report_error("Failed to create GStreamer pipeline/decodebin")
                return
            # Connect the pad-added signal for decodebin
            decodebin.connect("pad-added", self._on_pad_added)

            # Add elements to the pipeline
            pipeline.add(source)
            pipeline.add(decodebin)

            # Link source bin to decodebin
            if not source.link(decodebin):
                logger.error(f"{self} GStreamer Thread: Failed to link source bin to decodebin.")
                self._report_error("Failed to link GStreamer source to decodebin")
                return
            logger.debug(f"{self} GStreamer Thread: Linked source to decodebin.")

            # Add bus watch
            bus = pipeline.get_bus()
            # Store the bus watch ID
            self._bus_watch_id = bus.add_watch(GLib.PRIORITY_DEFAULT, self._on_bus_message)
            if self._bus_watch_id is None:
                logger.error(f"{self} GStreamer Thread: Failed add bus watch.")
                self._report_error("Failed to add GStreamer bus watch")
                return
            logger.debug(f"{self} GStreamer Thread: Bus watch added (ID: {self._bus_watch_id}).")

            with self._pipeline_lock:
                # If another thread called stop before we got here
                if not self._pipeline_thread or self._pipeline_thread != threading.current_thread():
                    logger.warning(
                        f"{self} GStreamer Thread: Pipeline stop requested before startup completed. Aborting."
                    )
                    if self._bus_watch_id is not None:
                        GLib.source_remove(self._bus_watch_id)  # Clean up bus watch
                        self._bus_watch_id = None
                    return  # Don't assign self._pipeline or start playing

                self._pipeline = pipeline  # Assign to instance variable *only* when fully set up

            # Start playing
            logger.info(f"{self} GStreamer Thread: Setting pipeline to PLAYING state.")
            ret = self._pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error(f"{self} GStreamer Thread: Failed to set pipeline to PLAYING state.")
                self._report_error("Failed to set pipeline state to PLAYING")
                # Cleanup should happen in finally block
                return
            elif ret == Gst.StateChangeReturn.ASYNC:
                logger.debug(f"{self} GStreamer Thread: Pipeline state change is asynchronous.")
                # State change will be confirmed via bus message

            # Run the GLib main loop
            logger.debug(f"{self} GStreamer Thread: Starting GLib main loop...")
            self._glib_loop.run()
            logger.debug(f"{self} GStreamer Thread: GLib main loop finished.")

        except Exception as e:
            logger.exception(f"{self} GStreamer Thread: Exception in pipeline thread: {e}")
            self._report_error(f"Pipeline error: {e}")
            # Ensure loop quits if exception happens before run() or during setup
            if self._glib_loop and not self._glib_loop.is_running():
                # If run() was never called or quit immediately
                pass
            elif self._glib_loop:
                self._glib_loop.quit()

        finally:
            logger.debug(f"{self} GStreamer Thread: Cleaning up pipeline...")
            # Use the lock for cleanup to ensure thread safety
            with self._pipeline_lock:
                # Remove bus watch *before* changing state or clearing pipeline ref
                if self._bus_watch_id is not None:
                    if self._pipeline:  # Check if bus still exists implicitly
                        bus = self._pipeline.get_bus()
                        # Check if source id is valid before removing
                        source = GLib.main_context_default().find_source_by_id(self._bus_watch_id)
                        if source:
                            source.destroy()  # Or GLib.source_remove(self._bus_watch_id)
                        else:
                            logger.warning(
                                f"{self} GStreamer Thread: Bus watch ID {self._bus_watch_id} not found for removal."
                            )
                    else:
                        logger.debug(
                            f"{self} GStreamer Thread: Pipeline already None, cannot remove bus watch."
                        )
                    self._bus_watch_id = None
                    logger.debug(f"{self} GStreamer Thread: Removed bus watch.")

                local_pipeline = self._pipeline  # Use local ref for final state change
                if local_pipeline:
                    logger.debug(f"{self} GStreamer Thread: Setting pipeline to NULL state...")
                    # No need to wait aggressively here, just set it
                    local_pipeline.set_state(Gst.State.NULL)
                    # Short wait might help elements release resources, but not strictly required
                    # _, current_state, _ = local_pipeline.get_state(timeout=100 * Gst.MSECOND) # 100ms timeout
                    # logger.debug(f"{self} GStreamer Thread: State after NULL request: {current_state.value_name}")

                    # Clear references held by this class instance *after* state change
                    self._pipeline = None
                    self._audio_sink = None
                    self._video_sink = None
                    logger.debug(f"{self} GStreamer Thread: Cleared internal pipeline references.")
                else:
                    logger.debug(f"{self} GStreamer Thread: self._pipeline was already None.")

                # Quit GLib loop if it hasn't already stopped
                # Check self._glib_loop again as it might be None if setup failed early
                if self._glib_loop and self._glib_loop.is_running():
                    logger.debug(f"{self} GStreamer Thread: Quitting GLib loop from finally block.")
                    self._glib_loop.quit()
                self._glib_loop = None  # Clear loop reference

                logger.debug(f"{self} GStreamer Thread: Notifying pipeline stopped condition.")
                self._pipeline_stopped_condition.notify_all()

                # Clear the thread reference *only if* this thread is the one assigned
                # This prevents a racing stop call from nulling out a *new* thread reference
                if self._pipeline_thread == threading.current_thread():
                    self._pipeline_thread = None
                    logger.debug(
                        f"{self} GStreamer Thread: Cleared self._pipeline_thread reference."
                    )
                else:
                    logger.warning(
                        f"{self} GStreamer Thread: self._pipeline_thread reference changed during execution, not clearing."
                    )

            logger.info(f"{self} GStreamer Thread: Pipeline cleanup complete, thread exiting.")

    def _on_pad_added(self, decodebin: Gst.Element, pad: Gst.Pad):
        """Handles dynamic pads added by decodebin."""
        # This runs in the GStreamer thread
        caps = pad.get_current_caps()
        if not caps:
            logger.warning(
                f"{self} GStreamer Thread: Pad '{pad.get_name()}' added to decodebin with no caps, ignoring."
            )

        caps_str = caps.to_string()
        logger.debug(f"decodebin pad added: {caps_str}")
        structure_name = caps.get_structure(0).get_name()
        pad_name = pad.get_name()
        logger.info(
            f"{self} GStreamer Thread: Pad '{pad_name}' added to decodebin with caps: {caps_str}"
        )

        # Check if it's audio
        if structure_name.startswith("audio/"):
            self._handle_audio_pad(pad, pad_name)
        # Check if it's video
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
            logger.warning(
                f"{self} GStreamer Thread: Audio sink already exists, ignoring new audio pad '{pad_name}'."
            )
            return

        logger.info(
            f"{self} GStreamer Thread: Creating audio processing chain for pad '{pad_name}'."
        )

        # Define element names (ensure uniqueness)
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

        elements_to_add = [
            queue_audio,
            convert,
            resample,
            capsfilter,
            sink,
        ]  # Keep track for adding/cleanup

        if not all(elements_to_add):
            logger.error(
                f"{self} GStreamer Thread: Failed to create one or more audio processing elements."
            )
            self._report_error("Failed to create audio processing elements")
            return

        # Configure capsfilter
        audio_caps_str = f"audio/x-raw, format=S16LE, layout=interleaved, rate={self._audio_sample_rate}, channels=1"
        target_caps = Gst.caps_from_string(audio_caps_str)
        capsfilter.set_property("caps", target_caps)
        logger.info(f"{self} GStreamer Thread: Setting audio capsfilter to: {audio_caps_str}")

        # Configure appsink
        sink.set_property("emit-signals", True)
        sink.set_property("max-buffers", 2)
        sink.set_property("drop", True)
        sink.connect("new-sample", self._on_new_audio_sample)
        self._audio_sink = sink  # Store the sink instance last, only if setup succeeds this far

        # Add elements to the pipeline
        if not all(self._pipeline.add(el) for el in elements_to_add):
            logger.error(
                f"{self} GStreamer Thread: Failed to add one or more audio elements to pipeline."
            )
            self._report_error("Failed to add audio elements to pipeline")
            self._audio_sink = None  # Reset on failure
            # Attempt to remove elements that might have been added
            for el in elements_to_add:
                if el.get_parent() == self._pipeline:  # Only remove if actually added
                    self._pipeline.remove(el)
            return

        # Sync state of newly added elements AFTER successful linking
        if not all(el.sync_state_with_parent() for el in elements_to_add):
            logger.warning(
                f"{self} GStreamer Thread: Failed to sync state for one or more audio elements."
            )
            # Continue anyway, but log warning

        # Link the elements sequentially (element1.link(element2))
        if not queue_audio.link(convert):
            logger.error(
                f"{self} GStreamer Thread: Failed to link {queue_audio.get_name()} -> {convert.get_name()}"
            )
            self._report_error("Failed link audio queue->convert")
            # Cleanup... (remove elements, reset sink)
            return  # Simplified error handling for brevity
        if not convert.link(resample):
            logger.error(
                f"{self} GStreamer Thread: Failed to link {convert.get_name()} -> {resample.get_name()}"
            )
            self._report_error("Failed link audio convert->resample")
            # Cleanup...
            return
        if not resample.link(capsfilter):
            logger.error(
                f"{self} GStreamer Thread: Failed to link {resample.get_name()} -> {capsfilter.get_name()}"
            )
            self._report_error("Failed link audio resample->capsfilter")
            # Cleanup...
            return
        if not capsfilter.link(self._audio_sink):
            logger.error(
                f"{self} GStreamer Thread: Failed to link {capsfilter.get_name()} -> {self._audio_sink.get_name()}"
            )
            self._report_error("Failed link audio capsfilter->sink")
            # Cleanup...
            return

        # Link the decodebin pad to the start of the chain (queue's sink pad)
        queue_pad = queue_audio.get_static_pad("sink")
        if not queue_pad:
            logger.error(
                f"{self} GStreamer Thread: Could not get sink pad for audio queue '{queue_name}'."
            )
            self._report_error("Failed to get audio queue sink pad")
            # Cleanup...
            return

        link_ret = pad.link(queue_pad)
        if link_ret != Gst.PadLinkReturn.OK:
            logger.error(
                f"{self} GStreamer Thread: Failed to link decodebin audio pad '{pad_name}' to queue '{queue_name}': {link_ret.value_name}"
            )
            self._report_error(f"Failed to link audio pad {pad_name} to queue")
            # Cleanup...
            return

        logger.info(
            f"{self} GStreamer Thread: Audio processing chain linked and synced for pad '{pad_name}'."
        )

    def _handle_video_pad(self, pad: Gst.Pad, pad_name: str = None):
        """Creates and links the necessary elements for a video stream."""
        logger.debug(f"{self} GStreamer Thread: Handling video pad '{pad_name}'.")

        if self._video_sink:
            logger.warning(
                f"{self} GStreamer Thread: Video sink already exists, ignoring new video pad '{pad_name}'."
            )
            return

        logger.info(
            f"{self} GStreamer Thread: Creating video processing chain for pad '{pad_name}'."
        )

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

        elements_to_add = [
            queue_video,
            convert,
            scale,
            capsfilter,
            sink,
        ]  # Keep track for adding/cleanup

        if not all(elements_to_add):
            logger.error(
                f"{self} GStreamer Thread: Failed to create one or more video processing elements."
            )
            self._report_error("Failed to create video processing elements")
            return

        # Configure capsfilter
        video_caps_str = (
            f"video/x-raw, format={self._video_format}, "
            f"width={self._video_width}, height={self._video_height}, "
            f"framerate=25/1"
        )  # Example framerate
        target_caps = Gst.caps_from_string(video_caps_str)
        capsfilter.set_property("caps", target_caps)
        logger.info(f"{self} GStreamer Thread: Setting video capsfilter to: {video_caps_str}")

        # Configure appsink
        sink.set_property("emit-signals", True)
        sink.set_property("max-buffers", 2)
        sink.set_property("drop", True)
        sink.connect("new-sample", self._on_new_video_sample)
        self._video_sink = sink  # Store the sink instance last

        # Add elements to the pipeline
        if not all(self._pipeline.add(el) for el in elements_to_add):
            logger.error(
                f"{self} GStreamer Thread: Failed to add one or more video elements to pipeline."
            )
            self._report_error("Failed to add video elements to pipeline")
            self._video_sink = None  # Reset on failure
            for el in elements_to_add:
                if el.get_parent() == self._pipeline:  # Only remove if actually added
                    self._pipeline.remove(el)
            return

        # Sync state of newly added elements AFTER successful linking
        if not all(el.sync_state_with_parent() for el in elements_to_add):
            logger.warning(
                f"{self} GStreamer Thread: Failed to sync state for one or more video elements."
            )
            # Continue anyway, but log warning

        # Link the elements sequentially
        if not queue_video.link(convert):
            logger.error(
                f"{self} GStreamer Thread: Failed to link {queue_video.get_name()} -> {convert.get_name()}"
            )
            self._report_error("Failed link video queue->convert")
            # Cleanup...
            return
        if not convert.link(scale):
            logger.error(
                f"{self} GStreamer Thread: Failed to link {convert.get_name()} -> {scale.get_name()}"
            )
            self._report_error("Failed link video convert->scale")
            # Cleanup...
            return
        if not scale.link(capsfilter):
            logger.error(
                f"{self} GStreamer Thread: Failed to link {scale.get_name()} -> {capsfilter.get_name()}"
            )
            self._report_error("Failed link video scale->capsfilter")
            # Cleanup...
            return
        if not capsfilter.link(self._video_sink):
            logger.error(
                f"{self} GStreamer Thread: Failed to link {capsfilter.get_name()} -> {self._video_sink.get_name()}"
            )
            self._report_error("Failed link video capsfilter->sink")
            # Cleanup...
            return

        # Link the decodebin pad to the start of the chain (queue's sink pad)
        queue_pad = queue_video.get_static_pad("sink")
        if not queue_pad:
            logger.error(
                f"{self} GStreamer Thread: Could not get sink pad for video queue '{queue_name}'."
            )
            self._report_error("Failed to get video queue sink pad")
            # Cleanup...
            return

        link_ret = pad.link(queue_pad)
        if link_ret != Gst.PadLinkReturn.OK:
            logger.error(
                f"{self} GStreamer Thread: Failed to link decodebin video pad '{pad_name}' to queue '{queue_name}': {link_ret.value_name}"
            )
            self._report_error(f"Failed to link video pad {pad_name} to queue")
            # Cleanup...
            return

        logger.info(
            f"{self} GStreamer Thread: Video processing chain linked and synced for pad '{pad_name}'."
        )

    def _on_new_audio_sample(self, sink: GstApp.AppSink) -> Gst.FlowReturn:
        """Handles new audio samples from the audio appsink."""
        # This runs in the GStreamer thread
        sample = sink.emit("pull-sample")
        if not sample:
            logger.warning(f"{self} GStreamer Thread: Audio sink emitted null sample.")
            return Gst.FlowReturn.OK  # Or maybe ERROR? Let's assume OK for now

        buffer = sample.get_buffer()
        if not buffer:
            logger.warning(f"{self} GStreamer Thread: Audio sample had no buffer.")
            return Gst.FlowReturn.OK

        caps = sample.get_caps()
        if not caps:
            logger.warning(f"{self} GStreamer Thread: Audio sample had no caps.")
            return Gst.FlowReturn.OK  # Cannot process without caps

        structure = caps.get_structure(0)
        rate = structure.get_value("rate")
        channels = structure.get_value("channels")

        # Map buffer to memory
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            logger.error(f"{self} GStreamer Thread: Failed to map audio buffer.")
            # Don't push frame, but let pipeline continue?
            return Gst.FlowReturn.OK  # Or maybe FlowReturn.ERROR?

        try:
            # Create Pipecat frame - copy data to ensure lifetime
            frame = OutputAudioRawFrame(
                audio=map_info.data,  # This is bytes
                sample_rate=rate,
                num_channels=channels,
            )

            # Push frame to asyncio loop
            if self._loop and self._loop.is_running():
                # logger.debug(f"{self} GStreamer Thread: Pushing OutputAudioRawFrame ({len(map_info.data)} bytes)") # Can be very noisy
                asyncio.run_coroutine_threadsafe(self.push_frame(frame), self._loop)
            else:
                logger.warning(
                    f"{self} GStreamer Thread: Event loop not available, dropping audio frame."
                )

        except Exception as e:
            logger.error(
                f"{self} GStreamer Thread: Error processing audio sample: {e}", exc_info=True
            )
            # Indicate error? For now, just log and continue.
        finally:
            buffer.unmap(map_info)  # Crucial: unmap the buffer

        return Gst.FlowReturn.OK

    def _on_new_video_sample(self, sink: GstApp.AppSink) -> Gst.FlowReturn:
        """Handles new video samples from the video appsink."""
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
                image=map_info.data,  # This is bytes
                format=self._video_format,
                size=(self._video_width, self._video_height),
            )

            # Push frame to asyncio loop
            if self._loop and self._loop.is_running():
                # logger.debug(f"{self} GStreamer Thread: Pushing OutputImageRawFrame ({width}x{height})") # Can be noisy
                asyncio.run_coroutine_threadsafe(self.push_frame(frame), self._loop)
            else:
                logger.warning(
                    f"{self} GStreamer Thread: Event loop not available, dropping video frame."
                )

        except Exception as e:
            logger.error(
                f"{self} GStreamer Thread: Error processing video sample: {e}", exc_info=True
            )
        finally:
            buffer.unmap(map_info)  # Crucial: unmap the buffer

        return Gst.FlowReturn.OK

    def _on_bus_message(self, bus: Gst.Bus, message: Gst.Message):
        """Handles messages from the GStreamer bus."""
        # This runs in the GStreamer thread
        msg_type = message.type
        msg_src_name = message.src.get_name() if message.src else "UnknownSource"

        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"{self} GStreamer Thread: Bus Error from {msg_src_name}: {err} ({debug})")
            self._report_error(f"GStreamer error from {msg_src_name}: {err}")
            if self._glib_loop:
                self._glib_loop.quit()
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.push_frame(ErrorFrame(f"GStreamer error: {err}")), self._loop
                )
                asyncio.run_coroutine_threadsafe(
                    self.push_frame(
                        PlayPipelineFrameEnd(error=err.__str__()), FrameDirection.DOWNSTREAM
                    ),
                    self._loop,
                )
        elif msg_type == Gst.MessageType.EOS:
            logger.info(f"{self} GStreamer Thread: Bus End-Of-Stream from {msg_src_name}.")
            # Don't quit the loop immediately on EOS from elements other than the pipeline itself?
            # Let the pipeline handle EOS propagation. Check if src is pipeline.
            if message.src == self._pipeline:
                logger.info(f"{self} GStreamer Thread: Pipeline EOS received, quitting GLib loop.")
                if self._glib_loop and self._glib_loop.is_running():
                    self._glib_loop.quit()
                if self._loop and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self.push_frame(PlayPipelineFrameEnd()), self._loop
                    )
            else:
                logger.debug(
                    f"{self} GStreamer Thread: EOS from element '{msg_src_name}', not quitting loop yet."
                )
        elif msg_type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            logger.warning(
                f"{self} GStreamer Thread: Bus Warning from {msg_src_name}: {warn} ({debug})"
            )
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            # We only care about state changes of the pipeline itself for debugging
            if message.src == self._pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                logger.debug(
                    f"{self} GStreamer Thread: Pipeline state changed from {old_state.value_nick} "
                    f"to {new_state.value_nick} (pending: {pending_state.value_nick})"
                )
                if new_state == Gst.State.PLAYING:
                    logger.info(f"{self} GStreamer Thread: Pipeline is now PLAYING.")
        # Add handling for other messages if needed (e.g., Buffering)
        # else:
        #     logger.debug(f"{self} GStreamer Thread: Bus Message from {msg_src_name}: type {msg_type.get_name()}")

        return True  # Important to return True to keep the signal watch active

    def _report_error(self, error_message: str):
        """Safely pushes an ErrorFrame back to the main asyncio loop."""
        if self._loop and self._loop.is_running():
            logger.debug(f"{self}: Scheduling ErrorFrame push: {error_message}")
            frame = ErrorFrame(f"{self}: {error_message}")
            # Ensure thread-safety when interacting with asyncio loop
            asyncio.run_coroutine_threadsafe(self.push_frame(frame), self._loop)
        else:
            logger.error(
                f"{self}: Cannot push ErrorFrame, loop not available. Error was: {error_message}"
            )

    def _stop_pipeline_sync(self):
        """Stops the GStreamer pipeline and joins the thread. Assumes lock is held."""
        if not self._pipeline_thread and not self._pipeline:
            logger.debug(f"{self} StopSync: No pipeline or thread running.")
            return

        logger.info(f"{self} StopSync: Stopping existing pipeline and thread...")

        # Stop the GLib loop first, if running
        # This should signal the _run_pipeline thread to exit its loop
        if self._glib_loop and self._glib_loop.is_running():
            logger.debug(f"{self} StopSync: Quitting GLib loop...")
            self._glib_loop.quit()
            # Don't reset self._glib_loop here, _run_pipeline does it in finally

        # Set pipeline to NULL state (this is often needed for clean shutdown)
        # It might trigger EOS/Error messages which also try to quit the loop
        if self._pipeline:
            logger.debug(f"{self} StopSync: Setting pipeline to NULL state...")
            self._pipeline.set_state(Gst.State.NULL)
            # _run_pipeline's finally block should handle setting self._pipeline = None

        # Join the thread
        if self._pipeline_thread and self._pipeline_thread.is_alive():
            logger.debug(f"{self} StopSync: Joining GStreamer thread...")
            self._pipeline_thread.join(timeout=2.0)  # Wait for thread to exit
            if self._pipeline_thread.is_alive():
                logger.warning(f"{self} StopSync: GStreamer thread did not exit after 2 seconds.")
            else:
                logger.debug(f"{self} StopSync: GStreamer thread joined successfully.")

        # Explicitly dereference sinks after thread is joined and pipeline stopped
        self._audio_sink = None
        self._video_sink = None
        self._pipeline = None  # Should be None already if thread exited cleanly
        self._pipeline_thread = None
        self._glib_loop = None  # Should be None already if thread exited cleanly

        logger.info(f"{self} StopSync: Pipeline and thread stopped.")

    async def cleanup(self):
        """Cleans up resources when the processor is destroyed."""
        logger.info(f"{self}: Cleaning up GStreamerPipelinePlayer...")
        with self._pipeline_lock:
            self._stop_pipeline_sync()  # Ensure everything is stopped and joined
        logger.info(f"{self}: Cleanup complete.")
        await super().cleanup()
