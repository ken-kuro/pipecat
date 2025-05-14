import os
import textwrap
from typing import Dict, List, TypedDict

from gst import PlayPipelineFrame
from loguru import logger
from pipecat_flows import FlowArgs, FlowManager, FlowResult, FlowsFunctionSchema, NodeConfig

from pipecat.frames.frames import ErrorFrame


class VideoConfig(TypedDict):
    filename: str  # e.g., "how-are-you-today.mp4"
    description: str  # Detailed description of the video content and purpose
    transcription: str  # Full transcription of the video's audio
    # New fields for better matching:
    pedagogical_action_tags: List[
        str
    ]  # e.g., ["greeting", "lesson_intro", "pronunciation_example", "practice_transition"]
    topic_tags: List[str]  # e.g., ["general", "insects", "animals", "colors"]
    specific_words_covered: List[str]  # e.g., ["ant", "butterfly"]


base_video_path = (
    "/home/ken-kuro/PycharmProjects/pipecat/examples/p2p-webrtc/video-transform/server/"
)

video_configs: List[VideoConfig] = [
    VideoConfig(
        filename="how-are-you-today.mp4",
        transcription="How are you today?",
        description="A general greeting video asking the user how they are.",
        pedagogical_action_tags=["greeting", "conversation_starter"],
        topic_tags=["general"],
        specific_words_covered=[],
    ),
    VideoConfig(
        filename="opening.mp4",
        transcription=textwrap.dedent("""\
            Let’s look at this picture!
            I see some tiny friends here!
            They are insects!
            Now Let’s learn together
        """),
        description="Introduces the topic of insects and prepares the child for learning.",
        pedagogical_action_tags=["lesson_intro", "topic_introduction"],
        topic_tags=["insects"],
        specific_words_covered=[],
    ),
    VideoConfig(
        filename="example.mp4",
        transcription=textwrap.dedent("""\
            This is an ant.
            a a Ant
            And look here — a beautiful butterfly!
            Let's. Say it — but--ter--fly.
        """),
        description="Shows examples of insects (ant, butterfly) and demonstrates their pronunciation.",
        pedagogical_action_tags=["pronunciation_example", "vocabulary_introduction"],
        topic_tags=["insects"],
        specific_words_covered=["ant", "butterfly"],
    ),
    VideoConfig(
        filename="practice.mp4",
        transcription=textwrap.dedent("""\
            So let's practice new insect words with Rino.
            Kids, let's listen and repeat after Rino
        """),
        description="Transitions the lesson to a practice session for new insect words.",
        pedagogical_action_tags=["practice_transition", "activity_prompt"],
        topic_tags=["insects"],
        specific_words_covered=[],
    ),
]

lesson_goal = (
    "Teach students about insects and help them practice new words like butterfly, ant, and bee."
)
retry_limit = 3


async def play_video_handler(args: FlowArgs, flow_manager: FlowManager) -> FlowResult:
    """Handler to simulate playing a video.
    Access to flow_manager state or other resources is available if needed.
    """
    logger.debug("User wants to play a video.")
    video_filename = args.get("filename")
    video_path = os.path.join(base_video_path, video_filename)
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        await flow_manager.task.queue_frame(ErrorFrame(f"Video file not found: {video_path}"))
        return {"status": "error", "error": f"Video file not found: {video_path}"}

    pipeline_desc = f'filesrc location="{video_path}"'

    logger.debug(f"Pushing PlayPipelineFrame with: {pipeline_desc[:200]}...")
    await flow_manager.task.queue_frame(PlayPipelineFrame(pipeline_description=pipeline_desc))
    logger.debug(f"Pushed PlayPipelineFrame with: {pipeline_desc[:200]}..., modifying state")
    logger.debug("Done request playing video")

    return {"status": "success"}


async def practice_transition_handler(args: FlowArgs, flow_manager: FlowManager) -> FlowResult:
    """Handler to transition to the practice section."""
    logger.debug("User wants to move to practice.")
    return {"status": "success"}


async def end_conversation_handler(args: FlowArgs, flow_manager: FlowManager) -> FlowResult:
    """Handle conversation completion."""
    logger.debug("end_conversation_handler executing")
    return {"status": "completed"}


async def transist_to_end(_: Dict, result: FlowResult, flow_manager: FlowManager):
    """Handle the end of the conversation."""
    await flow_manager.set_node("end", create_end_node())


async def transist_to_practice(_: Dict, result: FlowResult, flow_manager: FlowManager):
    """Handle the transition to the practice section."""
    await flow_manager.set_node("practice", create_practice_node())


def create_initial_node() -> NodeConfig:
    """Creates the initial node where the bot starts the conversation."""
    return {
        "role_messages": [
            {
                "role": "system",
                "content": (
                    "You are a Pedagogical Pattern Orchestrator for an AI English learning platform designed for young children.\n"
                    "Based on the lesson goal, conversation history, and assets provided, determine the most appropriate pedagogical action.\n"
                    "Your response might be converted to audio, so avoid special characters, symbols or emojis."
                ),
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": (
                    f"Lesson goal: {lesson_goal}\n"
                    f"Available videos: {video_configs}\n"
                    "Introduce the lesson, then present examples relevant, finally, move to practice.\n"
                    "If a suitable video exists, call play_video function with the filename. Otherwise, after complete all task, use move_to_practice function to move to the next section.\n"
                ),
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="play_video",
                description="Plays a video by filename.",
                properties={
                    "filename": {
                        "type": "string",
                        "description": "The filename of the video to play.",
                    }
                },
                required=["filename"],
                handler=play_video_handler,
            ),
            FlowsFunctionSchema(
                name="move_to_practice",
                description="Move to the practice section.",
                properties={},
                required=[],
                handler=practice_transition_handler,
                transition_callback=transist_to_practice,
            ),
        ],
    }


def create_practice_node() -> NodeConfig:
    """Creates the node where the bot handles practice."""
    return {
        "role_messages": [
            {
                "role": "system",
                "content": (
                    "You are a Pedagogical Pattern Orchestrator for an AI English learning platform designed for young children.\n"
                    "Based on the lesson goal, conversation history, and assets provided, determine the most appropriate pedagogical action.\n"
                    "Your response might be converted to audio, so avoid special characters, symbols or emojis."
                ),
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": (
                    f"Lesson goal: {lesson_goal}\n"
                    f"Available videos: {video_configs}\n"
                    "You will help students practice new words in this lesson here by make them listen and repeat.\n"
                    f"Encourage and correct as needed. If user do it wrong more than {retry_limit}, just summary that word and move on.\n"
                    "After all the words, conclude the lesson, summarize key points, say good bye and use end_conversation function to end the conversation.\n"
                ),
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="play_video",
                description="Plays a video by filename.",
                properties={
                    "filename": {
                        "type": "string",
                        "description": "The filename of the video to play.",
                    }
                },
                required=["filename"],
                handler=play_video_handler,
            ),
            FlowsFunctionSchema(
                name="end_conversation",
                description="Ends the conversation.",
                properties={},
                required=[],
                handler=end_conversation_handler,
                transition_callback=transist_to_end,
            ),
        ],
    }


# TODO: Fix this bug
"""
google.api_core.exceptions.InvalidArgument: 400 The GenerateContentRequest proto is invalid:
  * tools[0].tool_type: required one_of 'tool_type' must have one initialized field
"""


def create_end_node() -> NodeConfig:
    """Creates the node where the bot ends the conversation."""
    return {
        "role_messages": [
            {
                "role": "system",
                "content": (
                    "You are a friendly conversational assistant that can help user on various things. "
                    "Your response will be converted to audio, so avoid special characters, symbols or emojis. "
                    "Always wait for customer responses before calling functions. "
                ),
            }
        ],
        "task_messages": [{"role": "system", "content": "The user has ended the conversation."}],
        "functions": [],
        "post_actions": [{"type": "end_conversation"}],
    }
