import os
import textwrap
from typing import Dict, List, Literal, TypedDict

# from gst import PlayPipelineFrame
from gst_new import PlayPipelineFrame
from loguru import logger
from numba.scripts.generate_lower_listing import description
from pipecat_flows import FlowArgs, FlowManager, FlowResult, FlowsFunctionSchema, NodeConfig

from pipecat.frames.frames import ErrorFrame

base_video_path = (
    "/home/ken-kuro/PycharmProjects/pipecat/examples/p2p-webrtc/video-transform/server/"
)

PedagogicalActionTag = Literal[
    "GREETING",
    "LESSON_INTRODUCTION",
    "TOPIC_INTRODUCTION",
    "VOCABULARY_INTRODUCTION",
    "PRONUNCIATION_EXAMPLE",
    "REPETITION_PROMPT",  # e.g., "Can you say it?"
    "QUESTION_PROMPT",  # e.g., "What is this?"
    "ACTIVITY_PROMPT",  # e.g., "Let's listen and repeat"
    "PRACTICE_TRANSITION",
    "CLARIFICATION",  # e.g., saying a word slower, rephrasing
    "ENCOURAGEMENT",  # e.g., "Great job!", "Good try!"
    "AFFIRMATION",  # e.g., "That's right!"
    "CORRECTION_PROMPT",  # e.g., "Let's try again!"
    "SUMMARY",
    "GOODBYE",
    "LESSON_SEGMENT_TRANSITION",  # For moving between parts of a lesson
]

TopicTag = Literal[
    "GENERAL_CONVERSATION",  # For greetings, goodbyes, general encouragement
    "LESSON_STRUCTURE",  # For transitions, intros, outros not tied to specific content
    "INSECTS",
    "ANIMALS",
    "COLORS",
    "NUMBERS",
    "ACTIONS",
]


class VideoConfig(TypedDict):
    filename: str  # e.g., "how-are-you-today.mp4"
    description: str  # Detailed description of the video content and purpose
    transcription: str  # Full transcription of the video's audio
    # New fields for better matching:
    pedagogical_action_tags: List[PedagogicalActionTag]
    topic_tags: List[TopicTag]
    specific_words_covered: List[str]  # e.g., ["ant", "butterfly"]


video_configs: List[VideoConfig] = [
    VideoConfig(
        filename="letlookat.mp4",
        transcription=textwrap.dedent("""\
            Let’s look at this picture!
            I see some tiny friends here!
            They are insects!
            Now Let’s learn together
        """),
        description="Introduces the topic of insects and prepares the child for learning.",
        pedagogical_action_tags=["LESSON_INTRODUCTION", "TOPIC_INTRODUCTION", "ACTIVITY_PROMPT"],
        topic_tags=["INSECTS", "LESSON_STRUCTURE"],
        specific_words_covered=["insects"],
    ),
    VideoConfig(
        filename="thisisanant.mp4",
        transcription=textwrap.dedent("""\
            This is an ant.
            a a Ant
            And look here — a beautiful butterfly!
            Let's. Say it — but--ter--fly.
        """),
        description="Shows examples of insects (ant, butterfly) and demonstrates their pronunciation.",
        pedagogical_action_tags=[
            "VOCABULARY_INTRODUCTION",
            "PRONUNCIATION_EXAMPLE",
            "REPETITION_PROMPT",
        ],
        topic_tags=["INSECTS"],
        specific_words_covered=["ant", "butterfly"],
    ),
    VideoConfig(
        filename="soletpractice.mp4",
        transcription=textwrap.dedent("""\
            So let's practice new insect words with Rino.
            Kids, let's listen and repeat after Rino
        """),
        description="Transitions the lesson to a practice session for new insect words.",
        pedagogical_action_tags=["PRACTICE_TRANSITION", "ACTIVITY_PROMPT"],
        topic_tags=["INSECTS", "LESSON_STRUCTURE"],
        specific_words_covered=[],
    ),
    VideoConfig(
        filename="antcanyousay.mp4",
        transcription=textwrap.dedent("""\
            Ant! Can you say it?
        """),
        description="Shows an ant and prompts the child to say the word 'ant'.",
        pedagogical_action_tags=[
            "VOCABULARY_INTRODUCTION",
            "PRONUNCIATION_EXAMPLE",
            "REPETITION_PROMPT",
        ],
        topic_tags=["INSECTS"],
        specific_words_covered=["ant"],
    ),
    VideoConfig(
        filename="antslow.mp4",
        transcription=textwrap.dedent("""\
            Ah, this is an ant.
        """),
        description="Pronounces the word 'ant' clearly, possibly more slowly for clarification.",
        pedagogical_action_tags=[
            "VOCABULARY_INTRODUCTION",
            "PRONUNCIATION_EXAMPLE",
            "CLARIFICATION",
        ],
        topic_tags=["INSECTS"],
        specific_words_covered=["ant"],
    ),
    VideoConfig(
        filename="butterflylettry.mp4",
        transcription=textwrap.dedent("""\
            Let's try again! Butterfly
        """),
        description="Encourages another attempt and clearly pronounces 'butterfly'.",
        pedagogical_action_tags=[
            "CORRECTION_PROMPT",
            "PRONUNCIATION_EXAMPLE",
            "VOCABULARY_INTRODUCTION",
        ],
        topic_tags=["INSECTS"],
        specific_words_covered=["butterfly"],
    ),
    VideoConfig(
        filename="goodtrymoveon.mp4",
        transcription=textwrap.dedent("""\
            Ok good try! Let's move on.
        """),
        description="Encourages the child after an attempt and transitions to the next part.",
        pedagogical_action_tags=["ENCOURAGEMENT", "LESSON_SEGMENT_TRANSITION"],
        topic_tags=["LESSON_STRUCTURE"],
        specific_words_covered=[],
    ),
    VideoConfig(
        filename="greatjob.mp4",
        transcription=textwrap.dedent("""\
            Yes, great job! You did it!
        """),
        description="Praises the child for a successful attempt.",
        pedagogical_action_tags=["ENCOURAGEMENT", "AFFIRMATION"],
        topic_tags=["GENERAL_CONVERSATION"],
        specific_words_covered=[],
    ),
    VideoConfig(
        filename="greatyougotit.mp4",
        transcription=textwrap.dedent("""\
            Great! You got it!
        """),
        description="Praises the child for understanding or succeeding.",
        pedagogical_action_tags=["ENCOURAGEMENT", "AFFIRMATION"],
        topic_tags=["GENERAL_CONVERSATION"],
        specific_words_covered=[],
    ),
    VideoConfig(
        filename="insectagain.mp4",
        transcription=textwrap.dedent("""\
            I'll say it again. Insects
        """),
        description="Repeats the word 'insects' for clarification or emphasis.",
        pedagogical_action_tags=[
            "CLARIFICATION",
            "PRONUNCIATION_EXAMPLE",
            "VOCABULARY_INTRODUCTION",
        ],
        topic_tags=["INSECTS"],
        specific_words_covered=["insects"],
    ),
    VideoConfig(
        filename="insectscanyousay.mp4",
        transcription=textwrap.dedent("""\
            Insects! Can you say it?
        """),
        description="Prompts the child to say the word 'insects'.",
        pedagogical_action_tags=[
            "VOCABULARY_INTRODUCTION",
            "PRONUNCIATION_EXAMPLE",
            "REPETITION_PROMPT",
        ],
        topic_tags=["INSECTS"],
        specific_words_covered=["insects"],
    ),
    VideoConfig(
        filename="lettryagain.mp4",
        transcription=textwrap.dedent("""\
            Ok, let's try again!
        """),
        description="Encourages the child to make another attempt after a mistake.",
        pedagogical_action_tags=["ENCOURAGEMENT", "CORRECTION_PROMPT"],
        topic_tags=["LESSON_STRUCTURE"],
        specific_words_covered=[],
    ),
    VideoConfig(
        filename="slowbutterfly.mp4",
        transcription=textwrap.dedent("""\
            Here we go again. Butterfly
        """),
        description="Pronounces 'butterfly' again, possibly slower for clarity.",
        pedagogical_action_tags=[
            "CLARIFICATION",
            "PRONUNCIATION_EXAMPLE",
            "VOCABULARY_INTRODUCTION",
        ],
        topic_tags=["INSECTS"],
        specific_words_covered=["butterfly"],
    ),
    VideoConfig(
        filename="thatright.mp4",
        transcription=textwrap.dedent("""\
            That's right! You did it!
        """),
        description="Affirms the child's correct response and praises them.",
        pedagogical_action_tags=["AFFIRMATION", "ENCOURAGEMENT"],
        topic_tags=["GENERAL_CONVERSATION"],
        specific_words_covered=[],
    ),
]

lesson_goal = (
    "Teach students about insects and help them practice new words like insects, butterfly and ant."
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


async def transits_to_initial(_: Dict, result: FlowResult, flow_manager: FlowManager):
    """Handle the transition to the initial node."""
    await flow_manager.set_node("initial", create_initial_node())


async def transits_to_practice(_: Dict, result: FlowResult, flow_manager: FlowManager):
    """Handle the transition to the practice section."""
    await flow_manager.set_node("practice", create_practice_node())


async def transits_to_end(_: Dict, result: FlowResult, flow_manager: FlowManager):
    """Handle the end of the conversation."""
    await flow_manager.set_node("end", create_end_node())


def create_initial_node() -> NodeConfig:
    """Creates the initial node where the bot starts the conversation."""
    return {
        "role_messages": [
            {
                "role": "system",
                "content": (
                    "You are a Pedagogical Pattern Orchestrator for an AI English learning platform designed for young children.\n"
                    "Based on the lesson goal, conversation history, and assets provided, determine the most appropriate pedagogical action.\n"
                    "Your response might be converted to audio, so avoid special characters, symbols or emojis.\n"
                    "If you select a video to play, your response for that turn must ONLY be the function call. Do not output any text, encouragement, or narration in that turn. Do not output any text before or after the function call. Only use TTS/text if there is no suitable video for the pedagogical action.\n"
                    "Never output both a function call and a text response in the same turn. If a video is available and selected, do not output any text for that turn.\n"
                    "When you want to play a video, you must use the function/tool call interface, not text."
                ),
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": (
                    f"Lesson goal: {lesson_goal}\n"
                    f"Available videos: {video_configs}\n"
                    "Your goal is to guide the student through the lesson following this sequence: Introduce topic, present examples, then move to practice.\n"
                    "Rules for function calls:\n"
                    "- If you select a video to play, your response for that turn must ONLY be the function call. Do not output any text, encouragement, or narration in that turn.\n"
                    "- Never output both a function call and a text response in the same turn.\n"
                    "- Only use TTS/text if there is no suitable video for the pedagogical action.\n"
                    "- Never include any other text or speech in your response when you call a function.\n"
                ),
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="play_video",
                description="Plays a video by filename. When you call this function, do not include any other text or speech in your response.",
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
                transition_callback=transits_to_practice,
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
                    "Your response might be converted to audio, so avoid special characters, symbols or emojis.\n"
                    "If you select a video to play, your response for that turn must ONLY be the function call. Do not output any text, encouragement, or narration in that turn. Do not output any text before or after the function call. Only use TTS/text if there is no suitable video for the pedagogical action.\n"
                    "Never output both a function call and a text response in the same turn. If a video is available and selected, do not output any text for that turn.\n"
                    "When you want to play a video, you must use the function/tool call interface, not text."
                ),
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": (
                    f"Lesson goal: {lesson_goal}\n"
                    f"Available videos: {video_configs}\n"
                    "You are in the practice section. Help students practice new words from the lesson. You can use videos if they are suitable for practice (e.g., showing a word and its pronunciation clearly).\n"
                    f"Encourage and correct as needed. If the user makes a mistake more than {retry_limit} times for a word, summarize that word and move on.\n"
                    "After all target words are practiced, conclude the lesson, summarize key points, say goodbye, and then use the `end_conversation` function."
                    "Rules for function calls:\n"
                    "- If you select a video to play, your response for that turn must ONLY be the function call. Do not output any text, encouragement, or narration in that turn.\n"
                    "- Never output both a function call and a text response in the same turn.\n"
                    "- Only use TTS/text if there is no suitable video for the pedagogical action.\n"
                    "- Never include any other text or speech in your response when you call a function.\n"
                ),
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="play_video",
                description="Plays a video by filename. When you call this function, do not include any other text or speech in your response.",
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
                transition_callback=transits_to_end,
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
